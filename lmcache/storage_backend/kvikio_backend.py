import os
import glob
from pathlib import Path
from typing import Optional, Union, Dict, Any, Iterable, List

import torch
import kvikio
import kvikio.defaults

from lmcache.storage_backend.abstract_backend import LMCBackendInterface
from lmcache.utils import CacheEngineKey
from lmcache.logging import init_logger

logger = init_logger(__name__)

class KvikIOBackend(LMCBackendInterface):
    """
    LMCache backend that uses NVIDIA GDS cuFile via the kvikio library for
    efficient GPU Direct Storage operations.
    """

    def __init__(
        self,
        config: 'LMCacheEngineConfig',
        metadata: 'LMCacheEngineMetadata',
        dst_device: str = "cuda",
    ):
        """
        Initialize the KvikIO backend.
        
        Args:
            cache_dir: Directory to store cache files
            device: The device where the tensors are located
            buffer_size: Size of the buffer for I/O operations
            auto_create_dir: Whether to create the cache directory if it doesn't exist
            **kwargs: Additional arguments passed to kvikio
        """
        super().__init__()

        # Extract cache directory from config
        if config.kvikio_cache_dir is None:
            raise ValueError("kvikio_cache_dir must be specified in the config")

        self.cache_dir = Path(config.kvikio_cache_dir)
        self.device = torch.device(dst_device)
        self.buffer_size = config.kvikio_buffer_size

        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at {self.cache_dir}")
        else:
            self._clean_up()

        # Store metadata for tensor shape/dtype
        self.kv_shape = metadata.kv_shape
        self.kv_dtype = metadata.kv_dtype

        # Check if GDS is available
        try:
            # Try to create a CuFile object to check if GDS is available
            test_file = self.cache_dir / "gds_test.tmp"
            with open(test_file, "w") as f:
                f.write("test")

            try:
                with kvikio.CuFile(str(test_file), "r") as f:
                    self.gds_available = True
            except Exception:
                self.gds_available = False

            # Clean up test file
            if test_file.exists():
                os.remove(test_file)
        except Exception as e:
            logger.warning(f"Error checking GDS availability: {str(e)}")
            self.gds_available = False

        if not self.gds_available:
            print("Oh NO!!!")
            logger.warning("GDS is not available. Falling back to standard I/O path.")


    def _get_cache_path(self, key: CacheEngineKey) -> Path:
        """Get the path to the cache file for a given key."""
        return self.cache_dir / f"{key.chunk_hash}.pt"
    
    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key to check
            
        Returns:
            bool: True if the key exists, False otherwise
        """
        return self.exists(key)
    
    def get(self, key: CacheEngineKey) -> Optional[torch.Tensor]:
        """
        Get a tensor from the cache.
        
        Args:
            key: The cache key to retrieve
            
        Returns:
            Optional[torch.Tensor]: The tensor if found, None otherwise
        """
        # We need to get the shape and dtype from the file
        # For simplicity, we'll try to read the file and handle any errors
        try:
            cache_path = self._get_cache_path(key)

            if not cache_path.exists():
                logger.warning(f"Cache file for key {key} does not exist at {cache_path}")
                return None

            # Create an empty tensor with the specified shape and dtype
            tensor = torch.empty(self.kv_shape, dtype=torch.float32, device=self.device)

            # Use kvikio to read the tensor from disk
            with kvikio.CuFile(str(cache_path), "r") as f:
                bytes_read = f.read(tensor)

            expected_bytes = tensor.numel() * tensor.element_size()
            if bytes_read != expected_bytes:
                logger.error(f"Failed to read complete tensor for key {key}. "
                             f"Read {bytes_read} bytes, expected {expected_bytes}.")
                return None

            # all tensors are float32 in the cache
            tensor = tensor.to(self.kv_dtype)
            return tensor
        except Exception as e:
            logger.error(f"Error in offboarding tensor with key {key}: {str(e)}")
            return None

    def batched_get(self, keys: Iterable[CacheEngineKey]) -> Iterable[Optional[torch.Tensor]]:
        """
        Get multiple tensors from the cache.
        
        Args:
            keys: The cache keys to retrieve
            
        Returns:
            Iterable[Optional[torch.Tensor]]: The tensors if found, None for missing keys
        """
        results: List[Optional[torch.Tensor]] = []
        for key in keys:
            results.append(self.get(key))
        return results
    
    def put(self, key: CacheEngineKey, kv_chunk: torch.Tensor, blocking: bool = False) -> None:
        """
        Put a tensor into the cache.
        
        Args:
            key: The cache key to store
            kv_chunk: The tensor to store
            blocking: Whether to wait for the operation to complete
        """
        if blocking:
            # Use onboard for blocking operations
            success = self.onboard(key, kv_chunk)
            if not success:
                logger.error(f"Failed to put tensor with key {key} in blocking mode")
        else:
            # Use put_nonblocking for non-blocking operations
            self.put_nonblocking(key, kv_chunk)
    
    def put_nonblocking(self, key: CacheEngineKey, tensor: torch.Tensor) -> None:
        """
        Put a tensor into the cache without blocking.
        
        Args:
            key: The cache key to store
            kv_chunk: The tensor to store
        """
        # For simplicity, we'll use the same implementation as onboard
        # In a production implementation, we might want to use async I/O
        # or a separate thread pool to handle non-blocking writes

        try:
            cache_path = self._get_cache_path(key)

            if tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)

            # Ensure tensor is on GPU
            if not tensor.is_cuda:
                tensor = tensor.to(self.device)

            # Use kvikio to write the tensor to disk
            with kvikio.CuFile(str(cache_path), "w") as f:
                tensor = tensor.contiguous()
                bytes_written = f.write(tensor)

            expected_bytes = tensor.numel() * tensor.element_size()
            success = bytes_written == expected_bytes

            if not success:
                logger.error(f"Failed to write complete tensor for key {key}. "
                             f"Wrote {bytes_written} bytes, expected {expected_bytes}.")

        except Exception as e:
            logger.error(f"Error in onboarding tensor with key {str(cache_path)}: {str(e)}")
    
    
    def delete(self, key: CacheEngineKey) -> bool:
        """
        Delete a tensor from disk.
        
        Args:
            key: Unique identifier for the tensor
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cache_path = self._get_cache_path(key)
            
            if cache_path.exists():
                os.remove(cache_path)
                return True
            else:
                logger.warning(f"Cache file for key {key} does not exist at {cache_path}")
                return False
        except Exception as e:
            logger.error(f"Error in deleting tensor with key {key}: {str(e)}")
            return False
    
    def exists(self, key: CacheEngineKey) -> bool:
        """
        Check if a tensor exists on disk.
        
        Args:
            key: Unique identifier for the tensor
            
        Returns:
            bool: True if the tensor exists, False otherwise
        """
        cache_path = self._get_cache_path(key)
        return cache_path.exists()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the backend.
        
        Returns:
            Dict[str, Any]: Information about the backend
        """
        return {
            "backend_type": "KvikIOBackend",
            "cache_dir": str(self.cache_dir),
            "device": str(self.device),
            "buffer_size": self.buffer_size,
            "gds_available": self.gds_available,
            "kvikio_version": kvikio.__version__,
        }
    
    def close(self) -> None:
        """
        Close the backend and release any resources.
        """
        logger.info("Closing KvikIOBackend")
        self._clean_up()

    def _clean_up(self) -> None:
        # delete all cache files
        # Ensure the cache directory exists
        if not os.path.exists(self.cache_dir):
            logger.warning(f"Cache directory {self.cache_dir} does not exist. Skipping cleanup.")
            return
        # Find and delete all .pt files in the cache directory
        for file_path in glob.glob(os.path.join(self.cache_dir, "*.pt")):
            try:
                os.remove(file_path)
                logger.info(f"Deleted cache file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {str(e)}")