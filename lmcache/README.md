## GPU Direct Storage (GDS)

GDS support has been added using the NVIDIA Rapids based kvikio library. New configuration options have been added to support kvikio.

The configuration file entry looks something like this:

```
>> cat /config/lmcache_config.yaml 
max_local_cache_size: 1
chunk_size: 256
pipelined_backend: False
save_decode_cache: True
kvikio_cache_dir: "/mnt/nvme/lmc" <<== GDS location
```

In order to use GDS ensure that the libcufile.so library is available.

As an example, a local NVMe drive can be made available to the container using:

```
docker run ... -v /run/udev:/run/udev:ro ...
```

Inside the container, create the mount point, and mount the NVMe drive:

```
mkdir /mnt/nvme
mount -o data=ordered /dev/nvme0n1 /mnt/nvme
```

Install the kvikio package in the container:

https://docs.rapids.ai/api/kvikio/nightly/install/
