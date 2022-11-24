# GPU-RPMA

## Intro to GPU-RPMA
GPU-RPMA is designed for distributed DNN training scenarios, enabling users to do iteration-based fine-grained checkpointing without slowing down the training process. In this scenario, each client is a user’s DNN training job, while the server is a storage pool shared by all jobs. GPU-RPMA leverages the high I/O performance of persistent memories and the InfiniBand network. It enables the user to do fine-grained checkpointing in each iteration with barely zero overhead. We evaluate GPU-RPMA on 76 widely-used DNN models. Experiments show GPU-RPMA accelerates checkpointing up to 9.23×, restoring up to 7.0× to current methods. Also, our evaluation shows it only introduces 0.0029% overhead to training and 1.90× higher throughput than state-of-the-art checkpointing system in a multi-tenant training scenario.

## Architecture
[![arch.png](https://i.postimg.cc/KzsbHTd5/arch.png)](https://postimg.cc/5YvDHX9H)

## Installation
- **Client** is a pytorch extension running on GPU node, which can be installed using the following command
  ```cd client && python setup.py install```
  
- **Server** is a daemon program running on storage server with persistent memory, which can be built&installed by CMake:
  ```mkdir build && cd build && cmake .. && make -j```

## Dependency
- Server&Client: Infiniband/RDMA supports: `ibverbs`, `rdmacm`, `mlx5`
- Client: `Pytorch`, `nvidia-peer-mem`
