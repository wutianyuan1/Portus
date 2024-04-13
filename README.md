# Portus {#mainpage}

## Intro to Portus
Portus is an efficient checkpointing system for DNN models. The core of Portus is a three-level index structure and a direct RDMA datapath that enables fast checkpoints between GPUs and persistent memory in a serialization-free way. Portus offers a zero-copy approach between GPU and persistent memory without involving main memory and kernel crossings to underlying file systems. Portus also applies an asynchronous mechanism to hide the checkpointing overhead in the model training procedures. We integrated a Portus prototype into a high-performance AI cluster with NVIDIA V100 and A40 GPUs and Intel Optane persistent memory, then evaluated its performance in both single-GPU and multi-GPU large model training scenarios. Experiment results show that compared to a state-of-the-art checkpointing system, Portus achieves up to 9.23× and 7.0× speedup in checkpointing and restoring, respectively. Portus achieves up to 2.6× higher throughput and 8× faster checkpointing operation on a large language model, GPT-22B.

## Architecture
<div align="center">
  <a href="https://sm.ms/image/Dnxod3hskgmRa8I" target="_blank"><img src="https://s2.loli.net/2024/04/14/Dnxod3hskgmRa8I.png" ></a>
</div>

## Installation
- **Client** is a pytorch extension running on GPU node, which can be installed using the following command
  ```cd client && python setup.py install```


- **Server** is a daemon program running on storage server with persistent memory, which can be built&installed by CMake:
  ```mkdir build && cd build && cmake .. && make -j```

## Dependency
- Server&Client: Infiniband/RDMA supports: `ibverbs`, `rdmacm`, `mlx5`
- Client: `Pytorch`, `nvidia-peer-mem`
