# DDP
Training large-scale deep learning models is often gated by hardware. Single-GPU setups are frequently overwhelmed by the memory requirements and compute time required for modern architectures. This repository implements Distributed Data Parallel (DDP), providing a robust framework to parallelize training across multiple GPUs.


## Distributed

Distributed training is a model training paradigm that involves spreading training workload across multiple worker nodes, therefore significantly improving the speed of training and model accuracy. While distributed training can be used for any type of ML model training, it is most beneficial to use it for large models and compute demanding tasks as deep learning.

There are a few ways you can perform distributed training in PyTorch with each method having their advantages in certain use cases:

- `DistributedDataParallel (DDP)`
- `Fully Sharded Data Parallel (FSDP2)`
- `Tensor Parallel (TP)`
- `Device Mesh`
- `Remote Procedure Call (RPC) distributed training`
- `Monarch Framework`
- `Custom Extensions`
