## PyTorch Distributed Overview

The PyTorch Distributed library includes a collective of parallelism modules, a communications layer, and infrastructure for launching and debugging large training jobs.

**Parallelism APIs**

These Parallelism Modules offer high-level functionality and compose with existing models:

- `Distributed Data-Parallel (DDP)`
- `Fully Sharded Data-Parallel Training (FSDP2)`
- `Tensor Parallel (TP)`
- `Pipeline Parallel (PP)`


## Sharding primitives

`DTensor` and `DeviceMesh` are primitives used to build parallelism in terms of sharded or replicated tensors on N-dimensional process groups.

- `DTensor` represents a tensor that is sharded and/or replicated, and communicates automatically to reshard tensors as needed by operations.
- `DeviceMesh` abstracts the accelerator device communicators into a multi-dimensional array, which manages the underlying ProcessGroup instances for collective communications in multi-dimensional parallelisms. 


##  Launcher

`torchrun` is a widely-used launcher script, which spawns processes on the local and remote machines for running distributed PyTorch programs.


## Applying Parallelism To Scale Your Model

`Data Parallelism` is a widely adopted single-program multiple-data training paradigm where the model is replicated on every process, every model replica computes local gradients for a different set of input data samples, gradients are averaged within the data-parallel communicator group before each optimizer step.

`Model Parallelism` techniques (or Sharded Data Parallelism) are required when a model doesn’t fit in GPU, and can be combined together to form multi-dimensional (N-D) parallelism techniques.


When deciding what parallelism techniques to choose for your model, use these common guidelines:

1. Use `DistributedDataParallel (DDP)`, if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.
   - Use `torchrun`, to launch multiple pytorch processes if you are using more than one node.
2. Use `FullyShardedDataParallel (FSDP2)` when your model cannot fit on one GPU.
3. Use `Tensor Parallel (TP)` and/or `Pipeline Parallel (PP)` if you reach scaling limitations with FSDP2.




