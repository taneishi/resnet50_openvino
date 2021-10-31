# Examples of learning and inference optimizations in PyTorch

This repository illustrates optimizations for learning and inference in PyTorch, especially when using CPU devices, 
using a straightforward MNIST dataset and a CNN model.

## Multiprocessing

When running PyTorch on a CPU, in most cases there is no need for multiprocessing; 
PyTorch will set the appropriate number of threads based on the number of CPU cores, as can be seen by running torch.get_threads_num(). 
If your cores are highly utilized, you will get better performance by disabling HyperThreading.
If you are using multiple GPUs on a single node, you should also try torch.nn.DataParallel first, 
but this will run multiple GPUs synchronously, which may result in lower utilization of each GPU. i
In this case, you can improve the utilization of each GPU by running each GPU asynchronously in a multiprocess.
The following script shows an example of multi-process execution on a single node.

- mnist_mp.py

## Distributed Data Parallel (DDP)

The advantage of using DDP is that the above multi-process execution can be done on multiple nodes. 
As seen in recent language models, training models with a large number of parameters using a large data set requires distributed processing, 
and since DDP samples the dataset and trains it on each node, the load on each distributed process becomes lightweight. 
This property is suitable for today's distributed-oriented HPC environment. The following script is an example of DDP.

- mnist_ddp.py

## Optimization and Quantization of Inference Models with OpenVINO

OpenVINO is a technology developed by Intel to perform fast inference using CPUs, 
and it has the advantage of being able to perform fast inference without using relatively expensive accelerators.
The following script uses the mode argument to perform inference in three different modes: pytorch, FP32 optimization, and INT8 quantization.

- infer.py --mode [pytorch, fp32, int8].

The procedure for model optimization and quantization using OpenVINO is described in the following script.

- convert.sh


