# Examples of inference optimizations in PyTorch

This repository illustrates optimizations for inference in PyTorch, especially when using CPU devices, 
using a straightforward MNIST dataset and a CNN model.

## Optimization and Quantization of Inference Models with OpenVINO

OpenVINO is a technology developed by Intel to perform fast inference using CPUs, 
and it has the advantage of being able to perform fast inference without using relatively expensive accelerators.

The following script uses the mode argument to perform inference in three different modes: pytorch, FP32 optimization, and INT8 quantization.

- infer.py --mode [pytorch, fp32, int8].

The procedure for model optimization and quantization using OpenVINO is described in the following script.

- convert.sh
