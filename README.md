# Optimizing Inference with PyTorch and OpenVINO

## Overview

Unlike training, which involves many epochs of iteration, inference in deep learning uses a pre-trained model and processes the input only once. In this case, the weights of the model are not updated, so the computational load is lighter than in training.

On the other hand, in terms of the hardware environment in which inference is performed, it includes edge computing, where expensive accelerators represented by GPUs and high performance processors cannot be expected. In addition, real-time processing is sometimes required such as image recognition application for camera input, and it would be useful to have a acceleration method specific to inference.

Among such acceleration methods for inference, OpenVINO is a toolkit provided by Intel mainly for their processor products. OpenVINO acceleration consists of two parts: model optimization and quantization. Model optimization makes models more efficient by eliminating unnecessary operations in inference. Quantization improves speed by converting the weights of the floating point representation into an 8-bit integer representation. Unlike model optimization, quantization requires accuracy calibration because it changes the results of inference.

In this repository, you will find an example of speeding up inference using OpenVINO, using image recognition with an user-defined model as an example. The user-defined model is ConvNet defined in `model.py` and the dataset used is CIFAR10.

## Environments

As an example of the environments, a Python 3 using `venv` module is shown below.

```bash
python3 -m venv openvino
source openvino/bin/activate
pip install --upgrade pip
pip install openvino_dev torchvision onnx
```

`convert.sh`, which will be introduced later, will build an environment automatically if the virtual environment does not exist.

## Build a trained model and optimize and quantize the model.

### Prepare the trained model.

An user-defined model need to be trained first. The following script will train ConvNet on the CIFAR10 dataset and save the trained model.

Since OpenVINO cannot directly convert the PyTorch model format, you need to save the model in the portable ONNX format in addition to the PyTorch format. In this process, since the model is saved with the dimension including the last batch number, it is needed to specify `drop_last=True` in `train.py` script and discard the remainder of the data. The default destination of the trained model is `model/convnet.pth` and `model/convnet.onnx`. The file names can be changed with the `name` argument.

```bash
python train.py --epochs 100
```

Now that you have a trained model, you can run PyTorch inference on it. The `mode` argument specifies `pytorch`, but it defaults to `pytorch`, so you can run it without it.

```bash
python infer.py --mode pytorch
```

### Model Optimization

The next step is to optimize the model using OpenVINO.

If you have built the above environments, the script `mo` for model optimization is also added to the execution path of venv.

```bash
mo --input_model model/convnet.onnx --output_dir model
```

As a result of the above script, the files `convnet.xml`, `convnet.bin`, and `convnet.mapping` are generated in the `model` directory.

To run inference with optimized models, set the `mode` argument of the inference script to `fp32`.

```bash
python infer.py --mode fp32
```

So far only pytorch, fp32, and int8 are supported for mode argument.

### Model Quantization

The last step is to quantize the optimized model as input. 
In CIFAR10, to give the quantization script the validation data and their labels, you can generate images and annotations using the following converter script. Here, the annotation definition of `cifar` is pre-defined in OpenVINO. It is also possible to create user-defined annotations (ref. https://github.com/taneishi/CheXNet). Created annotations are stored as `cifar.pickle` and `cifar.json` under the `annotation` directory.

```bash
mkdir -p annotation
convert_annotation cifar -o annotation --convert_images 1 \
        --data_batch_file data/cifar-10-batches-py/test_batch
```

In OpenVINO quantization, the location of the optimization model, the quantization method, the location of the data for calibration, accuracy metric, etc. must be specified in a configuration file in JSON or YAML format. You can find these config files in `config` directory. Once the configuration files are ready, run the quantization script `pot`.

```bash
pot -c config/pot.yaml
```

If the script succeeds, the quantized models `convnet.xml`, `convnet.bin`, and `convnet.mapping` will be generated under `results/convnet_DefaultQuantization/[date time]/optimized`. The [date time] is set from the date and time of execution.

You can move the generated quantized model files to `model/INT8` and run the inference script with `int8` as the `mode` argument.

```bash
python infer.py --mode int8
```

You can use the following script to perform all the operations on ConvNet.

```bash
bash convert.sh
```
