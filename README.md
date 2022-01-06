# Optimizing Inference with PyTorch and OpenVINO

## Getting Started

Unlike learning, which involves repeating many epochs, inference in deep learning processes the input only once, using a trained model. 
Since the weights of the model are not updated in this process, the computational load is lighter than in learning.

On the other hand, the hardware environment in which inference is performed varies, and it is possible to consider environments such as edge computing, 
where accelerators and high-speed processors represented by GPUs are not expected. 
In addition, there are situations where real-time processing is required at the edge, such as performing image recognition and inference on the spot for camera input.

OpenVINO is one of such methods, and is a toolkit provided by Intel mainly for its processor products. 
OpenVINO acceleration has two stages: model optimization and quantization. Model optimization improves the efficiency of the model by eliminating unnecessary operations in inference. 
Quantization speeds up the model by converting the weights of the floating point representation into an 8-bit integer representation. 
Unlike model optimization, quantization changes the results of inference, so accuracy verification is required. 
It is also possible to perform only model optimization.

In this section, I use image recognition as an example to show how OpenVINO can be used to speed up inference. 
For the image recognition model, I use *ResNet-50* provided by torchvision, and for the dataset, 
I use a sampling dataset with labels defined according to *ImageNet*, a dataset for image recognition.

## Execution environment

As an example of the execution environment, a setup using Python3 venv is shown below.

```bash
python3 -m venv openvino
source openvino/bin/activate
pip install --upgrade pip
pip install openvino_dev torchvision onnx
```

If this environment does not exist, `run.sh`, which will be introduced later, will automatically create it.

## Build a learning model and optimize and quantize the model

### Preparing the Trained Model

In torchvision, you can download the trained model by setting `pretrained=True`.

```python
	net = torchvision.models.resnet50(pretrained=True)
```

We will save this trained model in the portable ONNX format, since OpenVINO cannot convert the PyTorch standard model format as is. 
The default destination for the trained model is `model/resnet-50.onnx`.

```bash
python export_onnx.py
```

Once you have a trained model, you can perform inference with PyTorch. Specify `torch` with the `mode` argument. 
The default is `torch`, so you can run it without it.

```bash
python infer.py --mode torch
```

### Model Optimization

The next step is to optimize the model using OpenVINO.

If you have built the above execution environment, the script `mo` for model optimization has also been added to the execution path of venv.

```bash
mo --input_model model/resnet-50.onnx --output_dir model
```

As a result of the above script, the files `resnet-50.xml`, `resnet-50.bin`, and `resnet-50.mapping` are generated in the model directory.

To run the inference with the optimization model, set the `mode` argument of the inference script to `fp32`.

```bash
python convnet_infer.py --mode fp32
```

The mode argument currently supports only torch, fp32, and int8.

### Model Quantization

Finally, quantization is performed using the optimization model as input. 
Quantization differs from the process up to optimization in that it changes the accuracy, so it is necessary to perform calibration using validation data during the conversion.

In addition, to provide the quantization script with the correspondence between the validation data and its supervised labels, annotations are generated using the following customized annotation script. 
The generated annotations will be saved as `images.pickle` and `images.json`.

```bash
python annotation.py images --data_dir images
```

In OpenVINO quantization, the location of the optimization model, the quantization method, the location of the test data for calibration, the index, etc. are specified in a configuration file in JSON or YAML format. 
The configuration files for *ResNet-50* are saved in `resnet-50.yaml` and `pot.yaml` under config in advance. 
When the configuration files are ready, run the quantization script `pot`.

```python
pot -c config/pot.yaml
```

If the script succeeds, the post-quantization models `resnet-50.xml`, `resnet-50.bin`, and `resnet-50.mapping` will be generated under `results/resnet-50_DefaultQuantization/[date time]/optimized`. The `[date time]` is set from the date and time of the runtime.

You can move the generated quantized model file to the directory under `model/INT8` and run the inference script with the `mode` argument set to `int8` to perform inference with the quantized model.

```bash
python infer.py --mode int8
```

The operations on *ResNet-50* up to this point can be summarized in the following script.

```bash
bash run.sh
```

