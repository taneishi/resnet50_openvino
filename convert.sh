#!/bin/bash

#python3 -m venv openvino
source openvino/bin/activate
#pip install --upgrade pip
#pip install openvino_dev torchvision onnx==1.8.1

#python mnist.py
python infer.py

mo --input_model model/convnet.onnx --output_dir model
python infer.py --mode openvino
