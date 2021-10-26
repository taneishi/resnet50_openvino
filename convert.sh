#!/bin/bash

#python3 -m venv openvino
source openvino/bin/activate
#pip install openvino_dev onnx==1.8.1

#python mnist.py
mo --input_model model/convnet.onnx --output_dir model
