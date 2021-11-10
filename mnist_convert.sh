#!/bin/bash

#python3 -m venv openvino
source openvino/bin/activate
#pip install --upgrade pip
#pip install openvino_dev torchvision onnx==1.8.1

python mnist.py --epochs 100

python infer.py --mode pytorch

mo --input_model model/convnet.onnx --output_dir model
python infer.py --mode fp32

mkdir -p annotation
convert_annotation mnist_csv --annotation_file val.txt -o annotation

pot -c config/pot.yaml
cp $(ls results/convnet_DefaultQuantization/*/optimized/* | tail -3) model/INT8
python infer.py --mode int8
