#!/bin/bash

if [ ! -d openvino ]
then
    python3 -m venv openvino
    source openvino/bin/activate
    pip install --upgrade pip
    pip install openvino_dev torchvision onnx==1.8.1
else
    source openvino/bin/activate
fi

if [ ! -f model/convnet.onnx ]
then
    python mnist.py --epochs 100
fi

python mnist_infer.py --mode pytorch

mo --input_model model/convnet.onnx --output_dir model
python mnist_infer.py --mode fp32

mkdir -p annotation
convert_annotation mnist_csv --annotation_file data/MNIST/val.txt -o annotation

pot -c config/pot.yaml
mkdir -p model/INT8
cp $(ls results/convnet_DefaultQuantization/*/optimized/* | tail -3) model/INT8
python mnist_infer.py --mode int8
