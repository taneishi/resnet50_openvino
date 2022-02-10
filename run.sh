#!/bin/bash

if [ -d openvino ]; then
    source openvino/bin/activate
else
    python3 -m venv openvino
    source openvino/bin/activate
    pip install --upgrade pip
    pip install openvino_dev torchvision onnx
fi

if [ ! -d images ]; then
    git clone https://github.com/EliSchwartz/imagenet-sample-images images
fi

if [ ! -f imagenet_classes.txt ]; then
    wget -c https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
fi

if [ ! -f model/resnet-50.onnx ]; then
    python export_onnx.py
fi

if [ ! -f images.json ]; then
    python annotation.py images --data_dir images
fi

if [ ! -f model/resnet-50.xml ]; then 
    mo --input_model model/resnet-50.onnx --output_dir model
fi

if [ ! -f model/INT8/resnet-50.xml ]; then
    pot -c config/pot.yaml
    mkdir -p model/INT8
    cp $(ls results/resnet-50_DefaultQuantization/*/optimized/* | tail -3) model/INT8
fi

python infer.py
python infer.py --mode fp32
python infer.py --mode int8
