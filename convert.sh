#!/bin/bash

if [ ! -d openvino ]
then
    python3 -m venv openvino
    source openvino/bin/activate
    pip install --upgrade pip
    pip install openvino_dev torchvision onnx
else
    source openvino/bin/activate
fi

if [ ! -f model/convnet.onnx ]
then
    python train.py --epochs 100
fi

python infer.py --mode pytorch

mo --input_model model/convnet.onnx --output_dir model
python infer.py --mode fp32

mkdir -p annotation
convert_annotation cifar -o annotation --convert_images 1 \
        --data_batch_file data/cifar-10-batches-py/test_batch

pot -c config/pot.yaml
mkdir -p model/INT8
cp $(ls results/convnet_DefaultQuantization/*/optimized/* | tail -3) model/INT8
python infer.py --mode int8
