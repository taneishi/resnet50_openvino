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
    python convnet.py --epochs 100
fi

python convnet_infer.py --mode pytorch

mo --input_model model/convnet.onnx --output_dir model
python convnet_infer.py --mode fp32

mkdir -p annotation
convert_annotation mnist -o annotation --convert_images 1 \
        --labels_file data/MNIST/raw/t10k-labels-idx1-ubyte.gz \
        --images_file data/MNIST/raw/t10k-images-idx3-ubyte.gz

pot -c config/pot.yaml
mkdir -p model/INT8
cp $(ls results/convnet_DefaultQuantization/*/optimized/* | tail -3) model/INT8
python convnet_infer.py --mode int8
