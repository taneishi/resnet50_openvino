#!/bin/bash

#source openvino/bin/activate

omz_downloader --name resnet-50-pytorch

omz_converter --name resnet-50-pytorch

# to change input batch size instead of default 1.
#python openvino/lib/python3.8/site-packages/open_model_zoo/model_tools/internal_scripts/pytorch_to_onnx.py \
#        --model-name=resnet50 --weights=public/resnet-50-pytorch/resnet50-19c8e357.pth --import-module=torchvision.models \
#        --input-shape=96,3,224,224 --output-file=public/resnet-50-pytorch/resnet-v1-50.onnx --input-names=data --output-names=prob
#
#mo --data_type=FP16 --output_dir=public/resnet-50-pytorch/FP16 --model_name=resnet-50-pytorch \
#        --input=data '--mean_values=data[123.675,116.28,103.53]' '--scale_values=data[58.395,57.12,57.375]' \
#        --reverse_input_channels --output=prob --input_model=public/resnet-50-pytorch/resnet-v1-50.onnx
#
#mo --data_type=FP32 --output_dir=public/resnet-50-pytorch/FP32 --model_name=resnet-50-pytorch \
#        --input=data '--mean_values=data[123.675,116.28,103.53]' '--scale_values=data[58.395,57.12,57.375]' \
#        --reverse_input_channels --output=prob --input_model=public/resnet-50-pytorch/resnet-v1-50.onnx

