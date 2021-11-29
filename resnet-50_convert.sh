#!/bin/bash

#source openvino/bin/activate

omz_downloader --name resnet-50-pytorch

omz_converter --name resnet-50-pytorch

convert_annotation cifar -o annotation --data_batch_file data/cifar-10-batches-py/test_batch
