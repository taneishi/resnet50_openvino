#!/bin/bash
#PBS -l nodes=2:gold6330
#PBS -N torch_ddp
#PBS -j oe
#PBS -o output_${PBS_NUM_NODES}.log

if [ ${PBS_O_WORKDIR} ]; then
    cd ${PBS_O_WORKDIR}
fi

if [ ! -d images ]; then
    git clone https://github.com/EliSchwartz/imagenet-sample-images images
fi

if [ ! -f imagenet_classes.txt ]; then
    wget -c https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
fi

CPUS=2
CORES=24
TOTAL_CORES=$((${CPUS}*${CORES}))

echo "CPUS=${CPUS} CORES=${CORES} TOTAL_CORES=${TOTAL_CORES}"
export OMP_NUM_THREADS=${TOTAL_CORES}
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

pip install -q -r requirements.txt

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

python main.py
python main.py --mode fp32
python main.py --mode int8
