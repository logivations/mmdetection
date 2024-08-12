#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]
gpu-id=0

# preprocessing parameters.
net-scale-factor=0.01735207357279195
offsets=123.675;116.28;103.53
model-color-format=0
scaling-filter=3 # 0=Nearest, 1=Bilinear 2=VIC-5 Tap interpolation 3=VIC-10 Tap interpolation

onnx-file=pvt_detector.onnx
model-engine-file=pvt_detector.onnx_b1_gpu0_fp16.engine

# model config
infer-dims=3;384;512
batch-size=1
network-mode=2 # 0=FP32, 1=INT8, 2=FP16
network-type=100 # >3 disables post-processing
cluster-mode=4 # 1=DBSCAN 4=No Clustering
gie-unique-id=1
output-tensor-meta=1
tensor-meta-pool-size=200

[custom]
min_confidence = 0.5
labels=$CLASSES
report_labels=$CLASSES
" > "$MODEL_DIR/object-config.txt"


