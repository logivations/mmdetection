#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]
onnx-file=pvt_detector.onnx
model-engine-file=pvt_detector.onnx_b1_gpu0_fp16.engine

# model config
infer-dims=3;384;512
gie-unique-id=1
detector-type=RetinaNet

[custom]
labels=$CLASSES
report-labels=$CLASSES
" > "$MODEL_DIR/object-config.txt"


