#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]
onnx-file=pvt_detector.onnx

# model config
infer-dims=3;384;512
gie-unique-id=1

[custom]
detector-type=1
labels=$CLASSES
report-labels=$CLASSES
" > "$MODEL_DIR/object-config.txt"
