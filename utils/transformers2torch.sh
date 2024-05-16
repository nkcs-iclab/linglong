#!/usr/bin/env bash

BASE_DIR=
MODEL=$1

PYTHONPATH="$BASE_DIR" \
CUDA_VISIBLE_DEVICES=None \
TF_CPP_MIN_LOG_LEVEL=2 \
python convert-model.py \
  --src-type=transformers \
  --dst-type=torch \
  --src-model="$MODEL" \
  --dst-model-config="$BASE_DIR"/common/compat/model-configs/317M-WSZ1024L24.yaml
