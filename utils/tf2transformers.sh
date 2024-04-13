#!/usr/bin/env bash

BASE_DIR=
MODEL=$1

PYTHONPATH="$BASE_DIR" \
CUDA_VISIBLE_DEVICES=None \
TF_CPP_MIN_LOG_LEVEL=2 \
TF_USE_LEGACY_KERAS=1 \
python convert-model.py \
  --src-type=tensorflow \
  --dst-type=transformers \
  --src-model="$MODEL" \
  --dst-model-config="$BASE_DIR"/common/model-configs/317M-WSZ1024L24.json \
  --src-model-config="$BASE_DIR"/common/legacy-model-configs/317M-WSZ1024L24.yaml \
  --vocab-path="$BASE_DIR"/common/vocab/char-13312.txt
