#!/usr/bin/env bash

DATASET=$1
BASE_DIR=
BASE_DATA_DIR=

PYTHONPATH="$BASE_DIR" \
TF_CPP_MIN_LOG_LEVEL=2 \
python pre-training.py \
  --dataset="$DATASET" \
  --model-config="$BASE_DIR"/common/model-configs/317M-WSZ1024L24.json \
  --input-path="$BASE_DATA_DIR"/datasets/original \
  --output-path="$BASE_DATA_DIR"/datasets/pre-training \
  --vocab-path="$BASE_DIR"/common/vocab/char-13312.txt \
  --pinyin-vocab-path="$BASE_DIR"/common/vocab/pinyin-1354.txt
