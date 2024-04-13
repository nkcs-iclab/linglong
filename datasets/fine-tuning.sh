#!/usr/bin/env bash

DATASET=$1
SPLIT=${2:-train}
BASE_DIR=
BASE_DATA_DIR=

PYTHONPATH="$BASE_DIR" \
TF_CPP_MIN_LOG_LEVEL=2 \
python fine-tuning.py \
  --dataset="$DATASET" \
  --split="$SPLIT" \
  --model-config="$BASE_DIR"/common/model-configs/317M-WSZ1024L24.json \
  --input-path="$BASE_DATA_DIR"/datasets/original \
  --output-path="$BASE_DATA_DIR"/datasets/fine-tuning \
  --dataset-config="$BASE_DIR"/datasets/configs/fine-tuning/local.yaml \
  --vocab-path="$BASE_DIR"/common/vocab/char-13312.txt \
  --pinyin-vocab-path="$BASE_DIR"/common/vocab/pinyin-1354.txt
