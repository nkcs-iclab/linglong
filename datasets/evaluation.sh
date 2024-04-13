#!/usr/bin/env bash

DATASET=$1
BASE_DIR=
BASE_DATA_DIR=
PYTHONPATH="$BASE_DIR" \
python evaluation.py \
  --dataset="$DATASET" \
  --input-path="$BASE_DATA_DIR"/datasets/original \
  --output-path="$BASE_DATA_DIR"/datasets/evaluation \
  --dataset-config="$BASE_DIR"/evaluation/configs/local.yaml \
  --vocab-path="$BASE_DIR"/common/vocab/char-13312.txt \
  --pinyin-vocab-path="$BASE_DIR"/common/vocab/pinyin-1354.txt
