#!/usr/bin/env bash

BASE_DIR=
BASE_DATA_DIR=

DATASET="$1"

PYTHONPATH="$BASE_DIR" \
TF_CPP_MIN_LOG_LEVEL=2 \
accelerate launch \
  --num_machines 1 \
  --num_processes 3 \
  --dynamo_backend no \
  --mixed_precision fp16 \
  evaluate.py \
  --per_device_batch_size 16 \
  --dataset "$DATASET" \
  --input_path "$BASE_DATA_DIR"/datasets/original \
  --output_path eval \
  --cache_path "$BASE_DATA_DIR"/datasets/evaluation \
  --dataset_config "$BASE_DIR"/evaluation/configs/local.yaml \
  --use_cache 0 \
  --vocab_path="$BASE_DIR"/common/vocab/char-13312.txt \
  --pinyin_vocab_path="$BASE_DIR"/common/vocab/pinyin-1354.txt
