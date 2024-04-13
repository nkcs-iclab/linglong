#!/usr/bin/env bash

BASE_DIR=

PYTHONPATH="$BASE_DIR" \
TF_CPP_MIN_LOG_LEVEL=2 \
python generate.py \
  --vocab-path="$BASE_DIR"/common/vocab/char-13312.txt \
  --pinyin-vocab-path="$BASE_DIR"/common/vocab/pinyin-1354.txt \
  --model=
