#!/usr/bin/env bash

BASE_DIR=

PYTHONPATH="$BASE_DIR" \
TF_CPP_MIN_LOG_LEVEL=2 \
python view-tfrecord.py \
  --vocab-path="$BASE_DIR"/common/vocab/char-13312.txt \
  --path= \
  --meta=
