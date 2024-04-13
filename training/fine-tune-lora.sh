#!/usr/bin/env bash

BASE_DIR=
BASE_DATA_DIR=

DATASET=
TEMPLATE=
VERSION=

PYTHONPATH="$BASE_DIR" \
TF_CPP_MIN_LOG_LEVEL=2 \
deepspeed \
  --num_gpus=3 \
  train.py \
  --deepspeed $BASE_DIR/training/ds-configs/stage2.json \
  --output_dir ckpt/FT-"$DATASET"-317M/models/v"$VERSION"-hf-lora \
  --do_train \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --fp16 \
  --learning_rate 1e-4 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.1 \
  --warmup_steps 20 \
  --max_grad_norm 1.0 \
  --gradient_checkpointing \
  --num_train_epochs 50 \
  --logging_steps 20 \
  --save_strategy epoch \
  --pretrained_model "$BASE_DATA_DIR"/base-models/PT-01-GENERIC-WSZ1024-317M/V12 \
  --training_data "$BASE_DATA_DIR"/datasets/fine-tuning/"$DATASET"/template-"$TEMPLATE" \
  --use_peft_lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --lora_target_modules c_attn,c_proj,c_fc
