#!/usr/bin/env bash

TRAIN_FILE=../../data/train.txt
TEST_FILE=../../data/eval.txt

CUDA_VISIBLE_DEVICE=0,1,2,3,4,5 python run_language_modeling.py \
    --output_dir=alignement3 \
    --model_type=roberta \
    --tokenizer_name=./alignement \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --line_by_line \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1