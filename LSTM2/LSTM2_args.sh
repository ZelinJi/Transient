#!/bin/bash
PYTHON=/home/zj303/.conda/envs/gpu/bin/python
GPUID=$1
batch_size=$2
lr=$3

output=./logs/Log_bsize_${batch_size}_lr_${lr_main}.log
echo saving log to $output
TF_CPP_MIN_LOG_LEVEL="3" \
CUDA_VISIBLE_DEVICES=$GPUID \
      $PYTHON matmodel.py \
      --batch_size $batch_size\
      --lr $lr > $output