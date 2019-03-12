#!/bin/bash
#
#

set -e
export CUDA_VISIBLE_DEVICES=0

NUM_CLASSES=10
IMAGE_SIZE=32
BATCH_SIZE=32

python3 profile_densenet.py \
    --nb_classes=${NUM_CLASSES} \
    --image_size=${IMAGE_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --nb_epoch=40 \
    --depth=40 \
    --nb_dense_block=3 \
    --growth_rate=12 \
    --learning_rate=0.1


python3 profile_densenet.py \
    --nb_classes=${NUM_CLASSES} \
    --image_size=${IMAGE_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --nb_epoch=40 \
    --depth=100 \
    --nb_dense_block=3 \
    --growth_rate=12 \
    --learning_rate=0.1

python3 profile_densenet.py \
    --nb_classes=${NUM_CLASSES} \
    --image_size=${IMAGE_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --nb_epoch=40 \
    --depth=100 \
    --nb_dense_block=3 \
    --growth_rate=24 \
    --learning_rate=0.1
