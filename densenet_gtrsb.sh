#!/bin/bash
#
#

set -e
export CUDA_VISIBLE_DEVICES=1


python3 run_gtsrb.py \
    --batch_size=64 \
    --nb_epoch=40 \
    --depth=40 \
    --nb_dense_block=3 \
    --growth_rate=12 \
    --learning_rate=0.1 \
    --logfile=l40_k12_gtsrb.json


python3 run_gtsrb.py \
    --batch_size=64 \
    --nb_epoch=40 \
    --depth=100 \
    --nb_dense_block=3 \
    --growth_rate=12 \
    --learning_rate=0.1 \
    --logfile=l100_k12_gtsrb.json


python3 run_gtsrb.py \
    --batch_size=64 \
    --nb_epoch=40 \
    --depth=100 \
    --nb_dense_block=3 \
    --growth_rate=24 \
    --learning_rate=0.1 \
    --logfile=l100_k24_gtsrb.json

