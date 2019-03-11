#!/bin/bash
#
#

set -e
export CUDA_VISIBLE_DEVICES=0


python3 run_cifar10.py \
    --batch_size=64 \
    --nb_epoch=40 \
    --depth=40 \
    --nb_dense_block=3 \
    --growth_rate=12 \
    --learning_rate=0.1 \
    --logfile=l40_k12_cifar10.json


python3 run_cifar10.py \
    --batch_size=64 \
    --nb_epoch=40 \
    --depth=100 \
    --nb_dense_block=3 \
    --growth_rate=12 \
    --learning_rate=0.1 \
    --logfile=l100_k12_cifar10.json


python3 run_cifar10.py \
    --batch_size=64 \
    --nb_epoch=40 \
    --depth=100 \
    --nb_dense_block=3 \
    --growth_rate=24 \
    --learning_rate=0.1 \
    --logfile=l100_k24_cifar10.json

