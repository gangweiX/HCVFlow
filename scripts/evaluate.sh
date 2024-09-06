#!/usr/bin/env bash


# evaluate chairs & things trained model on kitti (24 iters)
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval \
--val_dataset kitti \
--resume ./pretrained_models/things.pth \
--val_iters 24


# evaluate chairs & things trained model on sintel (32 iters)
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval \
--val_dataset sintel \
--resume ./pretrained_models/things.pth \
--val_iters 32


