#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--resume ./pretrained_models/sintel.pth \
--val_iters 24 \
--inference_dir demo-imgs/horses-kids \
--output_path output/horses-kids
