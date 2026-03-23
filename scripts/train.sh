#!/bin/bash
# Example training script for AxisSym
# Adjust --nproc_per_node to match the number of available GPUs

torchrun --nproc_per_node=4 train.py --cfg configs/train_dendi.py
