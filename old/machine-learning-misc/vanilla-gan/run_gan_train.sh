#!/bin/bash

# TRAINING
# BATCH SIZE: 32

python gan.py --phase train --save_dir 'saved'  --epochs 1000 --model_version 0 --batch_size 32 --sample_interval 200 --save_interval 2000 --gpu_fraction 1 --gpu 0

