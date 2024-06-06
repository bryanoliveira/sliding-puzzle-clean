#!/bin/bash

for seed in 176 594 907; do # 267 769
python ppo.py --seed $seed --env_configs '{"w": 3, "variation": "image", "image_folder": "imagenet-1k", "image_pool_size": 1}' --total_timesteps 10000000 --early_stop_patience 100
done

