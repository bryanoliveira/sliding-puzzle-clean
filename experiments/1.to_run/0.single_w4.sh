#!/bin/bash

for seed in 176 267 594 769 907; do
python ppo.py --seed $seed --env_configs '{"w": 2, "variation": "image", "image_folder": "single"}' --hidden_size 1024 --hidden_layers 5 --total_timesteps 20000000
done

