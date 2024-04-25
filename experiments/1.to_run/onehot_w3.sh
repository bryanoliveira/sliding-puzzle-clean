#!/bin/bash

for seed in 176 267 594 769 907; do
python ppo.py --seed $seed --env_configs '{"w": 3, "variation": "onehot"}' --hidden_size 512 --hidden_layers 2
done