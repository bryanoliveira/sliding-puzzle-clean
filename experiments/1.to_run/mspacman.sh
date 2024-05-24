#!/bin/bash

for seed in 176 594 907; do # 176 267 769 907
python ppo.py --seed $seed --env_id 'MsPacman-ramNoFrameskip-v4'
python ppo.py --seed $seed --env_id 'MsPacmanNoFrameskip-v4'
done
