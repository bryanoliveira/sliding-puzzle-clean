#!/bin/bash

for seed in 176 594 907; do # 176 267 769 907
python ppo.py --seed $seed --env_id 'ALE/Frostbite-ram-v5' --env_configs '{"frameskip": 1, "repeat_action_probability": 0}' --total_timesteps 10000000
python ppo.py --seed $seed --env_id 'ALE/Frostbite-v5' --env_configs '{"frameskip": 1, "repeat_action_probability": 0}' --total_timesteps 10000000
done
