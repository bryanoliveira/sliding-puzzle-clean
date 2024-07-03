import json
import os
import yaml
import datetime
import re

def generate_new_name(config, old_name):
    # Extract date and time from the old name
    date_time_match = re.match(r'(\d{8}_\d{6})', old_name)
    if date_time_match:
        date_time = date_time_match.group(1)
    else:
        raise ValueError(f"Can't extract date and time from {old_name}")

    exp_name = f"{old_name.split('-')[1].split('_slidingpuzzle')[0]}"

    if type(config['env_configs']) != dict:
        config['env_configs'] = json.loads(config['env_configs']) if config['env_configs'] else {}

    if config['env_id'] != "SlidingPuzzle-v0":
        exp_name += "_" + config['env_id'].replace("/", "").replace("-", "").lower()
    
    if "w" in config['env_configs']:
        exp_name += f"_w{config['env_configs']['w']}"
    if "variation" in config['env_configs']:
        exp_name += f"_{config['env_configs']['variation']}"
    if "image_folder" in config['env_configs']:
        exp_name += f"_{config['env_configs']['image_folder'].replace('/', '').replace('-', '').lower()}"
    if "image_pool_size" in config['env_configs']:
        exp_name += f"_p{config['env_configs']['image_pool_size']}"
    
    run_name = f"{date_time}-{exp_name}_{config['seed']}"
    return exp_name, run_name

def rename_runs(runs_folder):
    for folder_name in sorted(os.listdir(runs_folder)):
        folder_path = os.path.join(runs_folder, folder_name)
        if os.path.isdir(folder_path) and "slidingpuzzle" in folder_name:
            config_path = os.path.join(folder_path, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                exp_name, run_name = generate_new_name(config, folder_name)
                
                config['exp_name'] = exp_name
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                print(f"Updated exp_name in config for {folder_name}: {exp_name}")

                new_path = os.path.join(runs_folder, run_name)
                if folder_name != run_name:
                    os.rename(folder_path, new_path)
                    print(f"Renamed: {folder_name} -> {run_name}")
            else:
                print(f"Skipped: {folder_name} (no config.yaml found)")
        else:
            print(f"Skipped: {folder_name} (not a slidingpuzzle run)")

if __name__ == "__main__":
    runs_folder = "runs"
    rename_runs(runs_folder)

