import os
import yaml

def scan_runs_folder(runs_folder):
    for root, dirs, files in os.walk(runs_folder):
        if 'config.yaml' in files:
            config_path = os.path.join(root, 'config.yaml')
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                if config.get('checkpoint_load_path') is not None:
                    print(config_path)

if __name__ == "__main__":
    runs_folder = 'cleanrl/runs'
    scan_runs_folder(runs_folder)
