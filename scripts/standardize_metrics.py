import argparse
import json
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import yaml

metric_map = {
    "charts/mean_episodic_return": "mean_episodic_return",
    "charts/mean_episodic_success": "mean_episodic_success",
}
config_map = {
    "env_id": "env",
    "seed": "seed",
    "total_timesteps": "total_timesteps",
    "hidden_size": "hidden_size",
    "hidden_layers": "hidden_layers",
    "backbone": "backbone",
    "backbone_variant": "backbone_variant",
}


def extract_metrics_from_tensorboard(run_folder, output_folder):
    ea = event_accumulator.EventAccumulator(run_folder)
    ea.Reload()

    data = {metric_map[metric]: [] for metric in metric_map.keys()}
    data["steps"] = []

    for metric in metric_map.keys():
        try:
            events = ea.Scalars(metric)
        except KeyError as e:
            print("Available metrics:", ea.Tags()["scalars"])
            raise e
        for event in events:
            data[metric_map[metric]].append(event.value)
            if metric == list(metric_map.keys())[0]:  # Only add steps once
                data["steps"].append(event.step)

    df = pd.DataFrame(data)
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, "metrics.csv"), index=False)


def extract_config(run_folder, output_folder):
    config_path = os.path.join(run_folder, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        filtered_config = {
            config_map[key]: config[key] for key in config_map.keys() if key in config
        }

        if "backbone" not in filtered_config:
            filtered_config["backbone"] = "conv"

        filtered_config["algorithm"] = "PPO"
        if "env_configs" in config and config["env_configs"]:
            env_configs = json.loads(config["env_configs"])

            if filtered_config["env"] == "SlidingPuzzle-v0":
                filtered_config["env__size"] = (
                    env_configs["w"] if "w" in env_configs else env_configs["h"]
                )
                filtered_config["env__variation"] = env_configs["variation"]
                if env_configs["variation"] == "image":
                    filtered_config["env__image_folder"] = env_configs["image_folder"]

        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "config.yaml"), "w") as file:
            yaml.dump(filtered_config, file)


def process_all_experiments(
    runs_folder, output_base_folder, override_metrics, override_configs
):
    for experiment in tqdm(os.listdir(runs_folder)):
        experiment_path = os.path.join(runs_folder, experiment)
        if os.path.isdir(experiment_path):
            output_folder = os.path.join(output_base_folder, experiment)
            metrics_file = os.path.join(output_folder, "metrics.csv")
            config_file = os.path.join(output_folder, "config.yaml")
            if not os.path.exists(metrics_file) or override_metrics:
                try:
                    extract_metrics_from_tensorboard(experiment_path, output_folder)
                except Exception as e:
                    print(f"Error processing metrics for {experiment_path}: {e}")
            if not os.path.exists(config_file) or override_configs:
                try:
                    extract_config(experiment_path, output_folder)
                except Exception as e:
                    print(f"Error processing config for {experiment_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TensorBoard metrics.")
    parser.add_argument(
        "--runs_folder", type=str, default="runs", help="Folder containing the runs."
    )
    parser.add_argument(
        "--output_base_folder",
        type=str,
        default="../visualization/runs",
        help="Base folder for output.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override existing metrics and config files.",
    )
    parser.add_argument(
        "--override-metrics",
        action="store_true",
        help="Override existing metrics files.",
    )
    parser.add_argument(
        "--override-configs",
        action="store_true",
        help="Override existing config files.",
    )
    args = parser.parse_args()

    process_all_experiments(
        args.runs_folder,
        args.output_base_folder,
        args.override or args.override_metrics,
        args.override or args.override_configs,
    )
