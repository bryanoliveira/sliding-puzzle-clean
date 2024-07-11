import subprocess
import json
import concurrent.futures


def run_single_seed(image_pool_size):
    cmd = [
        "python", "ppo.py",
        "--env_configs", json.dumps(
            {
                "w": 3,
                "variation": "image",
                "image_folder": "imagenet-1k",
                "image_pool_size": int(image_pool_size),
            }
        ),
        # "--num_envs", "1",
        # "--num_steps", "32",
        # "--total_timesteps", "10000",
        # "--no_track",
    ]

    print(f"---- Running command: {cmd}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

    # Extract metrics from the last line of stdout
    last_line = result.stdout.strip().split("\n")[-1]
    try:
        metrics = json.loads(last_line)
        return metrics.get("success_rate", 0.0)
    except json.JSONDecodeError as e:
        print(f"---- STDOUT: {result.stdout}")
        print(f"---- STDERR: {result.stderr}")
        raise ValueError(f"Couldn't parse metrics from output: {last_line}. Error: {e}")


def run_ppo(image_pool_size, n_seeds=3):
    # success_rates = [run_single_seed(image_pool_size)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        success_rates = list(executor.map(run_single_seed, [image_pool_size] * n_seeds))

    # Calculate average success rate
    avg_success_rate = sum(success_rates) / len(success_rates)
    return avg_success_rate


def binary_search(difficulty):
    lower_bound = 1
    upper_bound = difficulty
    target_success_rate = 0.3

    while lower_bound <= upper_bound:
        success_rate = run_ppo(upper_bound)
        print(f"---- Image pool size: {upper_bound}, Success rate: {success_rate}")

        if success_rate < target_success_rate:
            upper_bound /= 2
            print(f"---- Decreasing upper bound to {upper_bound}")
        else:
            lower_bound = upper_bound
            upper_bound *= 2
            print(f"---- Increasing lower bound to {lower_bound}, upper bound to {upper_bound}")

    return upper_bound, lower_bound


if __name__ == "__main__":
    print("---- Running base experiment")
    base_success_rate = run_ppo(1)
    print(f"---- Base success rate: {base_success_rate}")
    if base_success_rate < 0.3:
        print("---- Base success rate is less than 0.3. Exiting.")
        exit(1)
    print("Found: ", binary_search(1024))

