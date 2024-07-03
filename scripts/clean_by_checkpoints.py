import os
import shutil

# Directory containing the experiment runs
runs_dir = "runs"

# Iterate over each experiment folder in the runs directory
for exp_folder in os.listdir(runs_dir):
    exp_path = os.path.join(runs_dir, exp_folder)
    # INSERT_YOUR_CODE
    # Define the cutoff datetime
    cutoff_datetime = "20240516_104714"
    # Check if the experiment folder's datetime is after the cutoff
    if exp_folder < cutoff_datetime: continue

    # Check if the experiment folder is a directory
    if os.path.isdir(exp_path):
        # Check for checkpoint files in the experiment folder
        checkpoint_files = [f for f in os.listdir(exp_path) if f.startswith("checkpoint_")]
        
        # Count the number of checkpoint files
        num_checkpoint_files = len(checkpoint_files)
        
        # Check if there are less than 2 checkpoint files
        if num_checkpoint_files <= 1:
            # Delete the experiment folder
            shutil.rmtree(exp_path)
            print(f"Deleted {exp_folder} ({num_checkpoint_files} checkpoint files)")
