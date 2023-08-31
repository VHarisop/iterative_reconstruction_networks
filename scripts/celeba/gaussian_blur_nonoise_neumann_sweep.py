import argparse
import datetime
import os
from typing import Any, Dict

from simple_slurm import Slurm

from utils.parsing import experiment_config_from_yaml_file, setup_common_parser
from utils.train_utils import hash_dict


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a neumann network training sweep")
    parser.add_argument(
        "--path_to_script",
        help="Path to the script (default: scripts/celeba/gaussian_blur_nonoise_neumann.py)",
        default="scripts/celeba/gaussian_blur_nonoise_neumann.py",
    )
    parser.add_argument(
        "--slurm_log_folder_base",
        help="Path to the folder where Slurm should log output and stderr",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        help="The wandb entity (default: vchariso-lab)",
        type=str,
        default="vchariso-lab",
    )
    parser.add_argument(
        "--wandb_project_name", help="The wandb project name", type=str, required=True
    )
    parser.add_argument(
        "--wandb_mode", choices=["online", "offline", "disabled"], default="offline"
    )
    parser.add_argument(
        "--config_file", help="Path to the configuration file", type=str
    )
    return parser.parse_args()


def validate_experiment_config(config: Dict[str, Any]):
    required_keys = {
        "data_folder",
        "num_train_samples",
        "num_epochs",
        "num_solver_iterations",
        "kernel_size",
        "batch_size",
        "learning_rate",
        "save_frequency",
        "save_location",
        "verbose",
        "use_cuda",
        "algorithm_step_size",
    }
    missing_keys = required_keys - set(config)
    if len(missing_keys) > 0:
        raise ValueError(f"The following keys are missing: {missing_keys}")


def run_sweep():
    args = setup_args()
    experiment_configs = experiment_config_from_yaml_file(args.config_file)
    for config in experiment_configs:
        validate_experiment_config(config)
        # Set up experiment
        experiment_id = hash_dict(config)
        job_name = f"{args.wandb_project_name}_{experiment_id}"
        # Create a slurm job
        slurm_job = Slurm(
            cpus_per_task=1,
            nodes=1,
            ntasks=1,
            gres="gpu:1",
            job_name=job_name,
            time=datetime.timedelta(hours=4),
            mem="8G",
            mail_type="FAIL",
            mail_user="vchariso@uchicago.edu",
            output=os.path.join(
                args.slurm_log_folder_base,
                f"{job_name}_{Slurm.JOB_ID}.out",
            ),
            error=os.path.join(
                args.slurm_log_folder_base,
                f"{job_name}_{Slurm.JOB_ID}.err",
            ),
            requeue="",
        )
        slurm_job.sbatch(
            f"""PYTHONPATH=$(pwd) python {args.path_to_script} \
            --wandb_mode={args.wandb_mode} \
            --wandb_entity={args.wandb_entity} \
            --wandb_project_name={args.wandb_project_name} \
            --data_folder={config["data_folder"]} \
            --num_train_samples={config["num_train_samples"]} \
            --num_epochs={config["num_epochs"]} \
            --num_solver_iterations={config["num_solver_iterations"]} \
            --kernel_size={config["kernel_size"]} \
            --batch_size={config["batch_size"]} \
            --learning_rate={config["learning_rate"]} \
            --log_file_location={config["log_file_location"]} \
            --save_frequency={config["save_frequency"]} \
            --save_location={config["save_location"]} \
            {"--verbose" if config["verbose"] else ""} \
            {"--use_cuda" if config["use_cuda"] else ""} \
            --algorithm_step_size={config["algorithm_step_size"]}""",
            shell="/bin/bash",
        )


if __name__ == "__main__":
    run_sweep()
