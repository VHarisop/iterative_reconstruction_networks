import argparse
import datetime
import itertools
import os
from typing import Any, Dict

from simple_slurm import Slurm

from utils.parsing import experiment_config_from_yaml_file
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
        "--solver_param_config_file", help="Path to the solver param config file", type=str, required=True,
    )
    parser.add_argument(
        "--common_param_config_file", help="Path to the common param config file", type=str, required=True,
    )
    parser.add_argument(
        "--data_folder", help="Root folder for the dataset", type=str, required=True,
    )
    parser.add_argument(
        "--save_location", help="Folder to store network weights in", type=str, required=True,
    )
    parser.add_argument(
        "--use_cuda", help="Set to use CUDA acceleration if available", action="store_true",
    )
    return parser.parse_args()


def validate_common_experiment_config(config: Dict[str, Any]):
    required_keys = {
        "num_train_samples",
        "num_epochs",
        "num_solver_iterations",
        "kernel_size",
        "batch_size",
        "learning_rate",
        "save_frequency",
        "algorithm_step_size",
    }
    missing_keys = required_keys - set(config)
    if len(missing_keys) > 0:
        raise ValueError(f"The following keys are missing: {missing_keys}")


def run_sweep():
    args = setup_args()
    common_param_configs = experiment_config_from_yaml_file(args.common_param_config_file)
    solver_param_configs = experiment_config_from_yaml_file(args.solver_param_config_file)
    for (param_config, solver_config) in itertools.product(common_param_configs, solver_param_configs):
        validate_common_experiment_config(param_config)
        # Set up experiment
        experiment_config = {**param_config, **solver_config}
        experiment_id = hash_dict(experiment_config)
        job_name = f"{args.wandb_project_name}_{experiment_id}"
        # Create a slurm job
        slurm_job = Slurm(
            cpus_per_task=1,
            nodes=1,
            ntasks=1,
            gres="gpu:1",
            job_name=job_name,
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
        # Extract the subcommand and subcommand-specific params
        solver = solver_config.pop("solver")
        solver_param_str = " ".join("--{k}={v}".format(k=k, v=v) for k, v in solver_config.items())
        slurm_job.sbatch(
            f"""PYTHONPATH=$(pwd) python {args.path_to_script} \
            --wandb_mode={args.wandb_mode} \
            --wandb_entity={args.wandb_entity} \
            --wandb_project_name={args.wandb_project_name} \
            --data_folder={args.data_folder} \
            --save_location={args.save_location} \
            {"--use_cuda" if args.use_cuda else ""} \
            --verbose \
            --num_train_samples={experiment_config["num_train_samples"]} \
            --num_epochs={experiment_config["num_epochs"]} \
            --num_solver_iterations={experiment_config["num_solver_iterations"]} \
            --kernel_size={experiment_config["kernel_size"]} \
            --batch_size={experiment_config["batch_size"]} \
            --learning_rate={experiment_config["learning_rate"]} \
            --save_frequency={experiment_config["save_frequency"]} \
            --algorithm_step_size={experiment_config["algorithm_step_size"]} \
            {solver} {solver_param_str}""",
        )


if __name__ == "__main__":
    run_sweep()
