import argparse
import itertools
import os
from typing import Dict, Iterable

import yaml


def experiment_config_from_yaml_file(filename: str) -> Iterable[Dict]:
    """Create a sequence of experiment configurations from a YAML file.

    Args:
        filename: Path to the YAML file.

    Returns:
        A generator of dictionaries containing experiment configurations.
    """
    with open(filename, "r") as f:
        contents = yaml.safe_load(f)
    params = contents.keys()
    return (
        dict(zip(params, config)) for config in itertools.product(*contents.values())
    )


def setup_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Neumann network experiment for image reconstruction."
    )
    # wandb arguments
    parser.add_argument("--wandb_entity", help="The entity to use on wandb", type=str)
    parser.add_argument(
        "--wandb_project_name", help="The project name on wandb", type=str
    )
    parser.add_argument(
        "--wandb_mode",
        help="One of ['online', 'offline', 'disabled']",
        type=str,
        choices=["online", "offline", "disabled"],
        default="disabled",
    )
    parser.add_argument("--data_folder", help="Root folder for the dataset", type=str)
    parser.add_argument(
        "--num_train_samples",
        help="Number of samples to use in training",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--num_epochs", help="The number of training epochs", type=int, default=80
    )
    parser.add_argument(
        "--num_solver_iterations",
        help="The number of unrolled iterations",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--kernel_size", help="The size of the blur kernel", type=int, default=5
    )
    parser.add_argument("--batch_size", help="The batch size", type=int, default=64)
    parser.add_argument(
        "--learning_rate",
        help="The learning rate for the network",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--log_file_location", help="Path to a log file", type=str, default=""
    )
    parser.add_argument(
        "--save_frequency",
        help="The frequency with which to save network parameters",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--save_location",
        help="Path to the saved network weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--verbose", help="Set to log additional information", action="store_true"
    )
    # CUDA
    parser.add_argument(
        "--use_cuda",
        help="Set to use CUDA acceleration, if available",
        action="store_true",
    )

    return parser
