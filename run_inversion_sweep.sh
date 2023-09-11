#!/bin/bash

DATA_DIR="${1}"

# Run a small example with wandb disabled.
PYTHONPATH=$(pwd) python scripts/real/gaussian_blur_neumann_sweep.py \
    --path_to_script scripts/real/gaussian_blur_neumann.py \
    --slurm_log_folder_base "${HOME}/slurm_log" \
    --slurm_partition "general" \
    --wandb_entity vchariso --wandb_project_name sketched-neumann-inversion-test --wandb_mode online \
    --solver_param_config_file experiment_configs/solver_params_compare_cg.yaml \
    --common_param_config_file experiment_configs/common_params_compare_cg.yaml \
    --dataset celeba \
    --data_folder "${DATA_DIR}" \
    --save_location saved_model_weights \
    --use_cuda