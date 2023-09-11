#!/bin/bash

DATA_DIR="${1}"

# Run a small example with wandb disabled.
PYTHONPATH=$(pwd) python scripts/real/gaussian_blur_neumann.py \
    --wandb_entity vchariso --wandb_project_name test --wandb_mode disabled \
    --dataset cifar10 --data_folder "${DATA_DIR}" --num_train_samples 2048 \
    --num_epochs 5 --num_solver_iterations 5 --kernel_size 5 --noise_variance 0.0 \
    --batch_size 16 --learning_rate 0.001 --algorithm_step_size 0.1 \
    --save_frequency 10 --save_location . --verbose --use_cuda --report_inversion_error \
    sketchedneumann --rank 32 --sketch_type column