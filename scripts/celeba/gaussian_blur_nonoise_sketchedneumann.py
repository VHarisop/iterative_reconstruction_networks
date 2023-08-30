import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim

import operators.blurs as blurs
from networks.u_net import UnetModel
from operators.nystrom import NystromApproxBlur, NystromApproxBlurGaussian
from solvers.neumann import SketchedNeumannNet
from utils.celeba_dataloader import create_dataloaders, create_datasets
from utils.train_utils import hash_dict


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Neumann network experiment for blurry image reconstruction."
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
        "--rank", help="The rank of the Nystrom approximation", type=int, default=10
    )
    parser.add_argument(
        "--sketch_type",
        help="The sketch type",
        type=str,
        choices=["gaussian", "column"],
        required=True,
    )
    parser.add_argument(
        "--kernel_size", help="The size of the blur kernel", type=int, default=5
    )
    parser.add_argument("--batch_size", help="The batch size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--algorithm_step_size",
        help="The initial step size of the algorithm",
        type=float,
        default=0.1,
    )

    # Checkpointing options
    parser.add_argument("--log_file_location", type=str, default="")
    parser.add_argument("--save_frequency", type=int, default=5)
    parser.add_argument("--save_location", type=str, default=os.getenv("HOME"))
    parser.add_argument("--verbose", action="store_true")
    # CUDA
    parser.add_argument("--use_cuda", action="store_true")

    return parser.parse_args()


args = setup_args()
if args.use_cuda:
    if torch.cuda.is_available():
        _DEVICE_ = torch.device("cuda:0")
        logging.info("Using CUDA")
    else:
        raise ValueError("CUDA is not available!")
else:
    _DEVICE_ = torch.device("cpu")

# Set up logging
logging.basicConfig(
    level=(logging.DEBUG if args.verbose else logging.INFO),
    filename=args.log_file_location,
)
logging.debug(f"Device = {_DEVICE_}")

# Set up data and dataloaders
# Set up data and dataloaders
train_data, test_data = create_datasets(args.data_folder)
train_loader, test_loader = create_dataloaders(
    train_data,
    test_data,
    batch_size=args.batch_size,
    num_train_samples=args.num_train_samples,
)
logging.info(f"Using {args.num_train_samples} samples")

### Set up solver and problem setting

forward_operator = blurs.GaussianBlur(
    sigma=5.0, kernel_size=args.kernel_size, n_channels=3, n_spatial_dimensions=2
).to(device=_DEVICE_)
measurement_process = forward_operator

if args.sketch_type == "gaussian":
    sketched_forward_operator = NystromApproxBlurGaussian(
        forward_operator,
        dim=64,
        rank=args.rank,
    )
else:
    sketched_forward_operator = NystromApproxBlur(
        forward_operator,
        dim=64,
        rank=args.rank,
        pivots=None,
    )

# standard u-net
learned_component = UnetModel(
    in_chans=3,
    out_chans=3,
    num_pool_layers=4,
    drop_prob=0.0,
    chans=32,
)
solver = SketchedNeumannNet(
    linear_operator=forward_operator,
    sketched_operator=sketched_forward_operator,
    nonlinear_operator=learned_component,
    lambda_initial_val=args.algorithm_step_size,
)
solver = solver.to(device=_DEVICE_)

optimizer = optim.Adam(params=solver.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=(args.num_epochs // 2), gamma=0.1
)
cpu_only = not torch.cuda.is_available()

# set up loss and train
lossfunction = torch.nn.MSELoss()

# Training
time_elapsed = 0.0
for epoch in range(args.num_epochs):
    if epoch % args.save_frequency:
        torch.save(
            {
                "solver_state_dict": solver.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(
                args.save_location,
                f"{hash_dict(vars(args))}_{epoch}.pt",
            ),
        )
    start_time = time.time()
    for idx, (sample_batch, _) in enumerate(train_loader):
        optimizer.zero_grad()
        sample_batch = sample_batch.to(device=_DEVICE_)
        y = forward_operator(sample_batch)
        # Reconstruct image using `num_solver_iterations` unrolled iterations.
        reconstruction = solver(y, iterations=args.num_solver_iterations)
        reconstruction = torch.clamp(reconstruction, -1, 1)

        # Evaluate loss function and take gradient step.
        loss = lossfunction(reconstruction, sample_batch)
        loss.backward()
        optimizer.step()

        # Log to stderr and wandb
        logging.info("Epoch: %d - Step: %d - Loss: %f" % (epoch, idx, loss.item()))
    # Compute elapsed time
    elapsed_time = time.time() - start_time
    logging.info("Epoch: %d - Time elapsed: %.3f" % (epoch, elapsed_time))

    # TODO: Report loss over training set and elapsed time to Wandb
    with torch.no_grad():
        loss_accumulator = []
        for idx, (sample_batch, _) in enumerate(train_loader):
            sample_batch = sample_batch.to(_DEVICE_)
            y = forward_operator(sample_batch)
            reconstruction = solver(y, iterations=args.num_solver_iterations)
            reconstruction = torch.clamp(reconstruction, -1, 1)
            # Append loss to accumulator
            loss_accumulator.append(lossfunction(reconstruction, sample_batch).item())
        loss_array = np.asarray(loss_accumulator)
        loss_mse = np.mean(loss_array)
        train_psnr = -10 * np.log10(loss_mse)
        percentiles_psnr = -10 * np.log10(np.percentile(loss_array, [25, 50, 75]))
        logging.info("Train MSE: %f - Train mean PSNR: %f" % (loss_mse, train_psnr))
        logging.info("Train PSNR quartiles: %.2f, %.2f, %.2f" % tuple(percentiles_psnr))

    if scheduler is not None:
        scheduler.step()

# Evaluation
loss_accumulator = []
with torch.no_grad():
    for ii, (sample_batch, _) in enumerate(test_loader):
        sample_batch = sample_batch.to(device=_DEVICE_)
        y = measurement_process(sample_batch)
        reconstruction = solver(y)
        reconstruction = torch.clamp(reconstruction, -1, 1)

        # Evalute loss function
        loss = lossfunction(reconstruction, sample_batch)
        loss_accumulator.append(loss.item())

loss_array = np.asarray(loss_accumulator)
loss_mse = np.mean(loss_array)
PSNR = -10 * np.log10(loss_mse)
percentiles = np.percentile(loss_array, [25, 50, 75])
percentiles = -10.0 * np.log10(percentiles)
logging.info("Test loss: %f - Test mean PSNR: %f" % (loss_mse, PSNR))
logging.info("Test PSNR quartiles: %f, %f, %f" % tuple(percentiles))
