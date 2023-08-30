import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
import wandb

import operators.blurs as blurs
from networks.u_net import UnetModel
from operators.nystrom import NystromApproxBlur, NystromApproxBlurGaussian
from solvers.neumann import SketchedNeumannNet
from utils.celeba_dataloader import create_dataloaders, create_datasets
from utils.parsing import setup_common_parser
from utils.train_utils import hash_dict


def setup_args() -> argparse.Namespace:
    parser = setup_common_parser()
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
        "--algorithm_step_size",
        help="The initial step size of the algorithm",
        type=float,
        default=0.1,
    )
    return parser.parse_args()


args = setup_args()
if args.use_cuda:
    if torch.cuda.is_available():
        _DEVICE_ = torch.device("cuda:0")
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

# Set up wandb logging
run = wandb.init(
    config=vars(args),
    id=hash_dict(vars(args)),
    entity=args.wandb_entity,
    project=args.wandb_project_name,
    resume=None,
    mode=args.wandb_mode,
    settings=wandb.Settings(start_method="fork"),
)
assert run is not None

# Set up data and dataloaders
train_data, test_data = create_datasets(args.data_folder)
train_loader, test_loader = create_dataloaders(
    train_data,
    test_data,
    batch_size=args.batch_size,
    num_train_samples=args.num_train_samples,
)
logging.info(f"Using {args.num_train_samples} samples")

# Set up solver
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

# set up loss and train
lossfunction = torch.nn.MSELoss()

# Training
for epoch in range(args.num_epochs):
    if epoch % args.save_frequency == 0:
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

    total_batches = 0
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

        # Log to stderr
        logging.info("Epoch: %d - Step: %d - Loss: %f" % (epoch, idx, loss.item()))
        total_batches += 1
    # Compute elapsed time
    elapsed_time = time.time() - start_time
    logging.info("Epoch: %d - Time elapsed: %.3f" % (epoch, elapsed_time))

    # Report loss over training set and elapsed time to Wandb
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
        run.log(
            {
                "epoch": epoch,
                "train_loss": loss_mse,
                "train_psnr_mean": train_psnr,
                "train_psnr_quartiles": tuple(percentiles_psnr),
                "elapsed_time": elapsed_time,
                "elapsed_time_per_batch": elapsed_time / total_batches,
            }
        )

    if scheduler is not None:
        scheduler.step()

# Evaluation
loss_accumulator = []
with torch.no_grad():
    for ii, (sample_batch, _) in enumerate(test_loader):
        sample_batch = sample_batch.to(device=_DEVICE_)
        y = measurement_process(sample_batch)
        reconstruction = solver(y, iterations=args.num_solver_iterations)
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
