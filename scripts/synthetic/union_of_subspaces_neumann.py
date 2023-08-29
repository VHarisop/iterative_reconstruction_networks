import argparse
import torch
import os
import random
import sys

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms

from operators.operator import LinearOperator, OperatorPlusNoise
from networks.u_net import UnetModel
from solvers.neumann import NeumannNet
from training import standard_training

parser = argparse.ArgumentParser(
    description="A synthetic experiment using Neumann networks to recover vectors lying on a union of subspaces."
)

# Data options
parser.add_argument("--dim", help="The ambient dimension", type=int, default=10)
parser.add_argument(
    "--rank", help="The dimension of each subspace", type=int, default=3
)
parser.add_argument(
    "--num_subspaces", help="The number of subspaces", type=int, default=3
)
parser.add_argument("--keep_coords", help="The number of coordinates to keep", type=int)
# Training options
parser.add_argument(
    "--num_train_samples", help="The number of training samples", type=int
)
parser.add_argument("--num_test_samples", help="The number of test samples", type=int)
parser.add_argument(
    "--num_epochs", help="The number of training epochs", type=int, default=80
)
parser.add_argument("--batch_size", help="The batch size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument(
    "--algorithm_step_size",
    help="The initial step size of the algorithm",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--start_epoch", help="The starting epoch for training", type=int, default=0
)
# Checkpointing options
parser.add_argument("--log_frequency", type=int, default=10)
parser.add_argument("--save_frequency", type=int, default=5)
parser.add_argument("--save_location", type=str, default=os.getenv("HOME"))
# CUDA
parser.add_argument("--use_cuda", action="store_true")

args = parser.parse_args()

if args.use_cuda:
    if torch.cuda.is_available():
        _DEVICE_ = torch.device("cuda:0")
    else:
        raise ValueError("CUDA is not available!")
else:
    _DEVICE_ = torch.device("cpu")


def generate_vectors(
    dim: int, rank: int, num_subspaces: int, num_samples: int
) -> torch.Tensor:
    """Generate synthetic data from a UoS model.

    Args:
        dim: The ambient dimension.
        rank: The subspace dimension.
        num_subspaces: The number of subspaces in the union.
        num_samples: The number of samples to generate.

    Returns:
        A Pytorch tensor containing the generated vectors as its rows.
    """
    U = torch.randn(num_subspaces, dim, rank, device=_DEVICE_)
    # One random coefficient vector per sample.
    W = torch.randn(num_samples, rank, device=_DEVICE_)
    inds = torch.randint(high=num_subspaces, size=(num_samples,))
    betas = U[inds, :, :] @ W[:, :, None]
    assert betas.shape == torch.Size([num_samples, dim])
    return betas


# Set up data and dataloaders
train_dataset = TensorDataset(
    generate_vectors(args.dim, args.rank, args.num_subspaces, args.num_train_samples)
)
test_dataset = TensorDataset(
    generate_vectors(args.dim, args.rank, args.num_subspaces, args.num_test_samples)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
)

# Set up solver and problem setting
# Here, the forward operator is just a coordinate selection operator.
# Its adjoint is given by appending the complementary number of zeros.
forward_operator = LinearOperator()
forward_operator.forward = lambda x: x[: args.keep_coords]
forward_operator.adjoint = lambda y: torch.cat(
    (y, torch.zeros(args.dim - args.keep_coords, device=_DEVICE_, dtype=torch.float))
)
forward_operator = forward_operator.to(_DEVICE_)

# Standard U-net
learned_component = UnetModel(
    in_chans=1,
    out_chans=1,
    num_pool_layers=4,
    drop_prob=0.0,
    chans=32,
)
solver = NeumannNet(
    linear_operator=forward_operator,
    nonlinear_operator=learned_component,
    eta_initial_val=args.algorithm_step_size,
)
solver = solver.to(device=_DEVICE_)

start_epoch = 0
optimizer = optim.Adam(params=solver.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=(args.learning_rate // 2), gamma=0.1
)
cpu_only = not torch.cuda.is_available()


if os.path.exists(save_location):
    if not cpu_only:
        saved_dict = torch.load(args.save_location)
    else:
        saved_dict = torch.load(args.save_location, map_location="cpu")

    start_epoch = saved_dict["epoch"]
    solver.load_state_dict(saved_dict["solver_state_dict"])
    optimizer.load_state_dict(saved_dict["optimizer_state_dict"])
    scheduler.load_state_dict(saved_dict["scheduler_state_dict"])


# set up loss and train
lossfunction = torch.nn.MSELoss()

# Do train
standard_training.train_solver(
    solver=solver,
    train_dataloader=dataloader,
    test_dataloader=test_dataloader,
    measurement_process=forward_operator,
    optimizer=optimizer,
    save_location=args.save_location,
    loss_function=lossfunction,
    n_epochs=args.num_epochs,
    use_dataparallel=False,
    device=_DEVICE_,
    scheduler=scheduler,
    print_every_n_steps=args.log_frequency,
    save_every_n_epochs=args.save_frequency,
    start_epoch=args.start_epoch,
)
