import argparse
import torch
import logging
import os

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from operators.operator import LinearOperator
from networks.relu_net import ReluNet
from solvers.neumann import NeumannNet

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
parser.add_argument(
    "--num_solver_iterations",
    help="The number of iterations of the unrolled solver",
    type=int,
    default=6,
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
parser.add_argument("--logfile", type=str)
parser.add_argument("--save_frequency", type=int, default=5)
parser.add_argument("--save_location", type=str, default=os.getenv("HOME"))
# CUDA
parser.add_argument("--use_cuda", action="store_true")

args = parser.parse_args()

# Set up logging
logging.basicConfig(filename=args.logfile, level=logging.INFO)

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
    assert betas.shape == torch.Size([num_samples, dim, 1])
    return betas[:, :, 0]


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


class ForwardOperator(LinearOperator):
    """An operator that selects the first `keep_coords` of a vector."""

    dim: int
    keep_coords: int

    def __init__(self, dim: int, keep_coords: int):
        super().__init__()
        self.dim = dim
        self.keep_coords = keep_coords

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 1:
            return x[:, : self.keep_coords]
        else:
            return x[: self.keep_coords]

    def adjoint(self, x: torch.Tensor):
        if len(x.shape) > 1:
            return torch.cat(
                (
                    x,
                    torch.zeros(
                        x.shape[0],
                        self.dim - self.keep_coords,
                        device=_DEVICE_,
                        dtype=torch.float,
                    ),
                ),
                dim=1,
            )
        else:
            return torch.cat(
                (
                    x,
                    torch.zeros(
                        self.dim - self.keep_coords, device=_DEVICE_, dtype=torch.float
                    ),
                ),
                dim=0,
            )


forward_operator = ForwardOperator(args.dim, args.keep_coords)
forward_operator = forward_operator.to(_DEVICE_)

# A 7-layer ReLU net.
learned_component = ReluNet(dims=[args.dim, 10, 10, 6, 10, 10, args.dim])

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

# set up loss and train
lossfunction = torch.nn.MSELoss()

for epoch in range(args.num_epochs):
    if epoch % args.save_frequency:
        torch.save(
            {
                "solver_state_dict": solver.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            args.save_location,
        )

    for idx, (sample_batch,) in enumerate(train_loader):
        optimizer.zero_grad()
        sample_batch = sample_batch.to(device=_DEVICE_)
        y = forward_operator(sample_batch)
        reconstruction = solver(y, iterations=args.num_solver_iterations)

        loss = lossfunction(reconstruction, sample_batch)
        loss.backward()
        optimizer.step()

        if idx % args.log_frequency == 0:
            logging.info("Epoch: %d - Step: %d - Loss: %f" % (epoch, idx, loss.item()))
