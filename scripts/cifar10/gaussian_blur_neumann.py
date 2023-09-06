import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim

import operators.blurs as blurs
import operators.nystrom as nystrom
import wandb
from networks.u_net import UnetModel
from operators.operator import LinearOperator, OperatorPlusNoise
from solvers.neumann import (
    NeumannNet,
    PrecondNeumannNet,
    RPCholeskyPrecondNeumannNet,
    SketchedNeumannNet,
)
from utils.cifar_10_dataloader import create_dataloaders, create_datasets
from utils.parsing import setup_common_parser
from utils.testing_utils import RegBlurInverse
from utils.train_utils import evaluate_batch_loss, hash_dict


class ExactPrecondNeumannNet(torch.nn.Module):
    """A PrecondNeumannNet using exact matrix inversions.

    Backward passes through this module are likely to be too costly."""

    linear_op: LinearOperator
    nonlinear_op: torch.nn.Module
    eta: torch.Tensor
    cached_blur_inverse: RegBlurInverse

    def __init__(
        self,
        base_net: PrecondNeumannNet | SketchedNeumannNet | RPCholeskyPrecondNeumannNet,
        cached_blur_inverse: RegBlurInverse,
    ):
        super().__init__()
        self.linear_op = base_net.linear_op
        self.nonlinear_op = base_net.nonlinear_op
        self.eta = base_net.eta
        self.cached_blur_inverse = cached_blur_inverse

    def forward(self, y: torch.Tensor, iterations: int) -> torch.Tensor:
        initial_point = self.cached_blur_inverse.forward(
            self.linear_op.adjoint(y),
            self.eta,
        )
        running_term = initial_point
        accumulator = initial_point

        for _ in range(iterations):
            running_term = self.eta * self.cached_blur_inverse.forward(
                running_term,
                self.eta,
            ) - self.nonlinear_op(running_term)
            accumulator = accumulator + running_term

        return accumulator


def setup_args() -> argparse.Namespace:
    parser = setup_common_parser()
    parser.add_argument(
        "--noise_variance",
        help="The variance of the measurement noise",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--report_inversion_error",
        help="Set to report the error of inverting T_λ",
        action="store_true",
    )
    # Create subparsers
    subparsers = parser.add_subparsers(help="The network type", dest="solver")
    parser_precondneumann = subparsers.add_parser(
        "precondneumann", help="A preconditioned Neumann network"
    )
    parser_precondneumann.add_argument(
        "--cg_iterations",
        help="The number of CG iterations",
        type=int,
        default=10,
    )
    parser_sketchedneumann = subparsers.add_parser(
        "sketchedneumann",
        help="A preconditioned Neumann network with sketching",
    )
    parser_sketchedneumann.add_argument(
        "--rank",
        help="The rank of the low-rank approximation",
        type=int,
        required=True,
    )
    parser_sketchedneumann.add_argument(
        "--sketch_type",
        choices=["gaussian", "column"],
        type=str,
        default="column",
    )
    parser_rpcholneumann = subparsers.add_parser(
        "rpcholneumann",
        help="A Nystrom-preconditioned Neumann network",
    )
    parser_rpcholneumann.add_argument(
        "--sketch_type", choices=["gaussian", "column"], type=str, default="column"
    )
    parser_rpcholneumann.add_argument(
        "--rank",
        help="The rank of the Nystrom approximation",
        type=int,
        required=True,
    )
    parser_rpcholneumann.add_argument(
        "--cg_iterations",
        help="The number of (preconditioned) CG iterations",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
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
    internal_forward_operator = blurs.GaussianBlur(
        sigma=5.0, kernel_size=args.kernel_size, n_channels=3, n_spatial_dimensions=2
    ).to(device=_DEVICE_)
    if args.noise_variance > 0.0:
        measurement_process = OperatorPlusNoise(
            internal_forward_operator,
            noise_sigma=np.sqrt(args.noise_variance),
        )
    else:
        measurement_process = internal_forward_operator

    # standard u-net
    learned_component = UnetModel(
        in_chans=3,
        out_chans=3,
        num_pool_layers=4,
        drop_prob=0.0,
        chans=32,
    )
    if args.solver == "precondneumann":
        solver = PrecondNeumannNet(
            linear_operator=internal_forward_operator,
            nonlinear_operator=learned_component,
            lambda_initial_val=args.algorithm_step_size,
            cg_iterations=args.cg_iterations,
        )
    elif args.solver in ["sketchedneumann", "rpcholneumann"]:
        if args.sketch_type == "column":
            sketched_operator = nystrom.NystromApproxBlur(
                lin_op=internal_forward_operator,
                dim=32,
                rank=args.rank,
                pivots=None,
            )
        else:
            sketched_operator = nystrom.NystromApproxBlurGaussian(
                lin_op=internal_forward_operator,
                dim=32,
                rank=args.rank,
            )
        if args.solver == "sketchedneumann":
            solver = SketchedNeumannNet(
                linear_operator=internal_forward_operator,
                sketched_operator=sketched_operator,
                nonlinear_operator=learned_component,
                lambda_initial_val=args.algorithm_step_size,
            )
        else:
            solver = RPCholeskyPrecondNeumannNet(
                linear_operator=internal_forward_operator,
                nystrom_op=sketched_operator,
                nonlinear_operator=learned_component,
                lambda_initial_val=args.algorithm_step_size,
                cg_iterations=args.cg_iterations,
            )
    else:
        solver = NeumannNet(
            linear_operator=internal_forward_operator,
            nonlinear_operator=learned_component,
            eta_initial_val=args.algorithm_step_size,
        )
    # Move solver to CUDA device, if necessary.
    solver = solver.to(device=_DEVICE_)

    optimizer = optim.Adam(params=solver.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=(args.num_epochs // 2), gamma=0.1
    )

    # set up loss and train
    lossfunction = torch.nn.MSELoss()

    if args.report_inversion_error:
        blur_inverse = RegBlurInverse(
            blur=internal_forward_operator,
            dim=32,
        )

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
        total_reconstruction_error = 0.0
        start_time = time.time()
        for idx, (sample_batch, _) in enumerate(train_loader):
            optimizer.zero_grad()
            sample_batch = sample_batch.to(device=_DEVICE_)
            y = measurement_process(sample_batch)
            # Reconstruct image using `num_solver_iterations` unrolled iterations.
            reconstruction = solver(y, iterations=args.num_solver_iterations)
            reconstruction = torch.clamp(reconstruction, -1, 1)

            if args.report_inversion_error and isinstance(
                solver,
                PrecondNeumannNet | SketchedNeumannNet | RPCholeskyPrecondNeumannNet,
            ):
                # Create a solver ExactPrecondNeumannNet
                # that performs a forward pass using the exact inversion
                # formula. Then compare how far `reconstruction` is from
                # the output of ExactPrecondNeumannNet.
                with torch.no_grad():
                    exact_net = ExactPrecondNeumannNet(
                        solver,
                        blur_inverse,
                    )
                    reconstruction_exact = exact_net.forward(
                        y, iterations=args.num_solver_iterations
                    )
                    reconstruction_exact = torch.clamp(reconstruction_exact, -1, 1)
                    reconstruction_error = torch.linalg.norm(
                        reconstruction - reconstruction_exact
                    ) / torch.linalg.norm(reconstruction_exact)
                    total_reconstruction_error += reconstruction_error.item() ** 2

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

        # Report loss over training/test sets and elapsed time to stderr + wandb
        with torch.no_grad():
            train_loss = evaluate_batch_loss(
                solver,
                lossfunction,
                measurement_process,
                train_loader,
                device=_DEVICE_,
                iterations=args.num_solver_iterations,
            )
            test_loss = evaluate_batch_loss(
                solver,
                lossfunction,
                measurement_process,
                test_loader,
                device=_DEVICE_,
                iterations=args.num_solver_iterations,
            )
            psnr_train = -10 * np.log10(train_loss)
            psnr_test = -10 * np.log10(test_loss)
            if args.report_inversion_error and isinstance(
                solver,
                PrecondNeumannNet | SketchedNeumannNet | RPCholeskyPrecondNeumannNet,
            ):
                avg_inversion_error = np.sqrt(
                    total_reconstruction_error / total_batches
                )
            else:
                avg_inversion_error = None
        # Report summary statistics to stderr + wandb
        logging.info("Train MSE: %f - Train mean PSNR: %f" % (train_loss, psnr_train))
        logging.info("Test MSE: %f - Test mean PSNR: %f" % (test_loss, psnr_test))
        run.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_psnr_mean": psnr_train,
                "test_psnr_mean": psnr_test,
                "elapsed_time": elapsed_time,
                "elapsed_time_per_batch": elapsed_time / total_batches,
                "save_path": os.path.join(
                    args.save_location, f"{hash_dict(vars(args))}_{epoch}.pt"
                ),
                "avg_inversion_error": avg_inversion_error,
            }
        )

        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    main()
