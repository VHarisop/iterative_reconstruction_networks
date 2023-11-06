import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim

import operators.blurs as blurs
import wandb
from networks.u_net import UnetModel
from operators.operator import OperatorPlusNoise
from neumann_networks.networks import NeumannNet, PreconditionedNeumannNet
from utils.parsing import setup_common_parser
from utils.train_utils import evaluate_batch_loss, hash_dict


def setup_args() -> argparse.Namespace:
    parser = setup_common_parser()
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
    logging.debug(f"Dataset = {args.dataset}")
    if args.dataset == "cifar10":
        from utils.cifar_10_dataloader import create_dataloaders, create_datasets

        train_loader, test_loader = create_dataloaders(
            *create_datasets(args.data_folder),
            batch_size=args.batch_size,
            num_train_samples=args.num_train_samples,
        )
    elif args.dataset == "celeba":
        from utils.celeba_dataloader import create_dataloaders, create_datasets

        train_loader, test_loader = create_dataloaders(
            *create_datasets(args.data_folder),
            batch_size=args.batch_size,
            num_train_samples=args.num_train_samples,
        )
    else:
        raise ValueError(f"Unknown value {args.dataset} for dataset")
    logging.debug(f"Using {args.num_train_samples} samples")

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
        solver = PreconditionedNeumannNet(
            measurement_operator=internal_forward_operator,
            nonlinear_operator=learned_component,
            num_blocks=args.num_solver_iterations,
            eta=args.algorithm_step_size,
            cg_iterations=args.cg_iterations,
        )
    else:
        solver = NeumannNet(
            measurement_operator=internal_forward_operator,
            nonlinear_operator=learned_component,
            num_blocks=args.num_solver_iterations,
            eta=args.algorithm_step_size,
        )
    # Move solver to CUDA device, if necessary.
    solver = solver.to(device=_DEVICE_)

    optimizer = optim.Adam(params=solver.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=(args.num_epochs // 2), gamma=0.1
    )

    logging.info(
        "Number of parameters: %d" % (sum(p.numel() for p in solver.parameters()))
    )

    # set up loss and train
    loss_function = torch.nn.MSELoss()
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
            y = measurement_process(sample_batch)
            reconstruction = solver(y)
            reconstruction = torch.clamp(reconstruction, -1.0, 1.0)

            # Evaluate loss function and take gradient step.
            loss = loss_function(reconstruction, sample_batch)
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
                loss_function,
                measurement_process,
                train_loader,
                device=_DEVICE_,
            )
            test_loss = evaluate_batch_loss(
                solver,
                loss_function,
                measurement_process,
                test_loader,
                device=_DEVICE_,
            )
            psnr_train = -10 * np.log10(train_loss)
            psnr_test = -10 * np.log10(test_loss)
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
            }
        )

        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    main()
