import logging
from typing import Set, Tuple

import numpy as np
from numpy.random import choice
import torch
import torch.nn.functional as torchfunc

from .operator import LinearOperator, SelfAdjointLinearOperator
from .blurs import GaussianBlur


class NystromFactoredBlur(SelfAdjointLinearOperator):
    lin_op: GaussianBlur
    dim: int
    rank: int
    nystrom_factor: torch.Tensor
    nys_U: torch.Tensor
    nys_S: torch.Tensor
    nys_Vh: torch.Tensor

    def __init__(self, lin_op: GaussianBlur, dim: int, rank: int):
        super().__init__()
        self.lin_op = lin_op
        self.dim = dim
        self.rank = rank
        self.nystrom_factor = self._create_approximation()
        self.nys_U, self.nys_S, self.nys_Vh = torch.linalg.svd(
            self.nystrom_factor.view(self.rank, self.dim**2).T,
            full_matrices=False,
        )

    def _create_approximation(self) -> torch.Tensor:
        """Create the Nystrom approximation from a random subset of columns.

        Returns:
            A tensor of shape `(self.rank, 1, self.dim, self.dim)` containing
            the left Nystrom factor. This method assumes that the kernel is the
            same for all channels if there are more than one.
        """
        flat_idx_pivots = np.random.randint(0, self.dim**2, self.rank)
        self.pivots = list(
            zip(
                flat_idx_pivots // self.dim,
                flat_idx_pivots % self.dim,
            )
        )
        # sub_imgs = (X P_S).T
        sub_imgs = self.lin_op.conv_with_bases(self.dim, self.pivots)
        sub_imgs = sub_imgs.view(self.rank, self.dim**2)
        # Gram matrix (X'X)_{S, S} and its inverse square root
        evals, evecs = torch.linalg.eigh(sub_imgs @ sub_imgs.T)
        sub_gram_inv = evecs @ ((evals ** (-1 / 2)) * evecs.T).T
        # sub_imgs.T @ sub_gram_inv has shape `d^2 * self.rank`
        # We now compute the result X.T @ (sub_imgs.T @ sub_gram_inv)
        # Recall: X.T = X, so self.rank-many convolutions are needed.
        gram_inv_prod = (sub_gram_inv @ sub_imgs).view(self.rank, 1, self.dim, self.dim)
        kernel = self.lin_op.gaussian_kernel[0, 0, :, :]
        kernel = kernel.view(1, 1, *kernel.size())
        nystrom_prod = torchfunc.conv2d(
            gram_inv_prod,
            kernel,
            groups=1,
            padding=self.lin_op.padding,
        )
        return nystrom_prod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        # Reshape x so that the last dimension is the flattened image.
        x = x.view(*x.size()[:-2], self.dim**2)
        return (((x @ self.nys_U) * self.nys_S**2) @ self.nys_U.T).view(*orig_size)


class NystromFactoredInverseBlur(SelfAdjointLinearOperator):
    """The inverse of a Nystrom-factored Gaussian blur."""

    nystrom_factored_blur: NystromFactoredBlur
    reg_lambda: float | torch.Tensor
    scale_vec: torch.Tensor
    nys_U: torch.Tensor

    def __init__(
        self,
        nystrom_factored_blur: NystromFactoredBlur,
        reg_lambda: float | torch.Tensor,
    ):
        super().__init__()
        self.nystrom_factored_blur = nystrom_factored_blur
        self.reg_lambda = reg_lambda
        self.scale_vec = nystrom_factored_blur.nys_S**2 / (
            reg_lambda + nystrom_factored_blur.nys_S**2
        )
        self.nys_U = nystrom_factored_blur.nys_U

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        # Reshape x so that the last dimension is the flattened image.
        x = x.view(*x.size()[:-2], self.nystrom_factored_blur.dim**2)
        return (1 / self.reg_lambda) * (
            x - ((x @ self.nys_U) * self.scale_vec) @ self.nys_U.T
        ).view(*orig_size)


class NystromFactoredOperator(SelfAdjointLinearOperator):
    factor: torch.Tensor

    def __init__(self, factor: torch.Tensor):
        super(NystromFactoredOperator, self).__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 1:
            # Treat the first dimension as the batch size.
            return (x.view(x.shape[0], -1) @ self.factor) @ self.factor.T
        else:
            return self.factor @ (self.factor.T @ x)


class NystromFactoredInverseOperator(SelfAdjointLinearOperator):
    """A linear operator implementing the inverse of (s * I + V @ V.T)."""

    U: torch.Tensor
    scale_vec: torch.Tensor
    shift: float | torch.Tensor

    @torch.no_grad()
    def __init__(self, operator: NystromFactoredOperator, shift: float | torch.Tensor):
        super(SelfAdjointLinearOperator, self).__init__()
        U, S, _ = torch.linalg.svd(operator.factor, full_matrices=False)
        self.U = U
        self.scale_vec = (S**2) / (S**2 + shift)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 1:
            z = x.view(x.shape[0], -1)
            return (1 / self.shift) * (
                z - ((z @ self.U) * self.scale_vec) @ self.U.T
            ).view(x.shape)
        else:
            return (1 / self.shift) * (x - self.U @ (self.scale_vec * (self.U.T @ x)))


@torch.no_grad()
def nystrom_approx_factored(
    operator: LinearOperator,
    dim: int,
    rank: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
) -> Tuple[NystromFactoredOperator, Set[int]]:
    """Compute the factored Nystrom approximation of a factored PSD linear operator.

    Args:
        operator: The Gram factor of the linear operator.
        dim: The ambient dimension.
        rank: The target rank of the Nystrom approximation.
        device: The Pytorch device (default: cpu).
        dtype: The Pytorch dtype (default: float).

    Returns:
        The left factor of the Nystrom approximation.
    """

    # Multiplication with a canonical basis vector.
    def _mul_with_basis(idx: int) -> torch.Tensor:
        basis_vec = torch.zeros(dim, dtype=dtype, device=device)
        basis_vec[idx] = 1
        return operator(basis_vec)

    sample_probs = torch.zeros(dim, device=device, dtype=dtype)
    # Create the vector of sample probabilities
    for idx in range(dim):
        sample_probs[idx] = torch.sum(_mul_with_basis(idx) ** 2)
    # Initialize Cholesky factor and list of pivots
    factor = torch.zeros(dim, rank, device=device, dtype=dtype)
    pivots = set()
    for idx in range(rank):
        try:
            sample_idx = choice(dim, p=np.array(sample_probs / sum(sample_probs)))
        except ValueError as err:
            logging.fatal(f"Problem with sampling: sample_probs={sample_probs}")
            # Re-throw ValueError
            raise ValueError(err.args)
        res_vector = operator.adjoint(_mul_with_basis(sample_idx)) - (
            factor[:, :idx] @ factor[sample_idx, :idx]
        )
        factor[:, idx] = res_vector / torch.sqrt(res_vector[sample_idx])
        # Update sample probabilities and list of pivots
        sample_probs = torch.clamp_min(sample_probs - factor[:, idx] ** 2, 0.0)
        pivots.add(sample_idx)
    return NystromFactoredOperator(factor), pivots
