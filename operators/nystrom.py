import logging
from typing import Set, Tuple

import numpy as np
from numpy.random import choice
import torch
from torch.nn import Linear

from .operator import LinearOperator, SelfAdjointLinearOperator
from .blurs import GaussianBlur


class NystromFactoredBlur(LinearOperator):
    lin_op: GaussianBlur
    dim: int
    rank: int

    def __init__(self, lin_op: GaussianBlur, dim: int, rank: int):
        super().__init__()
        self.lin_op = lin_op
        self.dim = dim
        self.rank = rank

    def create_approximation(self):
        flat_idx_pivots = np.random.randint((0, self.dim ** 2), self.rank)
        self.pivots = list(zip(
            flat_idx_pivots // self.dim,
            flat_idx_pivots % self.dim,
        ))
        # sub_imgs = (X P_S).T
        sub_imgs = self.lin_op.conv_with_bases(self.dim, self.pivots)
        assert sub_imgs.shape == [self.rank, 1, self.dim, self.dim]
        sub_imgs = sub_imgs.view(self.rank, self.dim ** 2)
        # Gram matrix (X'X)_{S, S} and its inverse square root
        evals, evecs = torch.linalg.eigh(sub_imgs @ sub_imgs.T)
        sub_gram_inv = evecs @ ((evals ** (-1/2)) * evecs.T).T
        # sub_imgs.T @ sub_gram_inv has shape `d^2 * self.rank`
        # TODO: Compute the result X.T @ (sub_imgs.T @ sub_gram_inv)
        # Recall: X.T = X, so self.rank-many convolutions are needed.
        pass

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
