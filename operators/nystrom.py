import logging
import numpy as np
from numpy.random import choice
import torch
import torch.linalg as linalg
from .operator import LinearOperator

from typing import Set, Tuple


@torch.no_grad()
def nystrom_approx_factored(
    operator: LinearOperator,
    rank: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float,
) -> Tuple[LinearOperator, Set[int]]:
    """Compute the factored Nystrom approximation of a factored PSD linear operator.

    Args:
        operator: The Gram factor of the linear operator.
        rank: The target rank of the Nystrom approximation.

    Returns:
        The left factor of the Nystrom approximation.
    """
    try:
        m, d = operator.shape
    except AttributeError:
        raise ValueError("operator must have a .shape attribute!")

    # Multiplication with a canonical basis vector.
    def _mul_op_basis(idx: int) -> torch.Tensor:
        basis_vec = torch.zeros(d, dtype=dtype, device=device)
        basis_vec[idx] = 1
        return operator.forward(basis_vec)

    sample_probs = torch.zeros(d, device=device, dtype=dtype)
    # Create the vector of sample probabilities
    for idx in range(d):
        sample_probs[idx] = torch.sum(_mul_op_basis(idx) ** 2)
    # Initialize Cholesky factor and list of pivots
    factor = torch.zeros(d, rank, device=device, dtype=dtype)
    pivots = set()
    for idx in range(rank):
        try:
            sample_idx = choice(d, p=np.array(sample_probs / sum(sample_probs)))
        except ValueError as err:
            logging.fatal(f"Problem with sampling: sample_probs={sample_probs}")
            # Re-throw ValueError
            raise ValueError(err.args)
        res_vector = operator.adjoint(_mul_op_basis(sample_idx)) - (
            factor[:, :idx] @ factor[sample_idx, :idx]
        )
        factor[:, idx] = res_vector / torch.sqrt(res_vector[sample_idx])
        # Update sample probabilities and list of pivots
        sample_probs = torch.clamp_min(sample_probs - factor[:, idx] ** 2, 0.0)
        pivots.add(sample_idx)
    factor_op = LinearOperator()
    factor_op.forward = lambda x: factor @ x
    factor_op.adjoint = lambda y: factor.T @ y
    factor_op.shape = torch.Size((d, rank))
    return factor_op, pivots
