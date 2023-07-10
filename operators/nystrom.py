import torch
import torch.linalg as linalg
from .operator import LinearOperator


def compute_nystrom_extension(operator: LinearOperator, rank: int) -> LinearOperator:
    """Compute the Nystrom extension of a positive-semidefinite linear operator.

    Args:
        operator: The linear operator.
        rank: The target rank of the Nystrom extension.

    Returns:
        The Nystrom extension as a `LinearOperator` instance.
    """
    raise NotImplementedError("Not available yet!")
