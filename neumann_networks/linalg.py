from abc import ABC, abstractmethod
from typing import Callable

import torch


def dotprod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.conj() * y, dim=tuple(range(1, x.ndim)), keepdim=True)


class LinearOperator(torch.nn.Module, ABC):
    """An abstract class representing a linear operator."""

    @abstractmethod
    def forward(self, _: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def adjoint(self, _: torch.Tensor) -> torch.Tensor:
        pass


def conjugate_gradient(
    linear_op: LinearOperator | Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    max_iters: int,
):
    """Solve the linear system `Ax = b` using a fixed number of CG iterations.

    Args:
        linear_op: A callable or LinearOperator implementing the matrix `A`.
        rhs: The right-hand-side of the equation.
        max_iters: The number of iterations.

    Returns:
        The solution of the linear system, having the same shape as `rhs`.
    """
    x = torch.zeros_like(rhs)
    d = rhs
    g = -d
    for _ in range(max_iters):
        Qd = linear_op(d)
        dQd = dotprod(d, Qd) + 1e-8
        alpha = -dotprod(g, d) / dQd
        x = x + alpha * d
        g = linear_op(x) - rhs
        beta = dotprod(g, Qd) / dQd
        d = -g + beta * d
    return x
