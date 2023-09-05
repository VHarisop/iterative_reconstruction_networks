import logging
from typing import List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as torchfunc
from numpy.random import choice

from .blurs import GaussianBlur
from .operator import LinearOperator, SelfAdjointLinearOperator


class NystromApproxBlurGaussian(SelfAdjointLinearOperator):
    """A Nystrom approximation to `X'X` where `X` is a Gaussian blur.

    This Nystrom approximation uses a matrix with standard Gaussian entries
    as the test matrix."""

    def __init__(self, lin_op: GaussianBlur, dim: int, rank: int):
        super().__init__()
        self.lin_op = lin_op
        self.dim = dim
        self.rank = rank
        self.nystrom_factor = self._create_approximation().view(
            self.rank,
            self.dim**2,
        )

    def _create_approximation(self):
        kernel = self.lin_op.gaussian_kernel[0, 0, :, :]
        kernel = kernel.view(1, 1, *kernel.size())
        test_mat = torch.randn(self.rank, 1, self.dim, self.dim, device=kernel.device)
        # sub_imgs = (X P_S).T
        sub_imgs = torchfunc.conv2d(
            test_mat,
            weight=kernel,
            groups=1,
            padding=self.lin_op.padding,
        )
        assert sub_imgs.size() == torch.Size([self.rank, 1, self.dim, self.dim])
        # Compute the left-singular vectors from the SVD of (XP_S).
        sub_U, _, _ = torch.linalg.svd(
            sub_imgs.view(self.rank, self.dim**2).T,
            full_matrices=False,
        )
        # We now compute the result: X.T @ U
        # Recall: X.T = X, so self.rank-many convolutions are needed.
        nystrom_prod = torchfunc.conv2d(
            (sub_U.T).view(self.rank, 1, self.dim, self.dim),
            weight=kernel,
            groups=1,
            padding=self.lin_op.padding,
        )
        assert nystrom_prod.size() == torch.Size([self.rank, 1, self.dim, self.dim])
        return nystrom_prod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        # Reshape x so that the last dimension is the flattened image.
        return (
            (x.view(*x.size()[:-2], self.dim**2) @ self.nystrom_factor.T)
            @ self.nystrom_factor
        ).view(orig_size)


class NystromApproxBlur(SelfAdjointLinearOperator):
    """A Nystrom approximation to `X.T @ X`, where `X` is a Gaussian blur.

    Attributes:
        lin_op: The linear operator representing the Gaussian blur.
        dim: The pixel dimension.
        rank: The rank of the Nystrom approximation.
        nystrom_factor: The right factor of the factored Nystrom approximation.
        pivots: A list of column indices used to form the Nystrom approximation.
    """

    lin_op: GaussianBlur
    dim: int
    rank: int
    nystrom_factor: torch.Tensor
    pivots: List[Tuple[int, int]]

    def __init__(
        self,
        lin_op: GaussianBlur,
        dim: int,
        rank: int | None = None,
        pivots: List[Tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.lin_op = lin_op
        self.dim = dim
        if rank is None:
            if pivots is None:
                raise ValueError("`pivots` and `rank` cannot be both `None`.")
            else:
                self.pivots = pivots
                self.rank = len(pivots)
        else:
            if pivots is not None:
                raise ValueError("`pivots` and `rank` cannot be both not `None`.")
            self.rank = rank
            flat_idx_pivots = np.random.randint(0, self.dim**2, self.rank)
            self.pivots = list(
                zip(
                    flat_idx_pivots // self.dim,
                    flat_idx_pivots % self.dim,
                )
            )
        self.nystrom_factor = self._create_approximation().view(self.rank, dim**2)

    @torch.no_grad()
    def _create_approximation(self) -> torch.Tensor:
        """Create the Nystrom approximation from a random subset of columns.

        Returns:
            A tensor of shape `(self.rank, 1, self.dim, self.dim)` containing
            the left Nystrom factor. This method assumes that the kernel is the
            same for all channels if there are more than one.
        """
        # sub_imgs = (X P_S).T
        sub_imgs = self.lin_op.conv_with_bases(self.dim, self.pivots)
        sub_imgs = sub_imgs.view(self.rank, self.dim**2)
        # Compute the left-singular vectors from the SVD of (XP_S).
        sub_U, _, _ = torch.linalg.svd(sub_imgs.T, full_matrices=False)
        # We now compute the result: X.T @ U
        # Recall: X.T = X, so self.rank-many convolutions are needed.
        kernel = self.lin_op.gaussian_kernel[0, 0, :, :]
        kernel = kernel.view(1, 1, *kernel.size())
        nystrom_prod = torchfunc.conv2d(
            (sub_U.T).view(self.rank, 1, self.dim, self.dim),
            kernel,
            groups=1,
            padding=self.lin_op.padding,
        )
        return nystrom_prod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        # Reshape x so that the last dimension is the flattened image.
        return (
            (x.view(*x.size()[:-2], self.dim**2) @ self.nystrom_factor.T)
            @ self.nystrom_factor
        ).view(orig_size)


class NystromApproxBlurInverse(SelfAdjointLinearOperator):
    """The inverse of a regularized Nystrom approximation.

    Attributes:
        nystrom_approx_blur: The Nystrom approximation.
        reg_lambda: The regularization constant.
    """

    nystrom_approx_blur: NystromApproxBlur | NystromApproxBlurGaussian
    reg_lambda: torch.Tensor | torch.nn.Parameter

    def __init__(
        self,
        nystrom_approx_blur: NystromApproxBlur | NystromApproxBlurGaussian,
        reg_lambda: torch.Tensor | torch.nn.Parameter,
    ):
        super().__init__()
        self.nystrom_approx_blur = nystrom_approx_blur
        self.reg_lambda = reg_lambda
        self._nys_U, self._scale_vec = self._cache_inverse()

    @torch.no_grad()
    def _cache_inverse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache the inverse operator."""
        nys_factor = self.nystrom_approx_blur.nystrom_factor
        nys_U, nys_S, _ = torch.linalg.svd(nys_factor.T, full_matrices=False)
        scale_vec = nys_S**2 / (self.reg_lambda + nys_S**2)
        return nys_U, scale_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        # Reshape x so that the last dimension is the flattened image.
        x = x.view(*x.size()[:-2], self.nystrom_approx_blur.dim**2)
        return (1 / self.reg_lambda) * (
            x - ((x @ self._nys_U) * self._scale_vec) @ self._nys_U.T
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


class NystromPreconditioner(SelfAdjointLinearOperator):
    """A Nystrom-based preconditioner for a Gaussian blur."""

    U: torch.Tensor
    S: torch.Tensor
    mu: torch.Tensor
    scale_vec: torch.Tensor
    dim: int

    @torch.no_grad()
    def __init__(
        self, nys_op: NystromApproxBlur | NystromApproxBlurGaussian, mu: torch.Tensor
    ):
        super().__init__()
        self.nystrom_factor = nys_op.nystrom_factor.t()
        self.dim = nys_op.dim
        self.U, self.S, _ = torch.linalg.svd(self.nystrom_factor, full_matrices=False)
        self.mu = mu
        self.scale_vec = torch.sqrt(1 / (self.S**2 + self.mu))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        z = x.view(*orig_size[:-2], self.dim**2)
        # Reshape x so that the last dimension is the flattened image.
        Ux = z @ self.U
        return (
            ((Ux) * self.scale_vec) @ self.U.T
            + torch.sqrt(1 / self.mu) * (z - Ux @ self.U.T)
        ).view(*orig_size)


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
