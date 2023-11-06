import math
import numbers
from typing import Sequence, Tuple

import torch
import torch.nn.functional as torchfunc


from neumann_networks.linalg import LinearOperator


class GaussianBlur(LinearOperator):
    def __init__(self, sigma, kernel_size=5, n_channels=3, n_spatial_dimensions=2):
        super(GaussianBlur, self).__init__()
        self.groups = n_channels
        if isinstance(kernel_size, numbers.Number):
            self.padding = int(math.floor(kernel_size / 2))
            kernel_size = [kernel_size] * n_spatial_dimensions
        else:
            print(
                "KERNEL SIZE MUST BE A SINGLE INTEGER - RECTANGULAR KERNELS NOT SUPPORTED AT THIS TIME"
            )
            exit()
        self.gaussian_kernel = torch.nn.Parameter(
            self.create_gaussian_kernel(sigma, kernel_size, n_channels),
            requires_grad=False,
        )

    def create_gaussian_kernel(self, sigma, kernel_size, n_channels):
        kernel = torch.tensor(1.0)
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, mgrid in zip(kernel_size, meshgrids):
            mean = (size - 1) / 2
            kernel = kernel * torch.exp(-(((mgrid - mean) / sigma) ** 2) / 2)

        # Make sure norm of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(n_channels, *[1] * (kernel.dim() - 1))
        return kernel

    def forward(self, x):
        return torchfunc.conv2d(
            x, weight=self.gaussian_kernel, groups=self.groups, padding=self.padding
        )

    def adjoint(self, x):
        return torchfunc.conv2d(
            x, weight=self.gaussian_kernel, groups=self.groups, padding=self.padding
        )

    def conv_with_basis(self, dim: int, i: int, j: int):
        """Compute the convolution with a 'basis' image.

        The result of this operation is an image containing the kernel
        centered at the `(i, j)` pixel and zero everywhere else.

        Args:
            dim: The dimension of the image.
            i: The row index of the nonzero pixel.
            j: The col index of the nonzero pixel.

        Returns:
            The resulting image.
        """
        kernel = self.gaussian_kernel[0, 0, :, :]
        kernel = kernel.view(1, 1, *kernel.size())

        img = torch.zeros(
            dim,
            dim,
            dtype=kernel.dtype,
            device=kernel.device,
        )
        img[i, j] = 1.0
        return torchfunc.conv2d(
            img.view(1, 1, *img.size()),
            weight=kernel,
            groups=1,
            padding=self.padding,
        )

    def conv_with_bases(self, dim: int, indices: Sequence[Tuple[int, int]]):
        """Compute the convolution with a sequence of 'basis' images.

        The result of this operation is a sequence of images containing the
        kernel centered at given pixels and zero everywhere else. The pixels
        where the kernel starts correspond to the elements of `indices`.

        Args:
            dim: The dimension of each image.
            indices: A sequence of starting pixel indices.

        Returns:
            The resulting images in a tensor of shape `(len(indices), 1, dim, dim)`.
        """
        kernel = self.gaussian_kernel[0, 0, :, :]
        kernel = kernel.view(1, 1, *kernel.size())

        imgs = torch.zeros(
            len(indices),
            1,
            dim,
            dim,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        for ii, (i, j) in enumerate(indices):
            imgs[ii, 0, i, j] = 1.0
        return torchfunc.conv2d(
            imgs,
            weight=kernel,
            groups=1,
            padding=self.padding,
        )
