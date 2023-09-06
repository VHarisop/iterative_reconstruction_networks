from itertools import product

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from operators.blurs import GaussianBlur
from operators.operator import SelfAdjointLinearOperator


class RegBlurInverse(SelfAdjointLinearOperator):
    """A class implementing the inverse of the linear operator

        q -> X'(X(q)) + mu * q,

    where `X` is a Gaussian blur acting on 2D images and `mu` is a
    regularization parameter.
    """

    dim: int
    singular_values: torch.Tensor
    singular_vectors: torch.Tensor

    def __init__(self, blur: GaussianBlur, dim: int):
        super().__init__()
        self.dim = dim
        with torch.no_grad():
            _, S, Vh = torch.linalg.svd(
                blur.conv_with_bases(
                    dim, indices=list(product(range(dim), range(dim)))
                ).view(dim**2, dim**2),
                full_matrices=False,
            )
            self.singular_values = S
            self.singular_vectors = Vh.t()

    def forward(self, x: torch.Tensor, reg_lambda: torch.Tensor) -> torch.Tensor:
        orig_size = x.size()
        # Reshape x so that the last dimension is the flattened image.
        z = x.view(*orig_size[:-2], self.dim**2)
        return (
            (
                (z @ self.singular_vectors)
                * (1 / (self.singular_values**2 + reg_lambda))
            )
            @ self.singular_vectors.T
        ).view(orig_size)


def compute_blur_inverse(
    blur: GaussianBlur, shift: torch.Tensor, dim: int
) -> torch.Tensor:
    """Compute the inverse of a GaussianBlur (viewed as a matrix).

    Args:
        blur: The Gaussian blur.
        shift: The diagonal shift.
        dim: The number of rows/columns of the images the blur is applied to.

    Returns:
        A LinearOperator that wraps the inverse.
    """
    # Create a full basis.
    fwd_matrix = blur.conv_with_bases(
        dim,
        indices=list(product(range(dim), range(dim))),
    )
    fwd_matrix = fwd_matrix.view(dim**2, dim**2)
    full_matrix = fwd_matrix.T @ fwd_matrix + shift * torch.eye(dim**2)
    return torch.linalg.inv(full_matrix)


def save_tensor_as_color_img(img_tensor, filename):
    np_array = img_tensor.cpu().detach().numpy()
    imageio.save(filename, np_array)


def save_batch_as_color_imgs(tensor_batch, batch_size, ii, folder_name, names):
    # img_array = (np.transpose(tensor_batch.cpu().detach().numpy(),(0,2,3,1)) + 1.0) *  127.5
    img_array = (
        np.clip(np.transpose(tensor_batch.cpu().detach().numpy(), (0, 2, 3, 1)), -1, 1)
        + 1.0
    ) * 127.5
    # img_array = tensor_batch.cpu().detach().numpy()
    # print(np.max(img_array[:]))
    # print(np.min(img_array[:]))

    img_array = img_array.astype(np.uint8)
    for kk in range(batch_size):
        img_number = batch_size * ii + kk
        filename = folder_name + str(img_number) + "_" + str(names[kk]) + ".png"
        # print(np.shape(img_array))
        # print(filename)
        imageio.imwrite(filename, img_array[kk, ...])


def save_mri_as_imgs(tensor_batch, batch_size, ii, folder_name, names):
    # img_array = (np.transpose(tensor_batch.cpu().detach().numpy(),(0,2,3,1)) + 1.0) *  127.5
    img_array = tensor_batch.cpu().detach().numpy()

    for kk in range(batch_size):
        img_number = batch_size * ii + kk
        filename = folder_name + str(img_number) + "_" + str(names[kk]) + ".png"
        plt.imshow(np.sqrt(img_array[kk, 0, :, :] ** 2 + img_array[kk, 1, :, :] ** 2))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(filename, bbox_inches="tight")
