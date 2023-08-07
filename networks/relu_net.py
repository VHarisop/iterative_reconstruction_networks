from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ReluNet(nn.Module):
    """
    A simple neural network with ReLU activations.
    """

    def __init__(self, dims: Sequence[int]):
        """
        Args:
            dims (Sequence[int]): The dimensions of each layer.
        """
        super().__init__()
        self.layers = nn.Sequential(
            *(nn.Linear(feat_in, feat_out) for feat_in, feat_out in zip(dims, dims[1:]))
        )

    def forward(self, input_tensor: torch.Tensor):
        for layer in self.layers[:-1]:
            input_tensor = F.relu(layer(input_tensor), inplace=False)
        return self.layers[-1](input_tensor)
