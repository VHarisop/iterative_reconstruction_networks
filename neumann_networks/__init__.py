"""
A Pytorch implementation of Neumann networks from [1].

This module implements a standard as well as a preconditioned Neumann network.
Both networks subclass the `torch.nn.Module` and can be integrated into other
Pytorch modules, such as `torch.nn.Sequential`.

[1]: D. Gilton, G. Ongie and R. Willett, *Neumann networks for inverse problems in imaging*. URL: https://arxiv.org/abs/1901.03707
"""

from .linalg import LinearOperator
from .networks import NeumannNet, PreconditionedNeumannNet
