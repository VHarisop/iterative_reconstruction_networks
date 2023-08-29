import torch
import torch.nn as nn
from operators.blurs import GaussianBlur

from operators.nystrom import NystromApproxBlur, NystromApproxBlurGaussian
from operators.nystrom import NystromApproxBlurInverse
from operators.operator import LinearOperator
from solvers.cg_utils import conjugate_gradient


class NeumannNet(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta_initial_val=0.1):
        super(NeumannNet, self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = "linear_param_"
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(
            name="eta",
            param=torch.nn.Parameter(torch.tensor(eta_initial_val), requires_grad=True),
        )

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def single_block(self, input_tensor):
        return (
            input_tensor
            - self.eta * self.linear_op.gramian(input_tensor)
            - self.nonlinear_op(input_tensor)
        )

    def forward(self, y, iterations):
        initial_point = self.eta * self._linear_adjoint(y)
        running_term = initial_point
        accumulator = initial_point

        for _ in range(iterations):
            running_term = self.single_block(running_term)
            accumulator = accumulator + running_term

        return accumulator


class PrecondNeumannNet(nn.Module):
    """A preconditioned Neumann network.

    Attributes:
        linear_op: The linear operator.
        nonlinear_op: The nonlinear operator, which is often a learned component.
        cg_iterations: The number of CG iterations to unroll when computing
            inverses.
        eta: The amount of preconditioning added.
    """

    linear_op: LinearOperator
    nonlinear_op: nn.Module
    cg_iterations: int
    eta: nn.Module | torch.Tensor

    def __init__(
        self,
        linear_operator,
        nonlinear_operator,
        lambda_initial_val=0.1,
        cg_iterations=10,
    ):
        super(PrecondNeumannNet, self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator
        self.cg_iterations = cg_iterations

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = "linear_param_"
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(
            name="eta",
            param=torch.nn.Parameter(
                torch.tensor(lambda_initial_val), requires_grad=True
            ),
        )

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def _linear_gramian(self, x):
        return self.linear_op.gramian(x)

    # This is a bit redundant
    def initial_point(self, y):
        preconditioned_input = conjugate_gradient(
            self._linear_adjoint(y),
            self._linear_gramian,
            regularization_lambda=self.eta,
            n_iterations=self.cg_iterations,
        )
        return preconditioned_input

    def single_block(self, input_tensor):
        preconditioned_step = conjugate_gradient(
            input_tensor,
            self._linear_gramian,
            regularization_lambda=self.eta,
            n_iterations=self.cg_iterations,
        )
        return self.eta * preconditioned_step - self.nonlinear_op(input_tensor)

    def forward(self, y, iterations):
        initial_point = self.eta * self.initial_point(y)
        running_term = initial_point
        accumulator = initial_point

        for _ in range(iterations):
            running_term = self.single_block(running_term)
            accumulator = accumulator + running_term

        return accumulator


class SketchedNeumannNet(nn.Module):
    """A preconditioned Neumann network using a sketched version of X'X + λI.

    Attributes:
        dim: The feature dimension. When inputs are square images, this
            is the number of rows / columns of the image.
        rank: The rank of the Nystrom approximation.
        linear_op: The linear operator.
        nonlinear_op: The nonlinear operator, which is often a learned component.
        iterations: The number of iterations to unroll.
        eta: The coefficient in front of the added identity for preconditioning.
        sketch_op: The Nystrom sketch of `linear_op`.
        sketch_inverse_op: The inverse of the Nystrom sketch of `linear_op`.
    """

    dim: int
    rank: int
    linear_op: LinearOperator
    nonlinear_op: nn.Module
    iterations: int
    eta: torch.Tensor
    sketch_op: NystromApproxBlur | NystromApproxBlurGaussian
    sketch_inverse_op: NystromApproxBlurInverse

    def __init__(
        self,
        linear_operator: LinearOperator,
        sketched_operator: NystromApproxBlur | NystromApproxBlurGaussian,
        nonlinear_operator: nn.Module,
        lambda_initial_val: float = 0.1,
        iterations: int = 6,
    ):
        super(SketchedNeumannNet, self).__init__()
        self.linear_op = linear_operator
        self.sketch_op = sketched_operator
        self.nonlinear_op = nonlinear_operator
        self.iterations = iterations

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = "linear_param_"
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(
            name="eta",
            param=nn.Parameter(torch.tensor(lambda_initial_val), requires_grad=True),
        )

        if type(linear_operator) is not GaussianBlur:
            raise NotImplementedError("Only available for `GaussianBlur` operators.")
        else:
            self.sketch_inverse_op = NystromApproxBlurInverse(
                self.sketch_op,
                self.eta,
            )

    def single_block(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.eta * self.sketch_inverse_op(input_tensor) - self.nonlinear_op(
            input_tensor
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        initial_point = self.eta * self.sketch_inverse_op(
            self.linear_op.adjoint(input_tensor)
        )
        running_term = initial_point
        accumulator = initial_point
        for _ in range(self.iterations):
            running_term = self.single_block(running_term)
            accumulator = accumulator + running_term

        return accumulator
