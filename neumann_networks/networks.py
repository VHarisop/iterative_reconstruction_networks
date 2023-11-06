import torch
import torch.nn as nn

from .linalg import LinearOperator, conjugate_gradient


class NeumannNet(nn.Module):
    """A Neumann network.

    Attributes:
        measurement_op: The forward measurement operator.
        nonlinear_op: The nonlinear operator.
        num_blocks: The number of Neumann blocks.
        eta: A scaling factor for the Neumann series.
    """

    measurement_operator: LinearOperator
    nonlinear_operator: nn.Module
    num_blocks: int
    eta: torch.Tensor

    def __init__(
        self,
        measurement_operator: LinearOperator,
        nonlinear_operator: nn.Module,
        num_blocks: int,
        eta: float = 0.1,
    ):
        super(NeumannNet, self).__init__()
        self.measurement_operator = measurement_operator
        self.nonlinear_operator = nonlinear_operator
        self.num_blocks = num_blocks
        self.register_parameter(
            name="eta", param=nn.Parameter(torch.tensor(eta), requires_grad=True)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass of the network.

        Args:
            inputs: The input to the network (assumed to be batched).

        Returns:
            The output of the network.
        """
        initial_point = self.eta * self.measurement_operator.adjoint(inputs)
        running_term = initial_point
        accumulator = initial_point
        for _ in range(self.num_blocks):
            running_term = (
                running_term
                - self.eta
                * self.measurement_operator.adjoint(
                    self.measurement_operator(running_term)
                )
                - self.nonlinear_operator(running_term)
            )
            accumulator = accumulator + running_term

        return accumulator


class PreconditionedNeumannNet(nn.Module):
    """A preconditioned Neumann network.

    Attributes:
        measurement_operator: The forward measurement operator.
        nonlinear_operator: The nonlinear operator.
        num_blocks: The number of Neumann blocks.
        cg_iterations: The number of CG iterations to unroll.
        eta: The preconditioning strength.
    """

    measurement_operator: LinearOperator
    nonlinear_operator: nn.Module
    num_blocks: int
    cg_iterations: int
    eta: torch.Tensor

    def __init__(
        self,
        measurement_operator: LinearOperator,
        nonlinear_operator: nn.Module,
        num_blocks: int,
        cg_iterations: int,
        eta: float,
    ):
        super(PreconditionedNeumannNet, self).__init__()
        self.measurement_operator = measurement_operator
        self.nonlinear_operator = nonlinear_operator
        self.num_blocks = num_blocks
        self.cg_iterations = cg_iterations
        self.register_parameter(
            name="eta", param=nn.Parameter(torch.tensor(eta), requires_grad=True)
        )

    def _batched_cg(self, rhs: torch.Tensor) -> torch.Tensor:
        gram_op = (
            lambda z: self.measurement_operator.adjoint(self.measurement_operator(z))
            + self.eta * z
        )
        return torch.vmap(
            lambda inputs: conjugate_gradient(gram_op, inputs, self.cg_iterations),
            in_dims=0,
        )(rhs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute a forward pass of the network.

        Args:
            inputs: The input to the network (assumed to be batched).

        Returns:
            The output of the network.
        """
        gram_op = lambda x: (
            self.measurement_operator.adjoint(self.measurement_operator(x))
            + self.eta * x
        )
        initial_point = conjugate_gradient(
            gram_op,
            self.measurement_operator.adjoint(inputs),
            self.cg_iterations,
        )
        running_term = initial_point
        accumulator = initial_point
        for _ in range(self.num_blocks):
            preconditioned_step = conjugate_gradient(
                gram_op,
                running_term,
                self.cg_iterations,
            )
            running_term = self.eta * preconditioned_step - self.nonlinear_operator(
                running_term
            )
            accumulator = accumulator + running_term

        return accumulator
