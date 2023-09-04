import torch
import torch.nn as nn


def complex_conj(x):
    assert x.shape[1] == 2
    return torch.stack((x[:, 0, ...], -x[:, 1, ...]), dim=1)


def torchdotproduct(x, y):
    return torch.sum(x.view(x.shape[0], -1) * y.view(y.shape[0], -1), dim=1)


def single_cg_iteration(x, d, g, b, ATA, regularization_lambda):
    def regATA(input, ATA):
        return ATA(input) + regularization_lambda * input

    Qd = regATA(d, ATA)
    dQd = torchdotproduct(d, Qd)
    alpha = -torchdotproduct(g, d) / dQd
    # We write (x * y.T).T since broadcasting is along the last dimension.
    x = x + (alpha * d.T).T
    g = regATA(x, ATA) - b
    gQd = torchdotproduct(g, Qd)
    beta = gQd / dQd
    # We write (x * y.T).T since broadcasting is along the last dimension.
    d = -g + (beta * d.T).T
    return x, d, g


# This function solves the system ATA x = ATy, where initial_point is supposed
# to be ATy. This can be backpropagated through.
def conjugate_gradient(initial_point, ATA, regularization_lambda, n_iterations=10):
    x = torch.zeros_like(initial_point)
    d = initial_point
    g = -d
    for _ in range(n_iterations):
        x, d, g = single_cg_iteration(
            x, d, g, initial_point, ATA, regularization_lambda
        )
    return x


def complex_dotproduct(x, y):
    return torchdotproduct(complex_conj(x), y)


def single_cg_iteration_MRI(rTr, x, r, p, ATA, regularization_lambda):
    batch_size = x.shape[0]

    def regATA(input):
        return ATA(input) + regularization_lambda * input

    Ap = regATA(p)

    rTr = rTr.view(batch_size, 1, 1, 1)
    alpha = rTr / complex_dotproduct(p, Ap).view(batch_size, 1, 1, 1)

    x_new = x + alpha * p
    r_new = r - alpha * Ap
    rTr_new = complex_dotproduct(r_new, r_new)
    rTr_new = rTr_new.view(batch_size, 1, 1, 1)

    beta = rTr_new / rTr
    p_new = r + beta * p
    return rTr_new, x_new, r_new, p_new


def conjugate_gradient_MRI(initial_point, ATA, regularization_lambda, n_iterations=10):
    """Strightforward implementation of MoDLs code"""
    x = torch.zeros_like(initial_point)
    r = initial_point
    p = initial_point
    rTr = complex_dotproduct(r, r)
    for ii in range(n_iterations):
        rTr, x, r, p = single_cg_iteration_MRI(rTr, x, r, p, ATA, regularization_lambda)
    return x
