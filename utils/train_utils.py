import hashlib
from typing import Dict

import torch
from torch.utils.data import DataLoader

from operators.operator import LinearOperator


@torch.no_grad()
def evaluate_batch_loss(
    solver: torch.nn.Module,
    loss_fn: torch.nn.Module,
    measurement_operator: LinearOperator,
    data_loader: DataLoader,
    device: torch.device,
    **solver_kwargs,
) -> float:
    """Evaluate the batch loss.

    Args:
        solver: The solver reconstructing the samples.
        loss_fn: A loss function measuring the reconstruction quality.
        measurement_operator: The forward measurement operator.
        data_loader: A data loader for the dataset.
        device: The device on which to perform computations.
        **solver_kwargs: Keyword arguments for the solver, such as the
            number of iterations.

    Returns:
        The batch loss on the dataset."""
    loss = 0.0
    num_batches = 0
    for idx, (sample_batch, _) in enumerate(data_loader):
        num_batches += 1
        sample_batch = sample_batch.to(device)
        y = measurement_operator(sample_batch)
        reconstruction = solver(y, **solver_kwargs)
        reconstruction = torch.clamp(reconstruction, -1, 1)
        loss += loss_fn(reconstruction, sample_batch).item()
    return loss / num_batches


def hash_dict(dictionary: Dict) -> str:
    """Create a hash from a dictionary.

    Args:
        dictionary: The dictionary to hash.

    Returns:
        The hash string.
    """
    dict2hash = ""
    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]
        dict2hash += "%s_%s_" % (str(k), str(v))
    return hashlib.md5(dict2hash.encode()).hexdigest()
