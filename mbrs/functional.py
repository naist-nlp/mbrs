from typing import Optional

import torch
from torch import Tensor


def expectation(matrix: Tensor, lprobs: Optional[Tensor] = None) -> Tensor:
    """Compute expectation values for each row.

    Args:
        matrix (Tensor): Input matrix of shape `(H, R)`.
        lprobs (Tensor, optional): Log-probabilities for each column of shape `(R,)`.

    Returns:
        Tensor: Expected values for each row of shape `(H,)`.
    """
    if lprobs is None:
        return matrix.mean(dim=-1)
    else:
        if list(lprobs.shape) != list(matrix.shape)[1:]:
            raise ValueError(
                f"`weights` must have {list(matrix.shape)[1:]} elements, but got {list(lprobs.shape)}"
            )

        return (
            matrix * lprobs.softmax(dim=-1, dtype=torch.float32).to(matrix)[None, :]
        ).sum(dim=-1)
