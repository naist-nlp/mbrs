from typing import Optional

import pytest
import torch
from torch import Tensor

from .functional import expectation

H = 8
R = 5

A = torch.rand(H, R)


@pytest.mark.parametrize(
    "lprobs", [None, torch.FloatTensor([-3.0, -2.4, -1.9, -5.5, -4.3])]
)
def test_expectation(lprobs: Optional[Tensor]):
    e = expectation(A, lprobs=lprobs)
    assert list(e.shape) == [H]
    if lprobs is None:
        assert torch.allclose(e, A.mean(dim=-1))
    else:
        with pytest.raises(ValueError):
            expectation(torch.rand(H, R - 1), lprobs=lprobs)

        weights = lprobs.exp() / lprobs.exp().sum(dim=-1, keepdim=True)
        assert torch.allclose(e, (A @ weights[None, :].T).squeeze(-1))
