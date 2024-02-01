import numpy as np
import pytest

from .topk import topk


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("largest", [True, False])
def test_topk(k: int, largest: bool):
    x = np.random.rand(16).astype(np.float32)
    values, indices = topk(x, k=k, largest=largest)
    if largest:
        expected_indices = np.argsort(-x)[:k]
        expected_values = -np.sort(-x)[:k]
    else:
        expected_indices = np.argsort(x)[:k]
        expected_values = np.sort(x)[:k]
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(values, expected_values)
