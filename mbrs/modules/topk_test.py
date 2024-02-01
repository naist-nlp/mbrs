import numpy as np
import pytest

from .topk import topk


@pytest.mark.parametrize("n", [1, 3, 16, 256])
@pytest.mark.parametrize("k", [1, 3, 64, 256])
@pytest.mark.parametrize("largest", [True, False])
def test_topk(n: int, k: int, largest: bool):
    x = np.random.rand(n).astype(np.float32)
    k = min(k, n)
    values, indices = topk(x, k=k, largest=largest)
    if largest:
        expected_indices = np.argsort(-x)[:k]
        expected_values = -np.sort(-x)[:k]
    else:
        expected_indices = np.argsort(x)[:k]
        expected_values = np.sort(x)[:k]
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(values, expected_values)
