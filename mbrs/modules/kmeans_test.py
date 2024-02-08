from typing import List, Tuple, Union

import pytest
import torch

from .kmeans import Kmeans

N = 100
D = 8
C = 4


def is_equal_shape(
    a: Union[torch.Tensor, torch.Size],
    b: Union[torch.Tensor, torch.Size, List[int], Tuple[int]],
) -> bool:
    """Returns whether a and b have the same shape.

    Args:
        a (Union[torch.Tensor, torch.Size]): An input tensor.
        b (Union[torch.Tensor, torch.Size, List[int], Tuple[int]):
          An input tensor compared the shape with a.

    Returns:
        bool: Whether a and b have the same shape.

    Raises:
        NotImplementedError: When an unsupported type object is given.
    """
    if isinstance(a, torch.Tensor):
        a_shape = a.shape
    elif isinstance(a, torch.Size):
        a_shape = a
    else:
        raise NotImplementedError(f"Type of `a` (`{type(a)}`) is not supported.")

    if isinstance(b, torch.Tensor):
        b_shape = b.shape
    elif isinstance(b, torch.Size):
        b_shape = b
    elif isinstance(b, (list, tuple)):
        b_shape = torch.Size(b)
    else:
        raise NotImplementedError(f"Type of `b` (`{type(b)}`) is not supported.")
    return a_shape == b_shape


class TestKmeans:
    @pytest.mark.parametrize("kmeanspp", [False, True])
    def test___init__(self, kmeanspp: bool):
        kmeans = Kmeans(kmeanspp)
        assert kmeans.kmeanspp == kmeanspp

    def test_init_kmeanspp(self):
        x = torch.rand(N, D)
        kmeans = Kmeans(kmeanspp=True)
        rng = torch.Generator(x.device)
        rng = rng.manual_seed(0)
        centroids = kmeans.init_kmeanspp(x, rng, C)
        assert is_equal_shape(centroids, [C, D])

    def test_assign(self):
        x = torch.rand(N, D)
        centroids = torch.rand(C, D)
        kmeans = Kmeans()
        assigns = kmeans.assign(x, centroids)
        expected = ((x[:, None] - centroids[None, :]) ** 2).sum(dim=-1).argmin(dim=1)
        assert torch.equal(assigns, expected)

    @pytest.mark.parametrize("kmeanspp", [False, True])
    def test_train(self, kmeanspp: bool):
        torch.manual_seed(0)
        kmeans = Kmeans(kmeanspp)
        x = torch.rand(N, D)
        centroids, assigns = kmeans.train(x, C)
        assert is_equal_shape(centroids, [C, D])
        assert is_equal_shape(assigns, [N])
