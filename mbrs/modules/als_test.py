from typing import List, Tuple, Union

import torch

from .als import MatrixFactorizationALS

N = 64
M = 128
r = 8


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


class TestMatrixFactorizationALS:
    def test_factorize(self):
        als = MatrixFactorizationALS(rank=r)

        torch.manual_seed(0)
        matrix = torch.rand(N, M)

        X, Y = als.factorize(matrix)
        assert is_equal_shape(X, [N, r])
        assert is_equal_shape(Y, [M, r])
