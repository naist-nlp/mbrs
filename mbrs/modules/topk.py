import numpy as np


def topk(x: np.ndarray, k: int, largest: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-k elements and corresponding indices.

    Args:
        x (np.ndarray): Input 1-D array.
        k (int): Return the top-k values and indices.
        largest (bool): If True, the top-k largest elements will be returned.
          (default: True)

    Returns:
        tuple[np.ndarray, np.ndarray]
          - np.ndarray: The top-k values.
          - np.ndarray: The top-k indices.
    """

    if largest:
        indices = np.argpartition(x, -k)[-k:]
        argsort = np.argsort(-x[indices])
    else:
        indices = np.argpartition(x, k)[:k]
        argsort = np.argsort(x[indices])

    indices = np.take_along_axis(indices, argsort, 0)
    values = np.take_along_axis(x, indices, 0)

    return (values, indices)
