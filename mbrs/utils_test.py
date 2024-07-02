import pytest
import torch

from . import utils


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available on this machine."
)
def test_to_device():
    device = torch.device("cuda:0")

    for x in [1, 1.0, "a", True]:
        assert utils.to_device(x, device) == x

    assert utils.to_device(torch.ones(1), device).device == device
    assert utils.to_device({"a": torch.ones(1)}, device)["a"].device == device
    assert utils.to_device([torch.ones(1)], device)[0].device == device
    assert utils.to_device((torch.ones(1),), device)[0].device == device
