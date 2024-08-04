import pytest
import torch

from mbrs.metrics import MetricCOMET, MetricCOMETkiwi, MetricXCOMET


@pytest.fixture(scope="session")
def metric_comet():
    return MetricCOMET(MetricCOMET.Config())


@pytest.fixture(scope="session")
def metric_cometkiwi():
    return MetricCOMETkiwi(MetricCOMETkiwi.Config())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available on this machine."
)
@pytest.fixture(scope="session")
def metric_xcomet():
    return MetricXCOMET(MetricXCOMET.Config())
