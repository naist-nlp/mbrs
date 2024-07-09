import pytest
import torch

from mbrs.metrics import MetricCOMET, MetricCOMETQE, MetricXCOMET, MetricBLEURT


@pytest.fixture(scope="session")
def metric_comet():
    return MetricCOMET(MetricCOMET.Config())


@pytest.fixture(scope="session")
def metric_cometqe():
    return MetricCOMETQE(MetricCOMETQE.Config())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available on this machine."
)
@pytest.fixture(scope="session")
def metric_xcomet():
    return MetricXCOMET(MetricXCOMET.Config())


@pytest.fixture(scope="session")
def metric_bleurt():
    return MetricBLEURT(MetricBLEURT.Config())
