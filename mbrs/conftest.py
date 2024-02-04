import pytest

from mbrs.metrics import MetricCOMET, MetricCOMETQE


@pytest.fixture(scope="session")
def metric_comet():
    return MetricCOMET(MetricCOMET.Config())


@pytest.fixture(scope="session")
def metric_cometqe():
    return MetricCOMETQE(MetricCOMETQE.Config())
