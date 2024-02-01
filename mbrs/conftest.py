import pytest

from mbrs.metrics import MetricCOMET, MetricCOMETQE


@pytest.fixture(scope="session")
def metric_comet():
    return MetricCOMET(MetricCOMET.Config())


@pytest.fixture(scope="session")
def metric_comet_qe():
    return MetricCOMETQE(MetricCOMETQE.Config())
