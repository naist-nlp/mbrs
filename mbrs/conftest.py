import pytest

from mbrs.metrics import MetricCOMET


@pytest.fixture(scope="session")
def metric_comet():
    return MetricCOMET(MetricCOMET.Config())
