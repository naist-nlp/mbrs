import pytest
import torch

from mbrs.metrics import MetricBLEU, MetricCOMET, MetricCOMETkiwi, Metrics, MetricXCOMET
from mbrs.selectors import SelectorDiverse, SelectorNbest


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


@pytest.fixture(
    params=[
        SelectorNbest(SelectorNbest.Config()),
        SelectorDiverse(
            SelectorDiverse.Config(
                diversity_metric=Metrics.bleu,
                diversity_metric_config=MetricBLEU.Config(effective_order=True),
            )
        ),
    ]
)
def selector(request):
    return request.param
