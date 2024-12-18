import pytest

from mbrs.metrics import MetricBLEU, MetricCOMET, MetricCOMETkiwi, Metrics
from mbrs.selectors import SelectorDiverse, SelectorNbest


@pytest.fixture(scope="session")
def metric_comet():
    return MetricCOMET(MetricCOMET.Config())


@pytest.fixture(scope="session")
def metric_cometkiwi():
    return MetricCOMETkiwi(MetricCOMETkiwi.Config())


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


def should_skip(markexpr: str, markers: list[str]) -> bool:
    if markers:
        if not markexpr:
            return True
    else:
        return False

    for marker in markers:
        if f"not {marker}" in markexpr:
            return True
        elif marker in markexpr:
            return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    markexpr: str = config.option.markexpr
    skip_marker = pytest.mark.skip(reason="Excluded by default.")
    for item in items:
        markers = [marker for marker in item.keywords if marker.startswith("metrics_")]
        if should_skip(markexpr, markers):
            item.add_marker(skip_marker)
