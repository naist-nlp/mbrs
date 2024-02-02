from mbrs import registry

from .base import Metric, MetricNeural, MetricReferenceless

register, get_metric = registry.setup("metric")

from .bleu import MetricBLEU
from .chrf import MetricChrF
from .comet import MetricCOMET
from .cometqe import MetricCOMETQE
from .ter import MetricTER

__all__ = [
    "Metric",
    "MetricNeural",
    "MetricReferenceless",
    "MetricBLEU",
    "MetricChrF",
    "MetricCOMET",
    "MetricCOMETQE",
    "MetricTER",
]
