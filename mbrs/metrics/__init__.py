from mbrs import registry

from .base import Metric, MetricNeural, MetricReferenceless

register, get_cls = registry.setup("metric")

from .bleu import MetricBLEU
from .chrf import MetricChrF
from .comet import MetricCOMET
from .comet_qe import MetricCOMETQE

__all__ = [
    "Metric",
    "MetricNeural",
    "MetricReferenceless",
    "MetricBLEU",
    "MetricChrF",
    "MetricCOMET",
    "MetricCOMETQE",
]
