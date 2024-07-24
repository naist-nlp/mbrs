from mbrs import registry

from .base import (
    Metric,
    MetricAggregatable,
    MetricBase,
    MetricCacheable,
    MetricReferenceless,
)

register, get_metric = registry.setup("metric")

from .bleu import MetricBLEU
from .bleurt import MetricBLEURT
from .chrf import MetricChrF
from .comet import MetricCOMET
from .cometkiwi import MetricCOMETkiwi
from .ter import MetricTER
from .xcomet import MetricXCOMET

__all__ = [
    "MetricBase",
    "Metric",
    "MetricAggregatable",
    "MetricCacheable",
    "MetricReferenceless",
    "MetricBLEU",
    "MetricChrF",
    "MetricCOMET",
    "MetricCOMETkiwi",
    "MetricTER",
    "MetricXCOMET",
    "MetricBLEURT",
]
