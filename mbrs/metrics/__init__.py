from __future__ import annotations

import enum

from mbrs import registry

from .base import (
    Metric,
    MetricAggregatable,
    MetricAggregatableCache,
    MetricBase,
    MetricCacheable,
    MetricReferenceless,
    get_metric,
    register,
)
from .bertscore import MetricBERTScore
from .bleu import MetricBLEU
from .bleurt import MetricBLEURT
from .chrf import MetricChrF
from .comet import MetricCOMET
from .cometkiwi import MetricCOMETkiwi
from .metricx import MetricMetricX
from .ter import MetricTER
from .xcomet import MetricXCOMET

__all__ = [
    "MetricBase",
    "Metric",
    "MetricAggregatable",
    "MetricAggregatableCache",
    "MetricCacheable",
    "MetricReferenceless",
    "get_metric",
    "register",
    "MetricBERTScore",
    "MetricBLEU",
    "MetricChrF",
    "MetricCOMET",
    "MetricCOMETkiwi",
    "MetricMetricX",
    "MetricTER",
    "MetricXCOMET",
    "MetricBLEURT",
]


class MetricEnum(str, enum.Enum): ...


Metrics = MetricEnum(
    "Metrics",
    {k: k for k in registry.get_registry(Metric | MetricReferenceless).keys()},
)
