import enum

from mbrs import registry

from .base import (
    DecoderBase,
    DecoderReferenceBased,
    DecoderReferenceless,
    register,
    get_decoder,
)


from .aggregate_mbr import DecoderAggregateMBR
from .centroid_mbr import DecoderCentroidMBR
from .mbr import DecoderMBR
from .probabilistic_mbr import DecoderProbabilisticMBR
from .pruning_mbr import DecoderPruningMBR
from .rerank import DecoderRerank

__all__ = [
    "DecoderBase",
    "DecoderReferenceBased",
    "DecoderReferenceless",
    "register",
    "get_decoder",
    "DecoderMBR",
    "DecoderAggregateMBR",
    "DecoderCentroidMBR",
    "DecoderProbabilisticMBR",
    "DecoderPruningMBR",
    "DecoderRerank",
]
