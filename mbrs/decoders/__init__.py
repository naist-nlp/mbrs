import enum

from mbrs import registry

from .base import DecoderBase, DecoderReferenceBased, DecoderReferenceless

register, get_decoder = registry.setup("decoder")

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
    "DecoderMBR",
    "DecoderAggregateMBR",
    "DecoderCentroidMBR",
    "DecoderProbabilisticMBR",
    "DecoderPruningMBR",
    "DecoderRerank",
]

Decoders = enum.Enum("Decoders", registry.get_registry("decoder"))
