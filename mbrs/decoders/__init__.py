from mbrs import registry

from .base import DecoderBase, DecoderReferenceBased, DecoderReferenceless

register, get_decoder = registry.setup("decoder")

from .mbr import DecoderMBR
from .aggregate_mbr import DecoderAggregateMBR
from .cbmbr import DecoderCBMBR
from .pmbr import DecoderProbabilisticMBR
from .pruning_mbr import DecoderPruningMBR
from .rerank import DecoderRerank

__all__ = [
    "DecoderBase",
    "DecoderReferenceBased",
    "DecoderReferenceless",
    "DecoderMBR",
    "DecoderAggregateMBR",
    "DecoderCBMBR",
    "DecoderProbabilisticMBR",
    "DecoderPruningMBR",
    "DecoderRerank",
]
