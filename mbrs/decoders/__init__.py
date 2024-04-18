from mbrs import registry

from .base import DecoderReferenceBased, DecoderReferenceless

register, get_decoder = registry.setup("decoder")

from .cbmbr import DecoderCBMBR
from .mbr import DecoderMBR
from .pruning_mbr import DecoderPruningMBR
from .rerank import DecoderRerank

__all__ = [
    "DecoderReferenceBased",
    "DecoderReferenceless",
    "DecoderMBR",
    "DecoderCBMBR",
    "DecoderPruningMBR",
    "DecoderRerank",
]
