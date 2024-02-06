from mbrs import registry

from .base import DecoderReferenceBased, DecoderReferenceless

register, get_decoder = registry.setup("decoder")

from .cbmbr import DecoderCBMBR
from .cbmbr_c2f import DecoderCBMBRC2F
from .mbr import DecoderMBR
from .pruning_mbr import DecoderPruningMBR

__all__ = [
    "DecoderReferenceBased",
    "DecoderReferenceless",
    "DecoderMBR",
    "DecoderCBMBR",
    "DecoderCBMBRC2F",
    "DecoderPruningMBR",
]
