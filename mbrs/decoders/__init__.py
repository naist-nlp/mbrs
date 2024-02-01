from mbrs import registry

from .base import DecoderReferenceBased, DecoderReferenceless

register, get_cls = registry.setup("decoder")

from .cbmbr import DecoderCBMBR
from .mbr import DecoderMBR

__all__ = [
    "DecoderReferenceBased",
    "DecoderReferenceless",
    "DecoderMBR",
    "DecoderCBMBR",
]
