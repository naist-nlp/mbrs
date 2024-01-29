from mbrs import registry

from .base import Decoder

register, get_cls = registry.setup("decoder")

from .mbr import DecoderMBR

__all__ = ["Decoder", "DecoderMBR"]
