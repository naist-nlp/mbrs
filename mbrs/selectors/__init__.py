from mbrs import registry

from .base import Selector

register, get_selector = registry.setup("selector")

from .diverse import SelectorDiverse
from .nbest import SelectorNbest

__all__ = ["Selector", "SelectorNbest", "SelectorDiverse"]
