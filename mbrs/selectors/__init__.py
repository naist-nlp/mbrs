from mbrs import registry

from .base import Selector, register, get_selector

from .diverse import SelectorDiverse
from .nbest import SelectorNbest

# Singleton of default selector
SELECTOR_NBEST = SelectorNbest(SelectorNbest.Config())

__all__ = [
    "Selector",
    "SelectorNbest",
    "SelectorDiverse",
    "register",
    "get_selector",
    "SELECTOR_NBEST",
]
