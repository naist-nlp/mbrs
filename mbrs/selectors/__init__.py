from mbrs import registry

from .base import Selector, register, get_selector

from .diverse import SelectorDiverse
from .nbest import SelectorNbest

__all__ = [
    "Selector",
    "SelectorNbest",
    "SelectorDiverse",
    "register",
    "get_selector",
]
