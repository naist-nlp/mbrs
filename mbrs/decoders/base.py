from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from mbrs.metrics.base import Metric


class Decoder(abc.ABC):
    """Base decoder class."""

    def __init__(self, cfg: Config, metric: Metric) -> None:
        self.cfg = cfg
        self.metric = metric

    @dataclass
    class Config:
        """Configuration for the decoder."""

    @dataclass
    class Output:
        """
        - idx (int): An index number of the best hypothesis.
        - sentence (str): A sentence of the best hypothesis.
        - score (float): A score of the best hypothesis.
        """

        idx: int
        sentence: str
        score: float

    @abc.abstractmethod
    def decode(
        self,
        hypotheses_set: list[list[str]],
        references_set: list[list[str]],
        source_set: Optional[list[str]] = None
    ) -> list[Output]:
        """Select the best hypothesis based on the strategy.

        Args:
            hypotheses_set (list[list[str]]): Set of hypotheses.
            references_set (list[list[str]]): Set of references.
            source_set (list[str], optional): Set of each source.

        Returns:
            list[Decoder.Output]: The best hypotheses.
        """
