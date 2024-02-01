from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from mbrs.metrics.base import Metric, MetricReferenceless
from mbrs.modules import topk


class DecoderBase(abc.ABC):
    """Decoder base class."""

    @dataclass
    class Config:
        """Configuration for the decoder."""

    @dataclass
    class Output:
        """
        - idx (list[int]): Index numbers of the n-best hypotheses.
        - sentence (list[str]): Sentences of the n-best hypotheses.
        - score (list[float]): Scores of the n-best hypotheses.
        """

        idx: list[int]
        sentence: list[str]
        score: list[float]

    def _topk(
        self, x: npt.NDArray[np.float32], k: int = 1
    ) -> tuple[list[float], list[int]]:
        """Return the top-k largest elements and corresponding indices.

        Args:
            x (NDArray[np.float32]): Input 1-D array.
            k (int): Return the top-k values and indices.

        Returns:
            tuple[list[float], list[int]]
              - list[float]: The top-k values.
              - list[int]: The top-k indices.
        """
        values, indices = topk(x, k=min(k, len(x)), largest=True)
        return values.tolist(), indices.tolist()


class DecoderReferenceBased(DecoderBase, metaclass=abc.ABCMeta):
    """Decoder base class for strategies that use references like MBR decoding."""

    def __init__(self, cfg: DecoderReferenceBased.Config, metric: Metric) -> None:
        self.cfg = cfg
        self.metric = metric

    @abc.abstractmethod
    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
    ) -> DecoderReferenceBased.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.

        Returns:
            Decoder.Output: The n-best hypotheses.
        """


class DecoderReferenceless(DecoderBase, metaclass=abc.ABCMeta):
    """Decoder base class for reference-free strategies."""

    def __init__(
        self, cfg: DecoderReferenceless.Config, metric: MetricReferenceless
    ) -> None:
        self.cfg = cfg
        self.metric = metric

    @abc.abstractmethod
    def decode(
        self, hypotheses: list[str], source: str, nbest: int = 1
    ) -> DecoderReferenceless.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            source (str): A source.
            nbest (int): Return the n-best hypotheses.

        Returns:
            Decoder.Output: The n-best hypotheses.
        """
