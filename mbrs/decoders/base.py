from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from mbrs.metrics.base import Metric, MetricBase, MetricReferenceless


class DecoderBase(abc.ABC):
    """Decoder base class."""

    def __init__(self, cfg: DecoderBase.Config, metric: MetricBase) -> None:
        self.cfg = cfg
        self.metric = metric

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


class DecoderReferenceBased(DecoderBase, metaclass=abc.ABCMeta):
    """Decoder base class for strategies that use references like MBR decoding."""

    metric: Metric

    @abc.abstractmethod
    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderReferenceBased.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            Decoder.Output: The n-best hypotheses.
        """


class DecoderReferenceless(DecoderBase, metaclass=abc.ABCMeta):
    """Decoder base class for reference-free strategies."""

    metric: MetricReferenceless

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
