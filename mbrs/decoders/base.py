from __future__ import annotations

import abc
from dataclasses import dataclass, fields, make_dataclass
from typing import Any, Optional

from torch import Tensor

from mbrs.metrics.base import Metric, MetricBase, MetricReferenceless
from mbrs.selectors import Selector, SelectorNbest


class DecoderBase(abc.ABC):
    """Decoder base class."""

    def __init__(
        self,
        cfg: DecoderBase.Config,
        metric: MetricBase,
        selector: Selector = SelectorNbest(SelectorNbest.Config()),
    ) -> None:
        self.cfg = cfg
        self.metric = metric
        self.selector = selector

    @property
    def maximize(self) -> bool:
        """Return `True` when maximizing the objective score."""
        return self.metric.HIGHER_IS_BETTER

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

        def __or__(self, other: Any):
            """Returns the union of dataclasses.

            Args:
                other (Any): An other dataclass.

            Returns:
                Output: New dataclass with the merged attributes of `self` and `other`.
            """
            new_fields = [(f.name, f.type, f) for f in fields(self)]
            new_fields += [
                (f.name, f.type, f)
                for f in fields(other)
                if f.name not in {f.name for f in fields(self)}
            ]
            new_dc_type = make_dataclass(
                "Output", fields=new_fields, bases=(type(self),)
            )
            attrs = {f.name: getattr(other, f.name) for f in fields(other)} | {
                f.name: getattr(self, f.name) for f in fields(self)
            }
            return new_dc_type(**attrs)

    def topk(self, x: Tensor, k: int = 1) -> tuple[list[float], list[int]]:
        """Return the top-k best elements and corresponding indices.

        Args:
            x (Tensor): Input 1-D array.
            k (int): Return the top-k values and indices.

        Returns:
            tuple[list[float], list[int]]
              - list[float]: The top-k values.
              - list[int]: The top-k indices.
        """
        return self.selector.topk(x, k=k, maximize=self.maximize)

    def argbest(self, x: Tensor) -> Tensor:
        """Return the index of the best element.

        Args:
            x (Tensor): Input 1-D array.

        Returns:
            Tensor: A scalar tensor of the best index.
        """
        return self.selector.argbest(x, maximize=self.maximize)

    def superior(self, a: float, b: float) -> bool:
        """Return whether the score `a` is superior to the score `b`.

        Args:
            a (float): A score.
            b (float): A score.

        Returns:
            bool: Return True when `a` is superior to `b`.
        """
        return self.selector.superior(a, b, maximize=self.maximize)

    def select(
        self,
        hypotheses: list[str],
        expected_scores: Tensor,
        nbest: int = 1,
        source: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Selector.Output:
        """Select the final output list.

        Args:
            hypotheses (list[str]): Hypotheses.
            expected_scores (Tensor): The expected scores for each hypothesis.
            nbest (int): Return the n-best hypotheses based on the selection rule.
            source (str, optional): A source.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            Selector.Output: Outputs.
        """
        return self.selector.select(
            hypotheses,
            expected_scores,
            nbest=nbest,
            source=source,
            maximize=self.maximize,
            *args,
            **kwargs,
        )


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
