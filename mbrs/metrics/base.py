from __future__ import annotations

import abc
import contextlib
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor


class MetricBase(abc.ABC):
    """Base metric class."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config: ...

    @property
    def device(self) -> torch.device:
        """Returns the device of the metric object."""
        return torch.device("cpu")

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
        values, indices = torch.topk(x, k=min(k, len(x)), largest=self.HIGHER_IS_BETTER)
        return values.tolist(), indices.tolist()

    def argbest(self, x: Tensor) -> Tensor:
        """Return the index of the best element.

        Args:
            x (Tensor): Input 1-D array.

        Returns:
            Tensor: A scalar tensor of the best index.
        """
        if self.HIGHER_IS_BETTER:
            return torch.argmax(x)
        return torch.argmin(x)


class Metric(MetricBase, metaclass=abc.ABCMeta):
    """Base metric class."""

    @abc.abstractmethod
    def score(
        self, hypothesis: str, reference: str, source: Optional[str] = None
    ) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            reference (str): A reference.
            source (str, optional): A source.

        Returns:
            float: The score of the given hypothesis.
        """

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> Tensor:
        """Calculate the pairwise scores.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        return Tensor(
            [[self.score(hyp, ref, source) for ref in references] for hyp in hypotheses]
        )

    def expected_scores(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> Tensor:
        """Calculate the expected scores for each hypothesis.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            Tensor: The expected scores for each hypothesis.
        """
        return self.pairwise_scores(hypotheses, references, source).mean(dim=1)


class MetricCacheable(Metric, metaclass=abc.ABCMeta):
    """Base class for cacheable metrics.

    This class supports to cache intermediate representations of the encoder."""

    @abc.abstractmethod
    def encode(self, sentences: list[str]) -> Tensor:
        """Encode the given sentences into their intermediate representations.

        Args:
            sentences (list[str]): Input sentences.

        Returns:
            Tensor: Intermediate representations of shape `(N, D)` where `N` is the
              number of hypotheses and `D` is a size of the embedding dimension.
        """

    @abc.abstractmethod
    def out_proj(
        self,
        hypotheses_ir: Tensor,
        references_ir: Tensor,
        source_ir: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward the output projection layer.

        Args:
            hypotheses_ir (Tensor): Intermediate representations of hypotheses.
            references_ir (Tensor): Intermediate representations of references.
            source_ir (Tensor, optional): Intermediate representations of a source.

        Returns:
            Tensor: N scores.
        """

    def score(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            reference (str): A reference.
            source (str, optional): A source.

        Returns:
            float: The score of the given hypothesis.
        """
        return self.out_proj(
            self.encode([hypothesis]),
            self.encode([reference]),
            self.encode([source]) if source is not None else None,
        ).item()

    def pairwise_scores_from_ir(
        self,
        hypotheses_ir: Tensor,
        references_ir: Tensor,
        source_ir: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate the pairwise scores from the intermediate representations.

        Args:
            hypotheses_ir (Tensor): Hypotheses.
            references_ir (Tensor): References.
            source_ir (Tensor, optional): A source.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        H, D = hypotheses_ir.size()
        R, _ = references_ir.size()
        if source_ir is not None:
            source_ir = source_ir.repeat(R, 1)

        scores = []
        for i in range(H):
            scores.append(
                self.out_proj(
                    hypotheses_ir[i, :].repeat(R, 1), references_ir, source_ir
                )
            )
        return torch.vstack(scores).float()

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> Tensor:
        """Calculate the pairwise scores.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        hypotheses_ir = self.encode(hypotheses)
        references_ir = self.encode(references)
        source_ir = self.encode([source]) if source is not None else None
        return self.pairwise_scores_from_ir(hypotheses_ir, references_ir, source_ir)


class MetricReferenceless(MetricBase, metaclass=abc.ABCMeta):
    """Base class for reference-less metrics like quality estimation."""

    @abc.abstractmethod
    def score(self, hypothesis: str, source: str) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            source (str): A source.

        Returns:
            float: The score of the given hypothesis.
        """

    def scores(self, hypotheses: list[str], source: str) -> Tensor:
        """Calculate the scores of hypotheses.

        Args:
            hypotheses (list[str]): Hypotheses.
            source (str): A source.

        Returns:
            Tensor: The scores of hypotheses.
        """
        return Tensor([self.score(hyp, source) for hyp in hypotheses])
