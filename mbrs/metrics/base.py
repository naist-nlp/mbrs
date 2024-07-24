from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer


class MetricBase(abc.ABC):
    """Base metric class."""

    def __init__(self, cfg: MetricBase.Config):
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

    def scores(
        self,
        hypotheses: list[str],
        references: list[str],
        sources: Optional[list[str]] = None,
    ) -> Tensor:
        """Calculate the scores of the given hypotheses.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str]): N references.
            sources (list[str], optional): N sources.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """
        with timer.measure("score") as t:
            t.set_delta_ncalls(len(hypotheses))
            if sources is None:
                return Tensor(
                    [self.score(hyp, ref) for hyp, ref in zip(hypotheses, references)]
                )
            else:
                return Tensor(
                    [
                        self.score(hyp, ref, src)
                        for hyp, ref, src in zip(hypotheses, references, sources)
                    ]
                )

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
        with timer.measure("score") as t:
            t.set_delta_ncalls(len(hypotheses) * len(references))
            return Tensor(
                [
                    [self.score(hyp, ref, source) for ref in references]
                    for hyp in hypotheses
                ]
            )

    def expected_scores(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        reference_lprobs: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate the expected scores for each hypothesis.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            Tensor: The expected scores for each hypothesis.
        """
        with timer.measure("expectation"):
            return functional.expectation(
                self.pairwise_scores(hypotheses, references, source),
                lprobs=reference_lprobs,
            )

    def corpus_score(
        self,
        hypotheses: list[str],
        references: list[str],
        sources: Optional[list[str]] = None,
    ) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            sources (list[str], optional): Sources.

        Returns:
            float: The corpus score.
        """
        if sources is None:
            return sum(
                [self.score(hyp, ref) for hyp, ref in zip(hypotheses, references)]
            ) / len(hypotheses)
        else:
            return sum(
                [
                    self.score(hyp, ref, src)
                    for hyp, ref, src in zip(hypotheses, references, sources)
                ]
            ) / len(hypotheses)


class MetricAggregatable(Metric, metaclass=abc.ABCMeta):
    """Base class for aggregatable metrics.

    This class supports reference aggregation."""

    @abc.abstractmethod
    def expected_scores_reference_aggregation(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        reference_lprobs: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate the expected scores for each hypothesis.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            Tensor: The expected scores for each hypothesis.
        """


class MetricCacheable(MetricAggregatable, metaclass=abc.ABCMeta):
    """Base class for cacheable metrics.

    This class supports to cache intermediate representations of the encoder."""

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Return the size of embedding dimension."""

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
        sources_ir: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward the output projection layer.

        Args:
            hypotheses_ir (Tensor): Intermediate representations of hypotheses.
            references_ir (Tensor): Intermediate representations of references.
            sources_ir (Tensor, optional): Intermediate representations of sources.

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

    def scores_from_ir(
        self,
        hypotheses_ir: Tensor,
        references_ir: Tensor,
        sources_ir: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate the scores of the given hypotheses from the intermediate representations.

        Args:
            hypotheses_ir (Tensor): N hypotheses.
            references_ir (Tensor): N references.
            sources_ir (Tensor, optional): N sources.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """
        H = len(hypotheses_ir)
        with timer.measure("score") as t:
            t.set_delta_ncalls(H)
            if sources_ir is None:
                return self.out_proj(hypotheses_ir, references_ir)
            else:
                return self.out_proj(hypotheses_ir, references_ir, sources_ir)

    def scores(
        self,
        hypotheses: list[str],
        references: list[str],
        sources: Optional[list[str]] = None,
    ) -> Tensor:
        """Calculate the scores of the given hypotheses.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str]): N references.
            source (list[str], optional): N sources.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """

        return self.scores_from_ir(
            self.encode(hypotheses),
            self.encode(references),
            self.encode(sources) if sources is not None else None,
        )

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
            source_ir = source_ir.repeat(H, 1)

        scores = []
        for i in range(R):
            with timer.measure("score") as t:
                t.set_delta_ncalls(H)
                scores.append(
                    self.out_proj(
                        hypotheses_ir, references_ir[i, :].repeat(H, 1), source_ir
                    )[:, None]
                )
        return torch.cat(scores, dim=-1)

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
        with timer.measure("encode/hypotheses"):
            hypotheses_ir = self.encode(hypotheses)
        if hypotheses == references:
            references_ir = hypotheses_ir
        else:
            with timer.measure("encode/references"):
                references_ir = self.encode(references)
        if source is None:
            source_ir = None
        else:
            with timer.measure("encode/source"):
                source_ir = self.encode([source])
        return self.pairwise_scores_from_ir(hypotheses_ir, references_ir, source_ir)

    def expected_scores_reference_aggregation(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        reference_lprobs: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate the expected scores for each hypothesis.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            Tensor: The expected scores for each hypothesis.
        """
        with timer.measure("encode/hypotheses"):
            hypotheses_ir = self.encode(hypotheses)
        if hypotheses == references:
            references_ir = hypotheses_ir
        else:
            with timer.measure("encode/references"):
                references_ir = self.encode(references)
        if source is None:
            source_ir = None
        else:
            with timer.measure("encode/source"):
                source_ir = self.encode([source])

        with timer.measure("aggregate/references"):
            if reference_lprobs is not None:
                aggregated_reference_ir = (
                    references_ir
                    * reference_lprobs.to(references_ir)
                    .softmax(dim=-1, dtype=torch.float32)
                    .to(references_ir)[:, None]
                ).sum(dim=0, keepdim=True)
            else:
                aggregated_reference_ir = references_ir.mean(dim=0, keepdim=True)

        with timer.measure("expectation"):
            return self.pairwise_scores_from_ir(
                hypotheses_ir, aggregated_reference_ir, source_ir=source_ir
            ).mean(dim=-1)


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

    def scores(self, hypotheses: list[str], sources: list[str]) -> Tensor:
        """Calculate the scores of hypotheses.

        Args:
            hypotheses (list[str]): N hypotheses.
            sources (list[str]): N sources.

        Returns:
            Tensor: The scores of hypotheses.
        """
        return Tensor([self.score(hyp, src) for hyp, src in zip(hypotheses, sources)])

    def corpus_score(self, hypotheses: list[str], sources: list[str]) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            sources (list[str]): Sources.

        Returns:
            float: The corpus score.
        """
        return sum(self.scores(hypotheses, sources=sources).cpu().float().item()) / len(
            hypotheses
        )
