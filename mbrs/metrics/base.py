from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from mbrs import functional, timer
from mbrs.modules.kmeans import Kmeans


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
        references_lists: list[list[str]],
        sources: Optional[list[str]] = None,
    ) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references_lists (list[list[str]]): Lists of references.
            sources (list[str], optional): Sources.

        Returns:
            float: The corpus score.
        """
        return sum(
            [
                self.scores(hypotheses, references, sources).sum().item()
                for references in references_lists
            ]
        ) / (len(hypotheses) * len(references_lists))


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


class MetricCacheable(Metric, metaclass=abc.ABCMeta):
    """Base class for cacheable metrics.

    This class supports to cache intermediate representations of sentences."""

    @dataclass
    class Cache(metaclass=abc.ABCMeta):
        """Intermediate representations of sentences."""

        @abc.abstractmethod
        def __len__(self) -> int:
            """Return the length of cache."""

        @abc.abstractmethod
        def __getitem__(
            self, key: int | Sequence[int] | slice | Tensor
        ) -> MetricCacheable.Cache:
            """Get the items."""

        @abc.abstractmethod
        def repeat(self, n: int) -> MetricCacheable.Cache:
            """Repeat the representations by n times.

            Args:
                n (int): The number of repetition.

            Returns:
                Cache: The repeated cache.
            """

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Return the size of embedding dimension."""

    @abc.abstractmethod
    def encode(self, sentences: list[str]) -> Cache:
        """Encode the given sentences into their intermediate representations.

        Args:
            sentences (list[str]): Input sentences.

        Returns:
            MetricCacheable.Cache: Intermediate representations.
        """

    @abc.abstractmethod
    def out_proj(
        self,
        hypotheses_ir: Cache,
        references_ir: Cache,
        sources_ir: Optional[Cache] = None,
    ) -> Tensor:
        """Forward the output projection layer.

        Args:
            hypotheses_ir (Cache): N intermediate representations of hypotheses.
            references_ir (Cache): N intermediate representations of references.
            sources_ir (Cache, optional): N intermediate representations of sources.

        Returns:
            Tensor: N scores.
        """

    def scores_from_ir(
        self,
        hypotheses_ir: Cache,
        references_ir: Cache,
        sources_ir: Optional[Cache] = None,
    ) -> Tensor:
        """Calculate the scores of the given hypotheses from the intermediate representations.

        Args:
            hypotheses_ir (Cache): N hypotheses.
            references_ir (Cache): N references.
            sources_ir (Cache, optional): N sources.

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
        return self.scores(
            [hypothesis],
            [reference],
            [source] if source is not None else None,
        ).item()

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
        hypotheses_ir: Cache,
        references_ir: Cache,
        source_ir: Optional[Cache] = None,
    ) -> Tensor:
        """Calculate the pairwise scores from the intermediate representations.

        Args:
            hypotheses_ir (Cache): Hypotheses.
            references_ir (Cache): References.
            source_ir (Cache, optional): A source.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        H = len(hypotheses_ir)
        R = len(references_ir)
        if source_ir is not None:
            source_ir = source_ir.repeat(H)

        scores = []
        for i in range(R):
            with timer.measure("score") as t:
                t.set_delta_ncalls(H)
                scores.append(
                    self.scores_from_ir(
                        hypotheses_ir, references_ir[i].repeat(H), source_ir
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


class MetricAggregatableCache(
    MetricAggregatable, MetricCacheable, metaclass=abc.ABCMeta
):
    """Base class for metrics that can aggregate the cache.

    This class supports to aggregate intermediate representations of sentences."""

    @dataclass
    class Cache(MetricCacheable.Cache, metaclass=abc.ABCMeta):
        """Intermediate representations of sentences."""

        @abc.abstractmethod
        def aggregate(
            self, reference_lprobs: Optional[Tensor] = None
        ) -> MetricAggregatableCache.Cache:
            """Aggregate the cached representations.

            Args:
                reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
                  The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

            Returns:
                Cache: An aggregated representation.
            """

        def cluster(
            self, kmeans: Kmeans
        ) -> tuple[MetricAggregatableCache.Cache, Tensor]:
            """Cluster the cached representations.

            Args:
                kmeans (Kmeans): k-means class to perform clustering.

            Returns:
                tuple[Cache, Tensor]:
                  - Cache: Centroid representations.
                  - Tensor: N assigned IDs.
            """
            raise NotImplementedError(type(self).__name__)

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
            aggregated_reference_ir = references_ir.aggregate(reference_lprobs)

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
        return self.scores(hypotheses, sources=sources).mean().cpu().float().item()
