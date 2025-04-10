from __future__ import annotations

import concurrent.futures
import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import fastchrf
import torch
from sacrebleu.metrics.chrf import CHRF
from sacrebleu.metrics.helpers import extract_all_char_ngrams, extract_word_ngrams
from torch import Tensor

from mbrs import timer

from . import Metric, MetricAggregatable, register


@register("chrf")
class MetricChrF(MetricAggregatable):
    """ChrF metric class."""

    @dataclass
    class Config(Metric.Config):
        """ChrF metric configuration.

        - char_order (int): Character n-gram order.
        - word_order (int): Word n-gram order. If equals to 2, the metric is referred to as chrF++.
        - beta (int): Determine the importance of recall w.r.t precision.
        - lowercase (bool): Enable case-insensitivity.
        - whitespace (bool): If `True`, include whitespaces when extracting character n-grams.
        - eps_smoothing (bool): If `True`, applies epsilon smoothing similar to reference chrF++.py, NLTK and Moses implementations.
            Otherwise, it takes into account effective match order similar to sacreBLEU < 2.0.0.
        - num_workers (int): Number of workers for multiprocessing.
        - fastchrf (bool): Use the rust implementation of chrF.
        """

        char_order: int = 6
        word_order: int = 0
        beta: int = 2
        lowercase: bool = False
        whitespace: bool = False
        eps_smoothing: bool = False
        num_workers: int = 8
        fastchrf: bool = False

        def __post_init__(self):
            if self.fastchrf and self.word_order > 0:
                raise ValueError("fastchrf does not support the `word_order` option.")

    cfg: Config

    @dataclass
    class AggregatedReference:
        """Aggregated reference representation.

        - ngrams (list[Counter]]): Bags of n-grams for each order.
        """

        ngrams: list[Counter]

    def __init__(self, cfg: MetricChrF.Config):
        super().__init__(cfg)
        self.scorer = CHRF(
            char_order=cfg.char_order,
            word_order=cfg.word_order,
            beta=cfg.beta,
            lowercase=cfg.lowercase,
            whitespace=cfg.whitespace,
            eps_smoothing=cfg.eps_smoothing,
        )

    def _fastchrf_pairwise_scores(
        self, hypotheses_lists: list[list[str]], references_lists: list[list[str]]
    ) -> Tensor:
        """Calculate the pairwise scores using fastchrf.

        Args:
            hypotheses_lists (list[list[str]]): N lists of hypotheses.
            references_lists (list[list[str]]): N lists of references.

        Returns:
            Tensor: Score matrix of shape `(N, H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        return Tensor(
            fastchrf.pairwise_chrf(
                hypotheses_lists,
                references_lists,
                char_order=self.cfg.char_order,
                beta=float(self.cfg.beta),
                remove_whitespace=not self.cfg.whitespace,
                eps_smoothing=self.cfg.eps_smoothing,
            )
        )

    def _fastchrf_expected_scores_reference_aggregation(
        self, hypotheses_lists: list[list[str]], references_lists: list[list[str]]
    ) -> Tensor:
        """Calculate the expected scores with reference aggregation using fastchrf.

        Args:
            hypotheses_lists (list[list[str]]): N lists of hypotheses.
            references_lists (list[list[str]]): N lists of references.

        Returns:
            Tensor: Score matrix of shape `(N, H)`, where `H` is the number
              of hypotheses.
        """
        return Tensor(
            fastchrf.aggregate_chrf(
                hypotheses_lists,
                references_lists,
                char_order=self.cfg.char_order,
                beta=float(self.cfg.beta),
                remove_whitespace=not self.cfg.whitespace,
                eps_smoothing=self.cfg.eps_smoothing,
            )
        )

    def score(self, hypothesis: str, reference: str, *_, **__) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): Hypothesis.
            reference (str): Reference.

        Returns:
            float: The score of the given hypothesis.
        """
        if self.cfg.fastchrf:
            return self._fastchrf_pairwise_scores([[hypothesis]], [[reference]]).item()

        return self.scorer.sentence_score(hypothesis, [reference]).score

    def scores(self, hypotheses: list[str], references: list[str], *_, **__) -> Tensor:
        """Calculate the scores of the given hypotheses.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str]): N references.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """
        if self.cfg.fastchrf:
            with timer.measure("score") as t:
                t.set_delta_ncalls(len(hypotheses))
                return self._fastchrf_pairwise_scores(
                    [[hypothesis] for hypothesis in hypotheses],
                    [[reference] for reference in references],
                ).flatten()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.cfg.num_workers,
        ) as executor:
            with timer.measure("score") as t:
                t.set_delta_ncalls(len(hypotheses))
                return Tensor(
                    list(
                        executor.map(
                            self.score,
                            hypotheses,
                            references,
                            chunksize=math.ceil(len(hypotheses) / self.cfg.num_workers),
                        )
                    )
                )

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], *_, **__
    ) -> Tensor:
        """Calculate the pairwise scores.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        if self.cfg.fastchrf:
            with timer.measure("score") as t:
                t.set_delta_ncalls(len(hypotheses) * len(references))
                return self._fastchrf_pairwise_scores(
                    [hypotheses], [references]
                ).squeeze(0)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.cfg.num_workers
        ) as executor:
            with timer.measure("score") as t:
                t.set_delta_ncalls(len(hypotheses) * len(references))

                return Tensor(
                    list(
                        executor.map(
                            self.score,
                            *zip(*itertools.product(hypotheses, references)),
                            chunksize=len(hypotheses),
                        )
                    )
                ).view(len(hypotheses), len(references))

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
        return self.scorer.corpus_score(hypotheses, references_lists).score

    def _aggregate_references(
        self, references: list[str], reference_lprobs: Optional[Tensor] = None
    ) -> AggregatedReference:
        """Aggregate references.

        Args:
            references (list[str]): References.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            MetricChrF.AggregatedReference: Aggregated reference representation.
        """
        num_references = len(references)
        reference_ngrams: list[list[Counter[str]]] = self.scorer._cache_references(
            [[ref] for ref in references]
        )[0]["ref_ngrams"]

        if reference_lprobs is not None:
            lprobs = reference_lprobs.log_softmax(dim=-1).tolist()
        else:
            lprobs = [-math.log(num_references)] * num_references

        acc_ngrams: defaultdict[int, Counter[str]] = defaultdict(Counter)
        for i, ngrams in enumerate(reference_ngrams):
            for order, ngram_counts in enumerate(ngrams):
                for ngram in ngram_counts:
                    # Note: Counter has float values.
                    ngram_counts[ngram] = math.exp(
                        math.log(ngram_counts[ngram]) + lprobs[i]
                    )
                acc_ngrams[order] += ngram_counts

        return self.AggregatedReference(
            [acc_ngrams[order] for order in range(len(acc_ngrams))]
        )

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
        if self.cfg.fastchrf:
            if reference_lprobs is not None:
                raise ValueError("fastchrf does not support model-based aggregation.")

            with timer.measure("expectation"):
                return self._fastchrf_expected_scores_reference_aggregation(
                    [hypotheses], [references]
                ).squeeze(0)

        with timer.measure("aggregate/references"):
            aggregated_reference = self._aggregate_references(
                references, reference_lprobs=reference_lprobs
            )

        expected_scores = torch.zeros((len(hypotheses),))
        for i, hypothesis in enumerate(hypotheses):
            with timer.measure("expectation"):
                hypothesis = self.scorer._preprocess_segment(hypothesis)

                # Extract character n-grams
                all_hyp_ngrams = extract_all_char_ngrams(
                    hypothesis, self.scorer.char_order, self.scorer.whitespace
                )

                # Check chrF+ mode to see if we'll add word n-grams as well
                if self.scorer.word_order > 0:
                    # Primitive tokenization: separate out punctuations
                    hwords = self.scorer._remove_punctuation(hypothesis)
                    _range = range(1, self.scorer.word_order + 1)
                    all_hyp_ngrams.extend(
                        [extract_word_ngrams(hwords, n) for n in _range]
                    )

                stats = []
                # Traverse all orders
                for h, r in zip(all_hyp_ngrams, aggregated_reference.ngrams):
                    stats.extend(self.scorer._get_match_statistics(h, r))
                f_score = self.scorer._compute_f_score(stats)
                expected_scores[i] = f_score

        return expected_scores
