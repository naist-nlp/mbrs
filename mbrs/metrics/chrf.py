from __future__ import annotations

import concurrent.futures
import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

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
        """

        char_order: int = 6
        word_order: int = 0
        beta: int = 2
        lowercase: bool = False
        whitespace: bool = False
        eps_smoothing: bool = False

    @dataclass
    class AggregatedReference:
        """Aggregated reference representation.

        - ngrams (list[Counter]]): Bags of n-grams for each order.
        """

        ngrams: list[Counter]

    def __init__(self, cfg: MetricChrF.Config):
        self.scorer = CHRF(
            char_order=cfg.char_order,
            word_order=cfg.word_order,
            beta=cfg.beta,
            lowercase=cfg.lowercase,
            whitespace=cfg.whitespace,
            eps_smoothing=cfg.eps_smoothing,
        )

    def score(self, hypothesis: str, reference: str, *_) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): Hypothesis.
            reference (str): Reference.

        Returns:
            float: The score of the given hypothesis.
        """
        return self.scorer.sentence_score(hypothesis, [reference]).score

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], *_
    ) -> Tensor:
        """Calculate the pairwise scores.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
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

    def corpus_score(self, hypotheses: list[str], references: list[str], *_) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            float: The corpus score.
        """
        return self.scorer.corpus_score(hypotheses, [references]).score

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
