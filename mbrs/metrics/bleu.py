from __future__ import annotations

import concurrent.futures
import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.helpers import extract_all_word_ngrams
from torch import Tensor

from mbrs import timer

from . import MetricAggregatable, register


@register("bleu")
class MetricBLEU(MetricAggregatable):
    """BLEU metric class."""

    @dataclass
    class Config(MetricAggregatable.Config):
        """BLEU metric configuration.

        - lowercase (bool): If True, lowercased BLEU is computed.
        - force (bool): Ignore data that looks already tokenized.
        - tokenize (str, optional): The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the fallback default.
        - smooth_method (str): The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
        - smooth_value (float, optional): The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
        - max_ngram_order (int): If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
        - effective_order (bool): If `True`, stop including n-gram orders for which precision is 0.
          This should be `True`, if sentence-level BLEU will be computed. (default: True)
        - trg_lang (str): An optional language code to raise potential tokenizer warnings.
        """

        lowercase: bool = False
        force: bool = False
        tokenize: Optional[str] = None
        smooth_method: str = "exp"
        smooth_value: Optional[float] = None
        max_ngram_order: int = 4
        effective_order: bool = True
        trg_lang: str = ""

    @dataclass
    class AggregatedReference:
        """Aggregated reference representation.

        - ngrams (Counter[tuple[str, ...]]): Bags of expected n-gram counts.
        - length (float): Expected length of references.
        """

        ngrams: Counter[tuple[str, ...]]
        length: float

    def __init__(self, cfg: MetricBLEU.Config):
        self.scorer = BLEU(
            lowercase=cfg.lowercase,
            force=cfg.force,
            tokenize=cfg.tokenize,
            smooth_method=cfg.smooth_method,
            smooth_value=cfg.smooth_value,
            max_ngram_order=cfg.max_ngram_order,
            effective_order=cfg.effective_order,
            trg_lang=cfg.trg_lang,
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
            MetricBLEU.AggregatedReference: Aggregated reference representation.
        """
        num_references = len(references)
        reference_stats = self.scorer._cache_references([references])
        reference_ngrams: list[Counter[tuple[str, ...]]] = [
            stat["ref_ngrams"] for stat in reference_stats
        ]
        expected_reference_length = (
            sum([stat["ref_lens"][0] for stat in reference_stats]) / num_references
        )

        reference_probs = [1.0 / num_references] * num_references
        if reference_lprobs is not None:
            reference_probs = reference_lprobs.softmax(dim=-1).tolist()

        acc_ngrams: Counter[tuple[str, ...]] = Counter()
        for i, ngrams in enumerate(reference_ngrams):
            for ngram in ngrams:
                # Note: Counter has float values.
                ngrams[ngram] *= reference_probs[i]
            acc_ngrams += ngrams

        return self.AggregatedReference(acc_ngrams, expected_reference_length)

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
                # Extract n-grams for the hypothesis
                hyp_ngrams, hyp_len = extract_all_word_ngrams(
                    hypothesis, 1, self.scorer.max_ngram_order
                )

                # Count the stats
                # Although counter has its internal & and | operators, this is faster
                correct = [0 for i in range(self.scorer.max_ngram_order)]
                total = correct[:]
                for hyp_ngram, hyp_count in hyp_ngrams.items():
                    # n-gram order
                    n = len(hyp_ngram) - 1
                    # count hypothesis n-grams
                    total[n] += hyp_count
                    # count matched n-grams
                    if hyp_ngram in aggregated_reference.ngrams:
                        correct[n] += min(
                            hyp_count, aggregated_reference.ngrams[hyp_ngram]
                        )

                bleu_score = self.scorer._compute_score_from_stats(
                    [hyp_len, aggregated_reference.length] + correct + total
                )
                expected_scores[i] = bleu_score.score

        return expected_scores
