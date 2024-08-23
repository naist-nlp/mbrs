from __future__ import annotations

import concurrent.futures
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
from sacrebleu.metrics.bleu import BLEU, MAX_NGRAM_ORDER
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
        - num_workers (int): Number of workers for multiprocessing.
        """

        lowercase: bool = False
        force: bool = False
        tokenize: Optional[str] = None
        smooth_method: str = "exp"
        smooth_value: Optional[float] = None
        max_ngram_order: int = 4
        effective_order: bool = True
        trg_lang: str = ""
        num_workers: int = 8

    cfg: Config

    @dataclass
    class AggregatedReference:
        """Aggregated reference representation.

        - ngrams (Counter[tuple[str, ...]]): Bags of expected n-gram counts.
        - length (float): Expected length of references.
        """

        ngrams: Counter[tuple[str, ...]]
        length: float

    def __init__(self, cfg: MetricBLEU.Config):
        super().__init__(cfg)
        self.scorer = self._initialize_bleu(cfg)

    @staticmethod
    def _initialize_bleu(cfg: MetricBLEU.Config) -> BLEU:
        scorer = BLEU(
            lowercase=cfg.lowercase,
            force=cfg.force,
            tokenize=cfg.tokenize,
            smooth_method=cfg.smooth_method,
            smooth_value=cfg.smooth_value,
            max_ngram_order=cfg.max_ngram_order,
            effective_order=cfg.effective_order,
            trg_lang=cfg.trg_lang,
        )
        MetricBLEU._score_worker.scorer = scorer
        return scorer

    def score(self, hypothesis: str, reference: str, *_, **__) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): Hypothesis.
            reference (str): Reference.

        Returns:
            float: The score of the given hypothesis.
        """
        return self.scorer.sentence_score(hypothesis, [reference]).score

    @staticmethod
    def _score_worker(hypothesis: str, reference: str, *_, **__) -> float:
        """Calculate the score of the given hypothesis.

        Beacause ja-mecab tokenizer cannot be pickled, this method is necessary to use
        multiprocessing.

        Args:
            hypothesis (str): Hypothesis.
            reference (str): Reference.

        Returns:
            float: The score of the given hypothesis.

        Todo:
            - Replace this method with a better logic.
        """
        return MetricBLEU._score_worker.scorer.sentence_score(
            hypothesis, [reference]
        ).score

    def scores(self, hypotheses: list[str], references: list[str], *_, **__) -> Tensor:
        """Calculate the scores of the given hypotheses.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str]): N references.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.cfg.num_workers,
            initializer=self._initialize_bleu,
            initargs=(self.cfg,),
        ) as executor:
            with timer.measure("score") as t:
                t.set_delta_ncalls(len(hypotheses))
                return Tensor(
                    list(
                        executor.map(
                            self._score_worker,
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
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.cfg.num_workers,
            initializer=self._initialize_bleu,
            initargs=(self.cfg,),
        ) as executor:
            with timer.measure("score") as t:
                t.set_delta_ncalls(len(hypotheses) * len(references))

                return Tensor(
                    list(
                        executor.map(
                            self._score_worker,
                            *zip(*itertools.product(hypotheses, references)),
                            chunksize=len(hypotheses),
                        )
                    )
                ).view(len(hypotheses), len(references))

    def corpus_score(
        self, hypotheses: list[str], references: list[str], *_, **__
    ) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            float: The corpus score.
        """
        return self.scorer.corpus_score(hypotheses, [references]).score

    @staticmethod
    def _compute_bleu(
        correct: list[float],
        total: list[float],
        sys_len: float,
        ref_len: float,
        smooth_method: str = "none",
        smooth_value: Optional[float] = None,
        effective_order: bool = False,
        max_ngram_order: int = MAX_NGRAM_ORDER,
    ) -> float:
        """Computes BLEU score from its sufficient statistics with smoothing.

        Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
        Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

        - none: No smoothing.
        - floor: Method 1 (requires small positive value (0.1 in the paper) to be set)
        - add-k: Method 2 (Generalizing Lin and Och, 2004)
        - exp: Method 3 (NIST smoothing method i.e. in use with mteval-v13a.pl)

        This method extends the original sacrebleu implementation to treat expected n-grams.

        Args:
            correct (list[float]): List of counts of correct ngrams, 1 <= n <= max_ngram_order.
            total (list[float]): List of counts of total ngrams, 1 <= n <= max_ngram_order
            sys_len (float): The cumulative system length
            ref_len (float): The cumulative reference length
            smooth_method (str): The smoothing method to use ('floor', 'add-k', 'exp' or 'none')
            smooth_value (float, optional): The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
            effective_order (bool): If `True`, stop including n-gram orders for which precision is 0. This should be
                `True`, if sentence-level BLEU will be computed.
            max_ngram_order (int): If given, it overrides the maximum n-gram order (default: 4) when computing precisions.

        Returns:
            float: A BLEU score.
        """
        assert (
            smooth_method in BLEU.SMOOTH_DEFAULTS.keys()
        ), "Unknown smooth_method {smooth_method!r}"

        # Fetch the default value for floor and add-k
        if smooth_value is None:
            smooth_value = BLEU.SMOOTH_DEFAULTS[smooth_method]

        # Compute brevity penalty
        if sys_len < ref_len:
            bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
        else:
            bp = 1.0

        # n-gram precisions
        precisions = [0.0 for x in range(max_ngram_order)]

        # Early stop if there are no matches (#141)
        if not any(correct):
            return 0.0

        smooth_mteval = 1.0
        eff_order = max_ngram_order
        for n in range(1, len(precisions) + 1):
            if smooth_method == "add-k" and n > 1:
                correct[n - 1] += smooth_value
                total[n - 1] += smooth_value

            if total[n - 1] == 0:
                break

            # If the system guesses no i-grams, 1 <= i <= max_ngram_order,
            # the BLEU score is 0 (technically undefined). This is a problem for sentence
            # level BLEU or a corpus of short sentences, where systems will get
            # no credit if sentence lengths fall under the max_ngram_order threshold.
            # This fix scales max_ngram_order to the observed maximum order.
            # It is only available through the API and off by default
            if effective_order:
                eff_order = n

            if correct[n - 1] == 0:
                if smooth_method == "exp":
                    smooth_mteval *= 2
                    precisions[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
                elif smooth_method == "floor":
                    precisions[n - 1] = 100.0 * smooth_value / total[n - 1]
            else:
                precisions[n - 1] = 100.0 * correct[n - 1] / total[n - 1]

        # Compute BLEU score
        score = bp * math.exp(
            sum(
                [
                    math.log(p) if p > 0.0 else -9999999999.0
                    for p in precisions[:eff_order]
                ]
            )
            / eff_order
        )

        return score

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
        if reference_lprobs is not None:
            lprobs = reference_lprobs.log_softmax(dim=-1, dtype=torch.float32).tolist()
        else:
            lprobs = [-math.log(num_references)] * num_references

        reference_stats = self.scorer._cache_references([references])
        reference_ngrams: list[Counter[tuple[str, ...]]] = [
            stat["ref_ngrams"] for stat in reference_stats
        ]

        expected_reference_length = sum(
            [
                math.exp(math.log(stat["ref_lens"][0]) + lprob)
                if stat["ref_lens"][0] > 0.0
                else 0.0
                for stat, lprob in zip(reference_stats, lprobs)
            ]
        )

        acc_ngrams: Counter[tuple[str, ...]] = Counter()
        for i, ngrams in enumerate(reference_ngrams):
            for ngram in ngrams:
                # Note: Counter has float values.
                ngrams[ngram] = math.exp(math.log(ngrams[ngram]) + lprobs[i])
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
                correct = [0.0 for i in range(self.scorer.max_ngram_order)]
                total = correct[:]
                for hyp_ngram, hyp_count in hyp_ngrams.items():
                    # n-gram order
                    n = len(hyp_ngram) - 1
                    # count hypothesis n-grams
                    total[n] += float(hyp_count)
                    # count matched n-grams
                    if hyp_ngram in aggregated_reference.ngrams:
                        correct[n] += float(
                            min(hyp_count, aggregated_reference.ngrams[hyp_ngram])
                        )

                stats = [hyp_len, aggregated_reference.length] + correct + total
                expected_scores[i] = self._compute_bleu(
                    correct=stats[2 : 2 + self.scorer.max_ngram_order],
                    total=stats[2 + self.scorer.max_ngram_order :],
                    sys_len=float(stats[0]),
                    ref_len=float(stats[1]),
                    smooth_method=self.scorer.smooth_method,
                    smooth_value=self.scorer.smooth_value,
                    effective_order=self.scorer.effective_order,
                    max_ngram_order=self.scorer.max_ngram_order,
                )

        return expected_scores
