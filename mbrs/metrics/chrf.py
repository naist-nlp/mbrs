from __future__ import annotations

from dataclasses import dataclass

from sacrebleu.metrics.chrf import CHRF

from . import Metric, register


@register("chrf")
class MetricChrF(Metric):
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
