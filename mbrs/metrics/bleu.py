from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sacrebleu.metrics.bleu import BLEU

from . import Metric, register


@register("bleu")
class MetricBLEU(Metric):
    """BLEU metric class."""

    @dataclass
    class Config(Metric.Config):
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

    def corpus_score(self, hypotheses: list[str], references: list[str], *_) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            float: The corpus score.
        """
        return self.scorer.corpus_score(hypotheses, [references]).score
