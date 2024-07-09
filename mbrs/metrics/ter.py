from __future__ import annotations

from dataclasses import dataclass

from sacrebleu.metrics.ter import TER

from . import Metric, register


@register("ter")
class MetricTER(Metric):
    """TER metric class."""

    HIGHER_IS_BETTER: bool = False

    @dataclass
    class Config(Metric.Config):
        """TER metric configuration.

        - normalized (bool): Enable character normalization.
          By default, normalizes a couple of things such as newlines being stripped,
          retrieving XML encoded characters, and fixing tokenization for punctuation.
          When 'asian_support' is enabled, also normalizes specific Asian (CJK)
          character sequences, i.e. split them down to the character level.
        - no_punct (bool): Remove punctuation. Can be used in conjunction with
          'asian_support' to also remove typical punctuation markers in Asian languages
          (CJK).
        - asian_support (bool): Enable special treatment of Asian characters.
          This option only has an effect when 'normalized' and/or 'no_punct' is enabled.
          If 'normalized' is also enabled, then Asian (CJK) characters are split down to
          the character level. If 'no_punct' is enabled alongside 'asian_support',
          specific unicode ranges for CJK and full-width punctuations are also removed.
        - case_sensitive (bool): If `True`, does not lowercase sentences.
        """

        normalized: bool = False
        no_punct: bool = False
        asian_support: bool = False
        case_sensitive: bool = False

    def __init__(self, cfg: MetricTER.Config):
        self.scorer = TER(
            normalized=cfg.normalized,
            no_punct=cfg.no_punct,
            asian_support=cfg.asian_support,
            case_sensitive=cfg.case_sensitive,
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
