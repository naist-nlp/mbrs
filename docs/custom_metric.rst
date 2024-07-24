How to define a new metric
==========================

Examples
~~~~~~~~

This tutorial explains how to define a new metric using an example of :code:`MetricTER`.

1. Inherit an abstract class defined in :code:`mbrs.metrics.base`.

   - :code:`Metric` calculates the score between a hypothesis and reference with optionally using a source.
   - If the lower score means better, the class variable :code:`HIGHER_IS_BETTER` should be set to :code:`False`.

   .. code:: python

      from mbrs.metrics.base import Metric


      class MetricTER(Metric):
          """TER metric class."""

          HIGHER_IS_BETTER: bool = False

2. Define the configuration dataclass.

   - Configuration dataclass :code:`MetricTER.Config` should inherit that of the parent class for consistency.
   - :code:`__init__()` receives an instance of configuration dataclass :code:`cfg: MetricTER.Config` and setup the scorer function.

   .. code:: python

      from dataclasses import dataclass

      from sacrebleu.metrics.ter import TER

      from mbrs.metrics.base import Metric


      class MetricTER(Metric):
          """TER metric class."""

          HIGHER_IS_BETTER: bool = False

          @dataclass
          class Config(Metric.Config):
              """TER metric configuration."""

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

3. Child classes of :code:`Metric` requires to implement the :code:`score()` method which calculates a score of a single example.

   - In the default, :code:`score()` is called iteratively in the MBR decoding.
   - If the metric can compute pairwise scores between hypotheses and pseudo-references in parallel, it would be better to override :code:`pairwise_scores()` to allow batch computation.

   .. code:: python

      from dataclasses import dataclass

      from sacrebleu.metrics.ter import TER

      from mbrs.metrics.base import Metric


      class MetricTER(Metric):
          """TER metric class."""

          HIGHER_IS_BETTER: bool = False

          @dataclass
          class Config(Metric.Config):
              """TER metric configuration."""

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

4. Register the class to call from CLI.

   - Just add :code:`@register("ter")` to the class definition.

   .. code:: python

      from dataclasses import dataclass

      from sacrebleu.metrics.ter import TER

      from mbrs.metrics.base import Metric, register


      @register("ter")
      class MetricTER(Metric):
          """TER metric class."""

          HIGHER_IS_BETTER: bool = False

          @dataclass
          class Config(Metric.Config):
              """TER metric configuration."""

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

Notes
~~~~~

- All methods should have the same input/output types as the base class.
