How to define a new decoder
===========================

Examples
~~~~~~~~

This tutorial explains how to define a new decoder.
The below example implements the naive MBR decoding and extends the output object to return other features.

1. Inherit an abstract class defined in :code:`mbrs.decoders.base`.

   - :code:`DecoderReferenceBased` is mainly used for MBR decoding that returns the N most probable hypotheses using sets of hypotheses and pseudo-references.

   .. code:: python

      from mbrs.decoders.base import DecoderReferenceBased


      class DecoderMBRWithAllScores(DecoderReferenceBased):
          """Naive MBR decoder class."""

2. Define the configuration dataclass if you need to add options.

   - Configuration dataclass :code:`DecoderMBRWithAllScores.Config` should inherit that of the parent class for consistency.

   .. code:: python

      from dataclasses import dataclass

      from mbrs.decoders.base import DecoderReferenceBased


      class DecoderMBRWithAllScores(DecoderReferenceBased):
          """Naive MBR decoder class."""

          @dataclass
          class Config(DecoderMBRWithAllScores.Config):
              """Naive MBR decoder configuration."""

              sort_scores: bool = False


3. Child classes of :code:`DecoderReferenceBased` requires to implement the :code:`decode()` method.

   .. code:: python

      from dataclasses import dataclass
      from typing import Optional

      from mbrs.decoders.base import DecoderReferenceBased


      class DecoderMBRWithAllScores(DecoderReferenceBased):
          """Naive MBR decoder class."""

          @dataclass
          class Config(DecoderMBRWithAllScores.Config):
              """Naive MBR decoder configuration."""

              sort_scores: bool = False

          def decode(
              self,
              hypotheses: list[str],
              references: list[str],
              source: Optional[str] = None,
              nbest: int = 1,
              reference_lprobs: Optional[Tensor] = None,
          ) -> DecoderMBRWithAllScores.Output:
              expected_scores = self.metric.expected_scores(
                  hypotheses, references, source, reference_lprobs=reference_lprobs
              )
              topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)
              return self.Output(
                  idx=topk_indices,
                  sentence=[hypotheses[idx] for idx in topk_indices],
                  score=topk_scores,
              )

4. In this example, we extend the output dataclass to include all expected scores.

   - :code:`DecoderMBRWithAllScores.Output` needs to inherit the parent output dataclass.

   .. code:: python

      from dataclasses import dataclass
      from typing import Optional

      from torch import Tensor

      from mbrs.decoders.base import DecoderReferenceBased


      class DecoderMBRWithAllScores(DecoderReferenceBased):
          """Naive MBR decoder class."""

          @dataclass
          class Config(DecoderMBRWithAllScores.Config):
              sort_scores: bool = False

          @dataclass
          class Output(DecoderReferenceBased.Output):
              all_scores: Optional[Tensor] = None

          def decode(
              self,
              hypotheses: list[str],
              references: list[str],
              source: Optional[str] = None,
              nbest: int = 1,
              reference_lprobs: Optional[Tensor] = None,
          ) -> DecoderMBRWithAllScores.Output:
              expected_scores = self.metric.expected_scores(
                  hypotheses, references, source, reference_lprobs=reference_lprobs
              )
              topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)

              if self.cfg.sort_scores:
                  all_scores = expected_scores.sort(dim=-1, descending=self.metric.HIGH_IS_BETTER)
              else:
                  all_scores = expected_scores

              return self.Output(
                  idx=topk_indices,
                  sentence=[hypotheses[idx] for idx in topk_indices],
                  score=topk_scores,
                  all_scores=all_scores,
              )

5. Finally, register the class to be called from CLI.

   - Just add :code:`@register("mbr_with_all_scores")` to the class definition.

   .. code:: python

      from dataclasses import dataclass
      from typing import Optional

      from torch import Tensor

      from mbrs.decoders.base import DecoderReferenceBased, register


      @register("mbr_with_all_scores")
      class DecoderMBRWithAllScores(DecoderReferenceBased):
          """Naive MBR decoder class."""

          @dataclass
          class Config(DecoderMBRWithAllScores.Config):
              sort_scores: bool = False

          @dataclass
          class Output(DecoderReferenceBased.Output):
              all_scores: Optional[Tensor] = None

          def decode(
              self,
              hypotheses: list[str],
              references: list[str],
              source: Optional[str] = None,
              nbest: int = 1,
              reference_lprobs: Optional[Tensor] = None,
          ) -> DecoderMBRWithAllScores.Output:
              expected_scores = self.metric.expected_scores(
                  hypotheses, references, source, reference_lprobs=reference_lprobs
              )
              topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)

              if self.cfg.sort_scores:
                  all_scores = expected_scores.sort(dim=-1, descending=self.metric.HIGH_IS_BETTER)
              else:
                  all_scores = expected_scores

              return self.Output(
                  idx=topk_indices,
                  sentence=[hypotheses[idx] for idx in topk_indices],
                  score=topk_scores,
                  all_scores=all_scores,
              )

.. note::

   All methods should have the same types for both inputs and outputs as the base class.
