from __future__ import annotations

from typing import Optional

from . import DecoderReferenceBased, register


@register("mbr")
class DecoderMBR(DecoderReferenceBased):
    """MBR decoder class."""

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
    ) -> DecoderMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.

        Returns:
            DecoderMBR.Output: The n-best hypotheses.
        """
        expected_scores = self.metric.expected_scores(hypotheses, references, source)
        topk_scores, topk_indices = self._topk(expected_scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
