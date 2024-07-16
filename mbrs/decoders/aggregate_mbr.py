from __future__ import annotations

from typing import Optional

from torch import Tensor

from mbrs.metrics import MetricAggregatable

from . import register
from .mbr import DecoderMBR


@register("aggregate_mbr")
class DecoderAggregateMBR(DecoderMBR):
    """MBR decoding with reference aggregation.

    - Time complexity: O(N)
    - Space complexity: O(N)

    References:
        J. DeNero et al., 2009,
        "Fast Consensus Decoding over Translation Forests".
        https://aclanthology.org/P09-1064/

        J. Vamvas and R. Sennrich, 2024,
        "Linear-time Minimum Bayes Risk Decoding with Reference Aggregation".
        https://arxiv.org/abs/2402.04251
    """

    def __init__(
        self, cfg: DecoderAggregateMBR.Config, metric: MetricAggregatable
    ) -> None:
        super().__init__(cfg, metric)

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderAggregateMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderAggregateMBR.Output: The n-best hypotheses.
        """
        assert isinstance(self.metric, MetricAggregatable)
        expected_scores = self.metric.expected_scores_reference_aggregation(
            hypotheses, references, source=source, reference_lprobs=reference_lprobs
        )
        topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
