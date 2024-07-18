from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from mbrs import functional, timer
from mbrs.metrics import MetricCacheable
from mbrs.modules.kmeans import Kmeans

from . import register
from .mbr import DecoderMBR


@register("centroid_mbr")
class DecoderCentroidMBR(DecoderMBR):
    """Centroid-Based MBR decoder class.

    - Time complexity: O(Nk)
    - Space complexity: O(Nk)
    where k << N.

    References:
        H. Deguchi et al., 2024.
        "Centroid-Based Efficient Minimum Bayes Risk Decoding".
        https://arxiv.org/abs/2402.11197
    """

    def __init__(self, cfg: Config, metric: MetricCacheable) -> None:
        super().__init__(cfg, metric)
        self.kmeans = Kmeans(kmeanspp=self.cfg.kmeanspp)

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - ncentroids (int): Number of centroids.
        - niter (int): Number of k-means iteration
        - count_weight: (bool) Weight the scores with counts.
        - kmeanspp (bool): Initialize the centroids using k-means++.
        - seed (bool): Random seed.
        """

        ncentroids: int = 8
        niter: int = 3
        count_weight: bool = False
        kmeanspp: bool = True
        seed: int = 0

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderCentroidMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderCBMBR.Output: The n-best hypotheses.
        """
        assert isinstance(self.metric, MetricCacheable)

        with timer.measure("encode/hypotheses"):
            hypotheses_ir = self.metric.encode(hypotheses)
        if hypotheses == references:
            references_ir = hypotheses_ir
        else:
            with timer.measure("encode/references"):
                references_ir = self.metric.encode(references)
        if source is None:
            source_ir = None
        else:
            with timer.measure("encode/source"):
                source_ir = self.metric.encode([source])
        centroids, assigns = self.kmeans.train(
            references_ir,
            min(self.cfg.ncentroids, len(references)),
            niter=self.cfg.niter,
            seed=self.cfg.seed,
        )

        lprobs = None
        if self.cfg.count_weight:
            centroid_ids, counts_nonzero = assigns.unique(return_counts=True)
            counts = centroid_ids.new_zeros(len(centroids))
            counts[centroid_ids] = counts_nonzero
            lprobs = counts.log()
        elif reference_lprobs is not None:
            lprobs = (
                centroids.new_zeros(len(centroids))
                .scatter_add(dim=-1, index=assigns.unique(), src=reference_lprobs.exp())
                .log()
            )

        with timer.measure("expectation"):
            pairwise_scores = self.metric.pairwise_scores_from_ir(
                hypotheses_ir, centroids, source_ir
            )
            expected_scores = functional.expectation(pairwise_scores, lprobs=lprobs)

        topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
