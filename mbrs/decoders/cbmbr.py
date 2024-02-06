from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mbrs import timer
from mbrs.metrics import MetricCacheable
from mbrs.modules.kmeans import Kmeans

from . import register
from .mbr import DecoderMBR


@register("cbmbr")
class DecoderCBMBR(DecoderMBR):
    """Centroid-Based MBR decoder class.

    - Time complexity: O(Nk)
    - Space complexity: O(Nk)
    where k << N.
    """

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - ncentroids (int): Number of centroids.
        - niter (int): Number of k-means iteration
        - kmeanspp (bool): Initialize the centroids using k-means++.
        - seed (bool): Random seed.
        """

        ncentroids: int = 8
        niter: int = 3
        kmeanspp: bool = True
        seed: int = 0

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
    ) -> DecoderCBMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.

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
        kmeans = Kmeans(
            min(self.cfg.ncentroids, len(references)),
            references_ir.size(-1),
            kmeanspp=self.cfg.kmeanspp,
        )
        centroids, _ = kmeans.train(
            references_ir, niter=self.cfg.niter, seed=self.cfg.seed
        )
        with timer.measure("expectation"):
            expected_scores = self.metric.pairwise_scores_from_ir(
                hypotheses_ir, centroids, source_ir
            ).mean(dim=-1)
        topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
