from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mbrs.metrics import MetricCOMET, MetricNeural
from mbrs.modules.kmeans import Kmeans

from . import register
from .mbr import DecoderMBR


@register("cbmbr")
class DecoderCBMBR(DecoderMBR):
    """Centroid-Based MBR decoder class."""

    SUPPORTED_METRICS = (MetricCOMET,)
    cfg: Config

    def __init__(self, cfg: DecoderCBMBR.Config, metric: MetricNeural) -> None:
        if not isinstance(metric, self.SUPPORTED_METRICS):
            raise ValueError(f"{type(metric)} is not supported in CBMBR.")
        super().__init__(cfg, metric)

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
        kmeanspp: bool = False
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

        assert isinstance(self.metric, self.SUPPORTED_METRICS)

        ir = self.metric.encode(hypotheses, references, source)
        kmeans = Kmeans(
            min(self.cfg.ncentroids, len(references)),
            ir.ref.size(-1),
            kmeanspp=self.cfg.kmeanspp,
        )
        centroids, _ = kmeans.train(ir.ref, niter=self.cfg.niter, seed=self.cfg.seed)
        centroid_ir = self.metric.IR(ir.hyp, centroids, ir.src)
        expected_scores = (
            self.metric.out_proj(centroid_ir).mean(dim=1).cpu().float().numpy()
        )
        topk_scores, topk_indices = self._topk(expected_scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
