from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from mbrs.metrics import Metric, MetricCOMET
from mbrs.modules.kmeans import Kmeans

from . import register
from .mbr import DecoderMBR


@register("cbmbr")
class DecoderCBMBR(DecoderMBR):
    """Centroid-Based MBR decoder class."""

    SUPPORTED_METRICS = (MetricCOMET,)
    cfg: Config

    def __init__(self, cfg: DecoderCBMBR.Config, metric: Metric) -> None:
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
        hypotheses_set: list[list[str]],
        references_set: list[list[str]],
        source_set: Optional[list[str]] = None,
    ) -> list[DecoderMBR.Output]:
        """Select the best hypothesis based on the strategy.

        Args:
            hypotheses_set (list[list[str]]): Set of hypotheses.
            references_set (list[list[str]]): Set of references.
            source_set (list[str], optional): Set of each source.

        Returns:
            list[DecoderMBR.Output]: The best hypotheses.
        """
        assert isinstance(self.metric, self.SUPPORTED_METRICS)

        outputs = []
        bsz = self.metric.cfg.batch_size
        for i, (hyps, refs) in enumerate(zip(hypotheses_set, references_set)):
            H, R = len(hyps), min(self.cfg.ncentroids, len(refs))
            scores = torch.zeros((H, R), dtype=torch.float32, device=self.metric.device)
            hyp_embeds = self.metric.compute_sentence_embedding(hyps, bsz)
            ref_embeds = self.metric.compute_sentence_embedding(refs, bsz)
            src_embeds = (
                self.metric.compute_sentence_embedding([source_set[i]], bsz).repeat(
                    H, 1
                )
                if source_set is not None
                else None
            )
            kmeans = Kmeans(R, ref_embeds.size(-1), kmeanspp=self.cfg.kmeanspp)
            centroids, assigns = kmeans.train(
                ref_embeds, niter=self.cfg.niter, seed=self.cfg.seed
            )
            for k in range(R):
                scores[:, k] = self.metric.compute_output_projection(
                    hyp_embeds, centroids[k].repeat(H, 1), src_embeds
                )
            scores = scores.cpu().float().numpy()

            expected_utility = scores.mean(axis=1)
            best_idx = int(np.argmax(expected_utility))
            outputs.append(
                self.Output(
                    idx=best_idx,
                    sentence=hyps[best_idx],
                    score=float(expected_utility[best_idx]),
                )
            )
        return outputs
