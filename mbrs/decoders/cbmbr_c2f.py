from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from mbrs import timer
from mbrs.metrics import MetricCacheable

from . import register
from .cbmbr import DecoderCBMBR
from .mbr import DecoderMBR


@register("cbmbr_c2f")
class DecoderCBMBRC2F(DecoderCBMBR):
    """Centroid-Based MBR decoder class.

    - Time complexity: O(k^2 + N)
    - Space complexity: O(k^2 + N)
    where k << N.
    """

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - ncentroids_hyp (int): Number of centroids for the hypothesis side.
        - ncentroids_ref_coarse (int): Number of centroids for the reference side in the
            coarse search.
        - ncentroids_ref_fine (int): Number of centroids for the reference side in the
            fine search.
        - niter (int): Number of k-means iteration
        - kmeanspp (bool): Initialize the centroids using k-means++.
        - seed (bool): Random seed.
        """

        ncentroids_hyp: int = 8
        ncentroids_ref_coarse: int = 1
        ncentroids_ref_fine: int = 8
        niter: int = 3
        kmeanspp: bool = True
        seed: int = 0

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
    ) -> DecoderCBMBRC2F.Output:
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

        # Encode sentences
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

        # Clustering
        with timer.measure("clustering/references/coarse"):
            ref_coarse_centroids, ref_coarse_assigns = self.kmeans.train(
                references_ir,
                min(self.cfg.ncentroids_ref_coarse, len(references)),
                niter=self.cfg.niter,
                seed=self.cfg.seed,
            )
        if self.cfg.ncentroids_ref_coarse == self.cfg.ncentroids_ref_fine:
            ref_fine_centroids = ref_coarse_centroids
            ref_fine_assigns = ref_coarse_assigns
        else:
            with timer.measure("clustering/references/fine"):
                ref_fine_centroids, ref_fine_assigns = self.kmeans.train(
                    references_ir,
                    min(self.cfg.ncentroids_ref_fine, len(references)),
                    niter=self.cfg.niter,
                    seed=self.cfg.seed,
                )
        if (
            hypotheses == references
            and self.cfg.ncentroids_hyp == self.cfg.ncentroids_ref_fine
        ):
            hyp_centroids = ref_fine_centroids
            hyp_assigns = ref_fine_assigns
        else:
            with timer.measure("clustering/hypotheses"):
                hyp_centroids, hyp_assigns = self.kmeans.train(
                    hypotheses_ir,
                    min(self.cfg.ncentroids_hyp, len(hypotheses)),
                    niter=self.cfg.niter,
                    seed=self.cfg.seed,
                )

        # Coarse search
        with timer.measure("expectation/coarse"):
            expected_scores = self.metric.pairwise_scores_from_ir(
                hyp_centroids, ref_coarse_centroids, source_ir
            ).mean(dim=-1)
            best_cluster_order = torch.argsort(
                expected_scores, dim=-1, descending=self.metric.HIGHER_IS_BETTER
            ).tolist()
            orig_indices = torch.arange(len(hypotheses_ir), device=hypotheses_ir.device)
            num_pruned_hypotheses = 0
            pruned_hypotheses_irs = []
            pruned_hypotheses_ids = []
            for k in best_cluster_order:
                k_mask = hyp_assigns.eq(k)
                hypotheses_k = hypotheses_ir[k_mask]
                pruned_hypotheses_irs.append(hypotheses_k)
                pruned_hypotheses_ids.append(orig_indices[k_mask])
                num_pruned_hypotheses += len(hypotheses_k)
                if num_pruned_hypotheses >= nbest:
                    break
            pruned_hypotheses_ir = torch.cat(pruned_hypotheses_irs)
            pruned_hypotheses_id = torch.cat(pruned_hypotheses_ids)

        # Fine search
        with timer.measure("expectation/fine"):
            expected_scores = self.metric.pairwise_scores_from_ir(
                pruned_hypotheses_ir, ref_fine_centroids, source_ir
            ).mean(dim=-1)
        topk_scores, pruned_topk_indices = self.metric.topk(expected_scores, k=nbest)
        topk_indices = pruned_hypotheses_id[pruned_topk_indices].tolist()
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
