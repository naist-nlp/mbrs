from __future__ import annotations

from typing import Optional

import torch

from mbrs import timer
from mbrs.metrics import MetricCacheable
from mbrs.modules.kmeans import Kmeans

from . import register
from .cbmbr import DecoderCBMBR


@register("cbmbr_c2f")
class DecoderCBMBRC2F(DecoderCBMBR):
    """Centroid-Based MBR decoder class.

    - Time complexity: O(k^2 + N)
    - Space complexity: O(k^2 + N)
    where k << N.
    """

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
        centroids, assigns = kmeans.train(
            references_ir, niter=self.cfg.niter, seed=self.cfg.seed
        )
        with timer.measure("expectation/coarse"):
            expected_scores = self.metric.pairwise_scores_from_ir(
                centroids, centroids, source_ir
            ).mean(dim=-1)
            best_cluster_order = torch.argsort(
                expected_scores, dim=-1, descending=self.metric.HIGHER_IS_BETTER
            ).tolist()
            orig_indices = torch.arange(len(hypotheses_ir), device=hypotheses_ir.device)
            num_pruned_hypotheses = 0
            pruned_hypotheses_irs = []
            pruned_hypotheses_ids = []
            for k in best_cluster_order:
                k_mask = assigns.eq(k)
                hypotheses_k = hypotheses_ir[k_mask]
                pruned_hypotheses_irs.append(hypotheses_k)
                pruned_hypotheses_ids.append(orig_indices[k_mask])
                num_pruned_hypotheses += len(hypotheses_k)
                if num_pruned_hypotheses >= nbest:
                    break
            pruned_hypotheses_ir = torch.cat(pruned_hypotheses_irs)
            pruned_hypotheses_id = torch.cat(pruned_hypotheses_ids)
        with timer.measure("expectation/fine"):
            expected_scores = self.metric.pairwise_scores_from_ir(
                pruned_hypotheses_ir, centroids, source_ir
            ).mean(dim=-1)
        topk_scores, pruned_topk_indices = self.metric.topk(expected_scores, k=nbest)
        topk_indices = pruned_hypotheses_id[pruned_topk_indices].tolist()
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
