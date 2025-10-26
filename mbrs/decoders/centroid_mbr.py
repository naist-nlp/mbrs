from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer
from mbrs.metrics import MetricAggregatableCache
from mbrs.modules.kmeans import Kmeans
from mbrs.selectors import SELECTOR_NBEST, Selector

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
        https://aclanthology.org/2024.findings-acl.654
    """

    def __init__(
        self,
        cfg: DecoderCentroidMBR.Config,
        metric: MetricAggregatableCache,
        selector: Selector = SELECTOR_NBEST,
    ) -> None:
        super().__init__(cfg, metric, selector=selector)
        self.kmeans = Kmeans(cfg.kmeans)

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - kmeans (Kmeans.Config): Configuration for k-means.
        - count_weight: (bool) Weight the scores with counts.
        """

        kmeans: Kmeans.Config = field(default_factory=Kmeans.Config)
        count_weight: bool = False

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
            DecoderCentroidMBR.Output: The n-best hypotheses.
        """
        assert isinstance(self.metric, MetricAggregatableCache)

        with timer.measure("encode/hypotheses"):
            hypotheses_ir = self.metric.encode(hypotheses)
        if hypotheses == references:
            references_ir: MetricAggregatableCache.Cache = hypotheses_ir
        else:
            with timer.measure("encode/references"):
                references_ir: MetricAggregatableCache.Cache = self.metric.encode(
                    references
                )
        if source is None:
            source_ir = None
        else:
            with timer.measure("encode/source"):
                source_ir = self.metric.encode([source])
        centroids, assigns = references_ir.cluster(self.kmeans)

        lprobs = None
        if self.cfg.count_weight:
            centroid_ids, counts_nonzero = assigns.unique(return_counts=True)
            counts = centroid_ids.new_zeros(len(centroids))
            counts[centroid_ids] = counts_nonzero
            lprobs = counts.log()
        elif reference_lprobs is not None:
            # Accumurate the log-probabilities for each centroid by logsumexp.
            lprobs = (
                torch.zeros(len(centroids), dtype=torch.float32, device=assigns.device)
                .scatter_add(
                    dim=-1,
                    index=assigns.unique(),
                    src=reference_lprobs.to(assigns.device).softmax(
                        dim=-1, dtype=torch.float32
                    ),
                )
                .log()
            )

        with timer.measure("expectation"):
            pairwise_scores = self.metric.pairwise_scores_from_ir(
                hypotheses_ir, centroids, source_ir
            )
            if lprobs is not None:
                lprobs = lprobs.to(pairwise_scores)
            expected_scores = functional.expectation(pairwise_scores, lprobs=lprobs)

        selector_outputs = self.select(
            hypotheses, expected_scores, nbest=nbest, source=source
        )
        return (
            self.Output(
                idx=selector_outputs.idx,
                sentence=selector_outputs.sentence,
                score=selector_outputs.score,
            )
            | selector_outputs
        )
