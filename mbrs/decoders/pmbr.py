from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from mbrs import timer
from mbrs.metrics import MetricCacheable
from mbrs.modules.als import MatrixFactorizationALS

from . import register
from .mbr import DecoderMBR


@register("pmbr")
class DecoderProbabilisticMBR(DecoderMBR):
    """Probablistic MBR decoder using alternating least squares (ALS) approximation.

    References:
        F. Trabelsi et al., 2024,
        "Efficient Minimum Bayes Risk Decoding using Low-Rank Matrix Completion Algorithms".
        https://arxiv.org/abs/2406.02832
    """

    cfg: Config

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - reduction_factor (float): Reduction factor.
          The computational budget will be reduced to `1 / reduction_factor`.
        - regularization_weight (float): Weight of L2 regularization.
        - rank (int): Rank of the factarized matrices.
        - niter (int): The number of alternating steps performed.
        - seed (int): Random seed.
        """

        reduction_factor: float = 8
        regularization_weight: float = 0.1
        rank: int = 8
        niter: int = 20
        seed: int = 0

    def expected_scores_probabilistic(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute the expected scores using the probabilistic MBR algorithm.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            torch.Tensor:
            - list[int]: Top-k indices.
        """
        rng = torch.Generator().manual_seed(self.cfg.seed)
        H = len(hypotheses)
        R = len(references)
        num_ucalcs = math.ceil(H * R / self.cfg.reduction_factor)

        pairwise_scores = torch.zeros((H, R), device=self.metric.device)
        pairwise_sample_indices = torch.randperm(H * R, generator=rng)[:num_ucalcs]
        observed_mask = pairwise_scores.bool()
        observed_mask = observed_mask.view(-1)
        observed_mask[pairwise_sample_indices] = True
        observed_mask = observed_mask.view(H, R)

        hypothesis_sample_indices: list[int] = (pairwise_sample_indices // R).tolist()
        reference_sample_indices: list[int] = (pairwise_sample_indices % R).tolist()
        hypothesis_samples = [hypotheses[i] for i in hypothesis_sample_indices]
        reference_samples = [references[j] for j in reference_sample_indices]

        # For COMET-22
        if isinstance(self.metric, MetricCacheable):
            hypotheses_ir = pairwise_scores.new_zeros((H, self.metric.embed_dim))
            references_ir = pairwise_scores.new_zeros((R, self.metric.embed_dim))

            hypothesis_sample_indices_set = set(hypothesis_sample_indices)
            reference_sample_indices_set = set(reference_sample_indices)
            hypothesis_samples_deduped = [
                hypotheses[i] for i in hypothesis_sample_indices_set
            ]
            reference_samples_deduped = [
                references[j] for j in reference_sample_indices_set
            ]
            with timer.measure("encode/hypotheses"):
                hypotheses_ir[list(hypothesis_sample_indices_set)] = self.metric.encode(
                    hypothesis_samples_deduped
                )
            with timer.measure("encode/references"):
                if hypotheses == references:
                    seen_indices = list(
                        hypothesis_sample_indices_set & reference_sample_indices_set
                    )
                    unseen_indices = list(
                        reference_sample_indices_set - hypothesis_sample_indices_set
                    )
                    if len(seen_indices) > 0:
                        references_ir[seen_indices] = hypotheses_ir[seen_indices]
                    if len(unseen_indices) > 0:
                        references_ir[unseen_indices] = self.metric.encode(
                            [references[j] for j in unseen_indices]
                        )
                else:
                    references_ir[list(reference_sample_indices_set)] = (
                        self.metric.encode(reference_samples_deduped)
                    )
            if source is None:
                source_ir = None
            else:
                with timer.measure("encode/source"):
                    source_ir = self.metric.encode([source])

        # Algorithm 2 in the paper.
        with timer.measure("PMBR"):
            if isinstance(self.metric, MetricCacheable):
                for i in range(0, len(hypothesis_sample_indices), H):
                    pairwise_scores[
                        hypothesis_sample_indices[i : i + H],
                        reference_sample_indices[i : i + H],
                    ] = self.metric.scores_from_ir(
                        hypotheses_ir[hypothesis_sample_indices[i : i + H]],
                        references_ir[reference_sample_indices[i : i + H]],
                        source_ir,
                    )
            else:
                pairwise_scores[hypothesis_sample_indices, reference_sample_indices] = (
                    self.metric.scores(hypothesis_samples, reference_samples, source)
                )

            # Algorithm 1 in the paper.
            als = MatrixFactorizationALS(
                regularization_weight=self.cfg.regularization_weight, rank=self.cfg.rank
            )
            X, Y = als.factorize(
                pairwise_scores,
                observed_mask=observed_mask,
                niter=self.cfg.niter,
                seed=self.cfg.seed,
            )
            pairwise_scores = X @ Y.T
            return pairwise_scores.mean(dim=-1)

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

        if self.cfg.reduction_factor <= 1.0:
            expected_scores = self.metric.expected_scores(
                hypotheses, references, source
            )
        else:  # Probabilistic MBR decoding
            expected_scores = self.expected_scores_probabilistic(
                hypotheses, references, source
            )
        topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )