from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer
from mbrs.metrics import MetricCacheable
from mbrs.modules.als import MatrixFactorizationALS

from . import register
from .mbr import DecoderMBR


@register("probabilistic_mbr")
class DecoderProbabilisticMBR(DecoderMBR):
    """Probabilistic MBR decoder using alternating least squares (ALS) approximation.

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

        reduction_factor: float = 8.0
        regularization_weight: float = 0.1
        rank: int = 8
        niter: int = 10
        seed: int = 0

    def pairwise_scores_probabilistic(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
    ) -> Tensor:
        """Compute approximated pairwise scores using the probabilistic MBR algorithm.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            Tensor: Approximated pairwise scores of shape `(H, R)`.
        """
        rng = torch.Generator().manual_seed(self.cfg.seed)
        H = len(hypotheses)
        R = len(references)
        num_ucalcs = math.ceil(H * R / self.cfg.reduction_factor)

        pairwise_scores = torch.zeros((H, R), device=self.metric.device)
        pairwise_sample_indices = torch.randperm(H * R, generator=rng)[:num_ucalcs]
        hypothesis_sample_indices: list[int] = (pairwise_sample_indices // R).tolist()
        reference_sample_indices: list[int] = (pairwise_sample_indices % R).tolist()

        # Algorithm 2 in the paper.
        if isinstance(self.metric, MetricCacheable):
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

            num_hyp_samples = len(hypothesis_sample_indices)
            for i in range(0, num_hyp_samples, H):
                pairwise_scores[
                    hypothesis_sample_indices[i : i + H],
                    reference_sample_indices[i : i + H],
                ] = self.metric.scores_from_ir(
                    hypotheses_ir[hypothesis_sample_indices[i : i + H]],
                    references_ir[reference_sample_indices[i : i + H]],
                    source_ir.repeat(min(H, num_hyp_samples - i))
                    if source_ir is not None
                    else None,
                ).float()
        else:
            hypothesis_samples = [hypotheses[i] for i in hypothesis_sample_indices]
            reference_samples = [references[j] for j in reference_sample_indices]
            pairwise_scores[hypothesis_sample_indices, reference_sample_indices] = (
                self.metric.scores(
                    hypothesis_samples,
                    reference_samples,
                    [source] * len(hypothesis_samples) if source is not None else None,
                ).float()
            )
        observed_mask = pairwise_scores.new_zeros((H, R), dtype=torch.bool)
        observed_mask[hypothesis_sample_indices, reference_sample_indices] = True

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
        reconstructed_pairwise_scores = X @ Y.T
        return reconstructed_pairwise_scores

    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderMBR.Output: The n-best hypotheses.
        """

        if self.cfg.reduction_factor <= 1.0:
            expected_scores = self.metric.expected_scores(
                hypotheses, references, source, reference_lprobs=reference_lprobs
            )
        else:  # Probabilistic MBR decoding
            pairwise_scores = self.pairwise_scores_probabilistic(
                hypotheses, references, source
            )
            expected_scores = functional.expectation(
                pairwise_scores, lprobs=reference_lprobs
            )

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
