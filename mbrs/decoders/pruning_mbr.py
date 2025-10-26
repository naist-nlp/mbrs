from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from mbrs import functional, timer
from mbrs.metrics import Metric, MetricCacheable
from mbrs.selectors import SELECTOR_NBEST, Selector, SelectorNbest

from . import register
from .mbr import DecoderMBR


@register("pruning_mbr")
class DecoderPruningMBR(DecoderMBR):
    """Pruning MBR decoder class.

    References:
        J. Cheng and A. Vlachos, 2023,
        "Faster Minimum Bayes Risk Decoding with Confidence-based Pruning".
        https://aclanthology.org/2023.emnlp-main.767/
    """

    def __init__(
        self,
        cfg: DecoderPruningMBR.Config,
        metric: Metric,
        selector: Selector = SELECTOR_NBEST,
    ) -> None:
        if not isinstance(selector, SelectorNbest):
            raise ValueError(
                "Confidence-based pruning cannot be combined with other selectors than the nbest."
            )
        super().__init__(cfg, metric, selector)

    @dataclass
    class Config(DecoderMBR.Config):
        """Configuration for the decoder.

        - alpha (float): Prune hypotheses based on this confidence threshold.
        - sampling_shceduler (list[int]): Sample size scheduler. For each step, the
          number of samples will be the t-th number.
        - num_boostrap_samples (int): Number of boostrap samples.
        - seed (int): Random seed for bootstrap sampling.
        """

        alpha: float = 0.99
        sampling_scheduler: list[int] = field(
            default_factory=lambda: [8, 16, 32, 64, 128, 256]
        )
        num_bootstrap_samples: int = 500
        seed: int = 0

    cfg: Config

    def decode_pruning(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> tuple[list[float], list[int]]:
        """Select the n-best hypotheses using pruning MBR decoding.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            - list[float]: Top-k scores.
            - list[int]: Top-k indices.
        """
        rng = torch.Generator(device=self.metric.device).manual_seed(self.cfg.seed)
        H = len(hypotheses)
        max_r = min(max(self.cfg.sampling_scheduler), len(references))
        pairwise_scores = torch.zeros((H, max_r), device=self.metric.device)
        orig_indices = torch.arange(H, device=self.metric.device)

        if isinstance(self.metric, MetricCacheable):
            with timer.measure("encode/hypotheses"):
                hypotheses_ir = self.metric.encode(hypotheses)
            references_ir = hypotheses_ir if hypotheses == references else None
            if source is None:
                source_ir = None
            else:
                with timer.measure("encode/source"):
                    source_ir = self.metric.encode([source])

        with timer.measure("pruning_mbr"):
            # Algorithm 1 in the paper.
            prev_r = 0
            for t, r in enumerate(self.cfg.sampling_scheduler):
                r = min(r, len(references))
                if r <= prev_r:
                    break

                # Equation 5 and Algorithm 2 in the paper.
                if isinstance(self.metric, MetricCacheable):
                    if references_ir is None:
                        with timer.measure("encode/references"):
                            references_ir_t = self.metric.encode(references[prev_r:r])
                    else:
                        references_ir_t = references_ir[prev_r:r]

                    pairwise_scores[:, prev_r:r] = self.metric.pairwise_scores_from_ir(
                        hypotheses_ir, references_ir_t, source_ir
                    )
                else:
                    pairwise_scores[:, prev_r:r] = self.metric.pairwise_scores(
                        hypotheses, references[prev_r:r], source
                    )

                expected_scores = functional.expectation(
                    pairwise_scores[:, :r],
                    lprobs=reference_lprobs[:r]
                    if reference_lprobs is not None
                    else None,
                )
                current_best_idx = self.argbest(expected_scores)
                sample_indices = torch.randint(
                    r,
                    size=(self.cfg.num_bootstrap_samples, r),
                    device=self.metric.device,
                    generator=rng,
                )
                bootstrap_expected_scores = functional.expectation(
                    pairwise_scores[:, sample_indices],
                    lprobs=reference_lprobs[sample_indices]
                    if reference_lprobs is not None
                    else None,
                )
                num_wins = (
                    (
                        bootstrap_expected_scores
                        >= bootstrap_expected_scores[current_best_idx]
                    )
                    if self.maximize
                    else (
                        bootstrap_expected_scores
                        <= bootstrap_expected_scores[current_best_idx]
                    )
                )
                win_rates = num_wins.float().mean(dim=1)
                winners = (win_rates > 1 - self.cfg.alpha).nonzero(as_tuple=True)[0]
                num_winners = len(winners)
                if num_winners >= nbest:
                    if isinstance(self.metric, MetricCacheable):
                        hypotheses_ir = hypotheses_ir[winners]
                    else:
                        hypotheses = [hypotheses[i] for i in winners]
                    pairwise_scores = pairwise_scores[winners]
                    orig_indices = orig_indices[winners]
                    prev_r = r
                else:
                    break
            expected_scores = functional.expectation(
                pairwise_scores[:, :prev_r],
                lprobs=reference_lprobs[:prev_r]
                if reference_lprobs is not None
                else None,
            )

        topk_scores, topk_indices = self.topk(expected_scores, k=nbest)
        return topk_scores, orig_indices[topk_indices].tolist()

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

        topk_scores, topk_indices = self.decode_pruning(
            hypotheses,
            references,
            source,
            nbest=nbest,
            reference_lprobs=reference_lprobs,
        )
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
