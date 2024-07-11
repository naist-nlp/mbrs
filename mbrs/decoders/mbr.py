from __future__ import annotations

from typing import Optional

from torch import Tensor

from . import DecoderReferenceBased, register


@register("mbr")
class DecoderMBR(DecoderReferenceBased):
    """Naive MBR decoder class.

    - Time complexity: O(N^2)
    - Space complexity: O(N^2)

    References:
        S. Kumar and W. Byrne, 2004,
        "Minimum Bayes-Risk Decoding for Statistical Machine Translation".
        https://aclanthology.org/N04-1022

        B. Eikema and W. Aziz, 2020,
        "Is MAP Decoding All You Need?
        The Inadequacy of the Mode in Neural Machine Translation".
        https://aclanthology.org/2020.coling-main.398
    """

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
        expected_scores = self.metric.expected_scores(
            hypotheses, references, source, reference_lprobs=reference_lprobs
        )
        topk_scores, topk_indices = self.metric.topk(expected_scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
