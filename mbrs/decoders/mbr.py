from __future__ import annotations

import numpy as np

from mbrs.metrics import Metric

from . import Decoder, register


@register("mbr")
class DecoderMBR(Decoder):
    """MBR decoder class."""

    def __init__(self, cfg: DecoderMBR.Config, metric: Metric):
        self.cfg = cfg
        self.metric = metric

    def decode(
        self,
        hypotheses_set: list[list[str]],
        references_set: list[list[str]],
        *args,
        **kwargs,
    ) -> list[DecoderMBR.Output]:
        """Select the best hypothesis based on the strategy.

        Args:
            hypotheses_set (list[list[str]]): Set of hypotheses.
            references_set (list[list[str]]): Set of references.

        Returns:
            list[DecoderMBR.Output]: The best hypotheses.
        """
        outputs = []
        for i, (hyps, refs) in enumerate(zip(hypotheses_set, references_set)):
            scores = self.metric.pairwise_score(hyps, refs)
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
