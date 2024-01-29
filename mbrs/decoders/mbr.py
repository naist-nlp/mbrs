from __future__ import annotations

from typing import Optional

import numpy as np

from . import Decoder, register


@register("mbr")
class DecoderMBR(Decoder):
    """MBR decoder class."""

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
        outputs = []
        for i, (hyps, refs) in enumerate(zip(hypotheses_set, references_set)):
            source = source_set[i] if source_set is not None else None
            scores = self.metric.pairwise_score(hyps, refs, source)
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
