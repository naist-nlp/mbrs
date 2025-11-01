from __future__ import annotations

from typing import Optional

from torch import Tensor

from mbrs.selectors import Selector, register


@register("nbest")
class SelectorNbest(Selector):
    def select(
        self,
        hypotheses: list[str],
        expected_scores: Tensor,
        nbest: int = 1,
        source: Optional[str] = None,
        maximize: bool = True,
        **kwargs,
    ) -> SelectorNbest.Output:
        """Select the final output list.

        Args:
            hypotheses (list[str]): Hypotheses.
            expected_scores (Tensor): The expected scores for each hypothesis.
            nbest (int): Return the n-best hypotheses based on the selection rule.
            source (str, optional): A source.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            Selector.Output: Selected hypotheses.
        """
        nbest = min(len(hypotheses), nbest)
        topk_scores, topk_indices = self.topk(
            expected_scores, k=nbest, maximize=maximize
        )
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
