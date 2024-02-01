from __future__ import annotations

from . import DecoderReferenceless, register


@register("rerank")
class DecoderRerank(DecoderReferenceless):
    """Reranking decoder class.

    - Time complexity: O(N)
    - Space complexity: O(N)
    """

    def decode(
        self, hypotheses: list[str], source: str, nbest: int = 1
    ) -> DecoderRerank.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            source (str): A source.
            nbest (int): Return the n-best hypotheses.

        Returns:
            DecoderRerank.Output: The n-best hypotheses.
        """
        scores = self.metric.scores(hypotheses, source)
        topk_scores, topk_indices = self.metric.topk(scores, k=nbest)
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
        )
