from __future__ import annotations

from mbrs import timer

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
        with timer.measure("rerank"):
            scores = self.metric.scores(hypotheses, sources=[source] * len(hypotheses))

        selector_outputs = self.select(hypotheses, scores, nbest=nbest, source=source)
        return (
            self.Output(
                idx=selector_outputs.idx,
                sentence=selector_outputs.sentence,
                score=selector_outputs.score,
            )
            | selector_outputs
        )
