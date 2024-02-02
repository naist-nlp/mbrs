import numpy as np
import pytest

from mbrs.metrics.cometqe import MetricCOMETQE

from .rerank import DecoderRerank

SOURCE = "これはテストです"
HYPOTHESES = [
    ["another test", "this is a test", "this is a fest"],
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
    ["Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"],
]

BEST_INDICES = [1, 2, 0, 0]
BEST_SENTENCES = [
    "this is a test",
    "this is a test",
    "this is a test",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SCORES = np.array([0.86415, 0.86415, 0.86415, 0.29771], dtype=np.float32)


class TestDecoderRerank:
    def test_decode(self, metric_comet_qe: MetricCOMETQE):
        decoder = DecoderRerank(DecoderRerank.Config(), metric_comet_qe)
        for i, hyps in enumerate(HYPOTHESES):
            output = decoder.decode(hyps, SOURCE, nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert np.allclose(
                np.array(output.score[0], dtype=np.float32),
                SCORES[i],
                atol=0.0005 / 100,
            )
