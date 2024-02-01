import numpy as np
import pytest

from mbrs.metrics.comet import MetricCOMET

from .cbmbr import DecoderCBMBR

SOURCE = [
    "これはテストです",
    "これはテストです",
    "これはテストです",
    "これはテストです",
]
HYPOTHESES = [
    ["another test", "this is a test", "this is a fest"],
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
    ["Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"],
]
REFERENCES = [
    ["another test", "this is a test", "this is a fest"],
    ["this is a test", "ref", "these are tests", "this is the test"],
    ["this is a test"],
    ["producţia de zahăr brut se exprimă în zahăr alb;"],
]

BEST_INDICES = [1, 2, 0, 0]
BEST_SENTENCES = [
    "this is a test",
    "this is a test",
    "this is a test",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SCORES = np.array([0.88974, 0.76127, 0.99257, 0.78060], dtype=np.float32)

NCENTROIDS = 2


class TestDecoderCBMBR:
    @pytest.mark.parametrize("kmeanspp", [True, False])
    def test_decode(self, metric_comet: MetricCOMET, kmeanspp: bool):
        decoder = DecoderCBMBR(
            DecoderCBMBR.Config(ncentroids=NCENTROIDS, kmeanspp=kmeanspp), metric_comet
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert np.allclose(
                np.array(output.score[0], dtype=np.float32),
                SCORES[i],
                atol=0.0005 / 100,
            )
