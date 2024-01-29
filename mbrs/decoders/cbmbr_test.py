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
    @pytest.fixture(scope="class")
    def metric(self):
        return MetricCOMET(MetricCOMET.Config())

    @pytest.mark.parametrize("kmeanspp", [True, False])
    def test_decode(self, metric: MetricCOMET, kmeanspp: bool):
        decoder = DecoderCBMBR(
            DecoderCBMBR.Config(ncentroids=NCENTROIDS, kmeanspp=kmeanspp), metric
        )
        outputs = decoder.decode(HYPOTHESES, REFERENCES, SOURCE)
        assert [o.idx for o in outputs] == BEST_INDICES
        assert [o.sentence for o in outputs] == BEST_SENTENCES
        assert np.allclose(
            np.array([o.score for o in outputs], dtype=np.float32),
            SCORES,
            atol=0.0005 / 100,
        )
