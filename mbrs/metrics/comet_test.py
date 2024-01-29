import numpy as np
import pytest

from .comet import MetricCOMET

SOURCE = "これはテストです"
HYPOTHESES = [
    "this is a test",
    "another test",
    "this is a fest",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
REFERENCES = [
    "ref",
    "this is a test",
    "producţia de zahăr brut se exprimă în zahăr alb;",
]
SCORES = np.array(
    [
        [0.54616, 0.99257, 0.40706],
        [0.59092, 0.81587, 0.38636],
        [0.50268, 0.75507, 0.38999],
        [0.40692, 0.37781, 0.78060],
    ],
    dtype=np.float32,
)


class TestMetricCOMET:
    @pytest.fixture(scope="class")
    def metric(self):
        return MetricCOMET(MetricCOMET.Config())

    def test_score(self, metric: MetricCOMET):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert np.isclose(
                    SCORES[i, j], metric.score(hyp, ref, SOURCE), atol=0.0005 / 100
                )

    def test_pairwise_score(self, metric: MetricCOMET):
        scores = metric.pairwise_score(HYPOTHESES, REFERENCES, SOURCE)
        assert np.allclose(scores, SCORES, atol=0.0005 / 100)
