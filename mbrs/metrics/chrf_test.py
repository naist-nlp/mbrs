import numpy as np

from .chrf import MetricChrF

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
        [7.246, 100.0, 3.411],
        [14.493, 23.060, 4.386],
        [14.493, 60.628, 3.411],
        [6.849, 7.224, 46.161],
    ],
    dtype=np.float32,
)


class TestMetricChrF:
    def test_score(self):
        metric = MetricChrF(MetricChrF.Config())
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert np.isclose(SCORES[i, j], metric.score(hyp, ref), atol=0.0005)

    def test_pairwise_score(self):
        metric = MetricChrF(MetricChrF.Config())
        scores = metric.pairwise_score(HYPOTHESES, REFERENCES)
        assert np.allclose(scores, SCORES, atol=0.0005)
