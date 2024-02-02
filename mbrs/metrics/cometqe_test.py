import numpy as np
import pytest

from .cometqe import MetricCOMETQE

SOURCE = "これはテストです"
HYPOTHESES = [
    "this is a test",
    "another test",
    "this is a fest",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SCORES = np.array([0.86415, 0.83704, 0.65335, 0.29771], dtype=np.float32)


class TestMetricCOMET:
    def test_score(self, metric_comet_qe: MetricCOMETQE):
        for i, hyp in enumerate(HYPOTHESES):
            assert np.isclose(
                SCORES[i],
                metric_comet_qe.score(hyp, SOURCE),
                atol=0.0005 / 100,
            )

    def test_scores(self, metric_comet_qe: MetricCOMETQE):
        scores = metric_comet_qe.scores(HYPOTHESES, SOURCE)
        assert np.allclose(scores, SCORES, atol=0.0005 / 100)
