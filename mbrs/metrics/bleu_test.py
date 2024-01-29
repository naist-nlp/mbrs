import numpy as np
import pytest

from .bleu import MetricBLEU

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
SCORES_EFFECTIVE_ORDER = {
    True: np.array(
        [
            [0.0, 100.0, 0.0],
            [0.0, 18.394, 0.0],
            [0.0, 59.460, 0.0],
            [0.0, 0.0, 8.493],
        ],
        dtype=np.float32,
    ),
    False: np.array(
        [
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 59.460, 0.0],
            [0.0, 0.0, 8.493],
        ],
        dtype=np.float32,
    ),
}


class TestMetricBLEU:
    @pytest.mark.parametrize("effective_order", [True, False])
    def test_score(self, effective_order: bool):
        metric = MetricBLEU(MetricBLEU.Config(effective_order=effective_order))
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert np.isclose(
                    SCORES_EFFECTIVE_ORDER[effective_order][i, j],
                    metric.score(hyp, ref),
                    atol=0.0005
                )

    @pytest.mark.parametrize("effective_order", [True, False])
    def test_pairwise_score(self, effective_order: bool):
        metric = MetricBLEU(MetricBLEU.Config(effective_order=effective_order))
        scores = metric.pairwise_score(HYPOTHESES, REFERENCES)
        assert np.allclose(scores, SCORES_EFFECTIVE_ORDER[effective_order], atol=0.0005)
