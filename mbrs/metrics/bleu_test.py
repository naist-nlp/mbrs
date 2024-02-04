import pytest
import torch

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
    True: torch.Tensor(
        [
            [0.0, 100.0, 0.0],
            [0.0, 18.394, 0.0],
            [0.0, 59.460, 0.0],
            [0.0, 0.0, 8.493],
        ]
    ),
    False: torch.Tensor(
        [
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 59.460, 0.0],
            [0.0, 0.0, 8.493],
        ]
    ),
}


class TestMetricBLEU:
    @pytest.mark.parametrize("effective_order", [True, False])
    def test_score(self, effective_order: bool):
        metric = MetricBLEU(MetricBLEU.Config(effective_order=effective_order))
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES_EFFECTIVE_ORDER[effective_order][i, j],
                    torch.tensor(metric.score(hyp, ref)),
                    atol=0.0005,
                )

    @pytest.mark.parametrize("effective_order", [True, False])
    def test_expected_scores(self, effective_order: bool):
        metric = MetricBLEU(MetricBLEU.Config(effective_order=effective_order))
        expected_scores = metric.expected_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            expected_scores,
            SCORES_EFFECTIVE_ORDER[effective_order].mean(dim=1),
            atol=0.0005,
            rtol=1e-4,
        )
