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
EXPECTED_SCORES_AGGREGATED = {
    True: torch.Tensor([25.96003, 6.44121, 20.31482, 5.13217]),
    False: torch.Tensor([25.96003, 0.0, 20.31482, 5.13217]),
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

    def test_corpus_score(self):
        hyps = [
            "this is a test",
            "another test",
            "this is a fest",
            "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
        ]
        refs = [
            "this is a test",
            "ref",
            "this is a test",
            "producţia de zahăr brut se exprimă în zahăr alb;",
        ]

        metric = MetricBLEU(MetricBLEU.Config())
        assert torch.isclose(
            torch.tensor(metric.corpus_score(hyps, refs)), torch.tensor(22.41424)
        )

    @pytest.mark.parametrize("effective_order", [True, False])
    def test_expected_scores_reference_aggregation(self, effective_order: bool):
        metric = MetricBLEU(MetricBLEU.Config(effective_order=effective_order))
        expected_scores = metric.expected_scores_reference_aggregation(
            HYPOTHESES, REFERENCES
        )
        torch.testing.assert_close(
            expected_scores,
            EXPECTED_SCORES_AGGREGATED[effective_order],
            atol=0.0005,
            rtol=1e-4,
        )

        expected_scores = metric.expected_scores_reference_aggregation(
            HYPOTHESES,
            REFERENCES,
            reference_lprobs=torch.Tensor([-2.000]).repeat(len(REFERENCES)),
        )
        torch.testing.assert_close(
            expected_scores,
            EXPECTED_SCORES_AGGREGATED[effective_order],
            atol=0.0005,
            rtol=1e-4,
        )
