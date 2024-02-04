import torch

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
SCORES = torch.Tensor(
    [
        [7.246, 100.0, 3.411],
        [14.493, 23.060, 4.386],
        [14.493, 60.628, 3.411],
        [6.849, 7.224, 46.161],
    ],
)


class TestMetricChrF:
    def test_score(self):
        metric = MetricChrF(MetricChrF.Config())
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES[i, j], torch.tensor(metric.score(hyp, ref)), atol=0.0005
                )

    def test_expected_scores(self):
        metric = MetricChrF(MetricChrF.Config())
        expected_scores = metric.expected_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            expected_scores,
            SCORES.mean(dim=1),
            atol=0.0005,
            rtol=1e-4,
        )
