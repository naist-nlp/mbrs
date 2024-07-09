import torch

from .bleurt import MetricBLEURT

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
        [0.10943, 0.97992, 0.08830],
        [0.06851, 0.59438, 0.03788],
        [0.12863, 0.43761, 0.08151],
        [0.16558, 0.17050, 0.65812],
    ]
)


class TestMetricBLEURT:
    def test_score(self, metric_bleurt: MetricBLEURT):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES[i, j],
                    torch.tensor(metric_bleurt.score(hyp, ref)),
                    atol=0.0005 / 100,
                )

    def test_scores(self, metric_bleurt: MetricBLEURT):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        torch.testing.assert_close(
            metric_bleurt.scores(hyps, refs).cpu().float(),
            torch.FloatTensor([0.98427, 0.46749, 0.88140]),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_pairwise_scores(self, metric_bleurt: MetricBLEURT):
        pairwise_scores = metric_bleurt.pairwise_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            pairwise_scores,
            SCORES.to(metric_bleurt.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_expected_scores(self, metric_bleurt: MetricBLEURT):
        expected_scores = metric_bleurt.expected_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            expected_scores,
            SCORES.mean(dim=1).to(metric_bleurt.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )
