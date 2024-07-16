import torch

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
SCORES = torch.Tensor(
    [
        [0.54616, 0.99257, 0.40706],
        [0.59092, 0.81587, 0.38636],
        [0.50268, 0.75507, 0.38999],
        [0.40692, 0.37781, 0.78060],
    ]
)
EXPECTED_SCORES_AGGREGATED = torch.Tensor([0.85933, 0.75443, 0.68586, 0.62650])


class TestMetricCOMET:
    def test_score(self, metric_comet: MetricCOMET):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES[i, j],
                    torch.tensor(metric_comet.score(hyp, ref, SOURCE)),
                    atol=0.0005 / 100,
                )

    def test_expected_scores(self, metric_comet: MetricCOMET):
        expected_scores = metric_comet.expected_scores(HYPOTHESES, REFERENCES, SOURCE)
        torch.testing.assert_close(
            expected_scores,
            SCORES.mean(dim=1).to(metric_comet.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_comet: MetricCOMET):
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
        assert torch.isclose(
            torch.tensor(metric_comet.corpus_score(hyps, refs, [SOURCE] * len(hyps))),
            torch.tensor(0.77979),
        )

    def test_expected_scores_reference_aggregation(self, metric_comet: MetricCOMET):
        expected_scores = metric_comet.expected_scores_reference_aggregation(
            HYPOTHESES, REFERENCES, SOURCE
        )
        torch.testing.assert_close(
            expected_scores,
            EXPECTED_SCORES_AGGREGATED.to(metric_comet.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )
