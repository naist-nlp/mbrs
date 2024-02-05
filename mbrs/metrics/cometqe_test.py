import torch

from .cometqe import MetricCOMETQE

SOURCE = "これはテストです"
HYPOTHESES = [
    "this is a test",
    "another test",
    "this is a fest",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SCORES = torch.Tensor([0.86415, 0.83704, 0.65335, 0.29771])


class TestMetricCOMETQE:
    def test_score(self, metric_cometqe: MetricCOMETQE):
        for i, hyp in enumerate(HYPOTHESES):
            assert torch.isclose(
                SCORES[i],
                torch.tensor(metric_cometqe.score(hyp, SOURCE)),
                atol=0.0005 / 100,
            )

    def test_scores(self, metric_cometqe: MetricCOMETQE):
        scores = metric_cometqe.scores(HYPOTHESES, SOURCE)
        torch.testing.assert_close(
            scores,
            SCORES.to(metric_cometqe.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )
