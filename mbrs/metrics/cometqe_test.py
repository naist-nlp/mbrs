import torch

from .cometqe import MetricCOMETQE

HYPOTHESES = [
    "this is a test",
    "another test",
    "this is a fest",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SOURCES = ["これはテストです"] * len(HYPOTHESES)

SCORES = torch.Tensor([0.86415, 0.83704, 0.65335, 0.29771])


class TestMetricCOMETQE:
    def test_score(self, metric_cometqe: MetricCOMETQE):
        for i, (hyp, src) in enumerate(zip(HYPOTHESES, SOURCES)):
            assert torch.isclose(
                SCORES[i],
                torch.tensor(metric_cometqe.score(hyp, src)),
                atol=0.0005 / 100,
            )

    def test_scores(self, metric_cometqe: MetricCOMETQE):
        scores = metric_cometqe.scores(HYPOTHESES, SOURCES)
        torch.testing.assert_close(
            scores,
            SCORES.to(metric_cometqe.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_cometqe: MetricCOMETQE):
        hyps = [
            "this is a test",
            "another test",
            "this is a fest",
            "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
        ]
        assert torch.isclose(
            torch.tensor(metric_cometqe.corpus_score(hyps, SOURCES)),
            torch.tensor(0.66306),
        )
