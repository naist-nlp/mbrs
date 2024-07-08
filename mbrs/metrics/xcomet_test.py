import pytest
import torch

from .xcomet import MetricXCOMET

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
        [0.97671, 1.00000, 0.49054],
        [0.94399, 0.99120, 0.43007],
        [0.71786, 0.71210, 0.30775],
        [0.21788, 0.22079, 0.61004],
    ]
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available on this machine."
)
class TestMetricXCOMET:
    def test_score(self, metric_xcomet: MetricXCOMET):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES[i, j],
                    torch.tensor(metric_xcomet.score(hyp, ref, SOURCE)),
                    atol=0.0005 / 100,
                )

    def test_scores(self, metric_xcomet: MetricXCOMET):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * 3

        torch.testing.assert_close(
            metric_xcomet.scores(hyps, refs, srcs).cpu().float(),
            torch.FloatTensor([1.00000, 0.90545, 1.00000]),
            atol=0.0005 / 100,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            metric_xcomet.scores(hyps, sources=srcs).cpu().float(),
            torch.FloatTensor([0.99120, 0.99120, 0.99120]),
            atol=0.0005 / 100,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            metric_xcomet.scores(hyps, references=refs).cpu().float(),
            torch.FloatTensor([1.00000, 0.77420, 1.00000]),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_expected_scores(self, metric_xcomet: MetricXCOMET):
        expected_scores = metric_xcomet.expected_scores(HYPOTHESES, REFERENCES, SOURCE)
        torch.testing.assert_close(
            expected_scores,
            SCORES.mean(dim=1).to(metric_xcomet.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )
