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
SCORES_XCOMET = torch.Tensor(
    [
        [0.97671, 1.00000, 0.49054],
        [0.94399, 0.99120, 0.43007],
        [0.71786, 0.71210, 0.30775],
        [0.21788, 0.22079, 0.61004],
    ]
)
SCORES_XCOMETLITE = torch.Tensor(
    [
        [0.75441, 0.93626, 0.52062],
        [0.65309, 0.74183, 0.42666],
        [0.42499, 0.52537, 0.36234],
        [0.36428, 0.36428, 0.70672],
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
                    SCORES_XCOMET[i, j],
                    torch.tensor(metric_xcomet.score(hyp, ref, SOURCE)),
                    atol=0.0005 / 100,
                )

    def test_scores(self, metric_xcomet: MetricXCOMET):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)

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
            SCORES_XCOMET.mean(dim=1).to(metric_xcomet.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_xcomet: MetricXCOMET):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)
        assert torch.isclose(
            torch.tensor(metric_xcomet.corpus_score(hyps, refs, srcs)),
            torch.tensor(0.96848),
        )
        assert torch.isclose(
            torch.tensor(metric_xcomet.corpus_score(hyps, sources=srcs)),
            torch.tensor(0.99120),
        )
        assert torch.isclose(
            torch.tensor(metric_xcomet.corpus_score(hyps, references=refs)),
            torch.tensor(0.92473),
        )


class TestMetricXCOMETLite:
    @pytest.fixture(scope="class")
    def metric_xcometlite(self):
        return MetricXCOMET(MetricXCOMET.Config(model="myyycroft/XCOMET-lite"))

    def test_score(self, metric_xcometlite: MetricXCOMET):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES_XCOMETLITE[i, j],
                    torch.tensor(metric_xcometlite.score(hyp, ref, SOURCE)),
                    atol=0.0005 / 100,
                )

    def test_scores(self, metric_xcometlite: MetricXCOMET):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)

        torch.testing.assert_close(
            metric_xcometlite.scores(hyps, refs, srcs).cpu().float(),
            torch.FloatTensor([0.8399, 0.6973, 0.9058]),
            atol=0.0005 / 100,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            metric_xcometlite.scores(hyps, sources=srcs).cpu().float(),
            torch.FloatTensor([0.7146, 0.9707, 0.9416]),
            atol=0.0005 / 100,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            metric_xcometlite.scores(hyps, references=refs).cpu().float(),
            torch.FloatTensor([0.9112, 0.4744, 0.8681]),
            atol=0.0005 / 100,
            rtol=1e-4,
        )

    def test_expected_scores(self, metric_xcometlite: MetricXCOMET):
        expected_scores = metric_xcometlite.expected_scores(
            HYPOTHESES, REFERENCES, SOURCE
        )
        torch.testing.assert_close(
            expected_scores,
            SCORES_XCOMETLITE.mean(dim=1).to(metric_xcometlite.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_xcometlite: MetricXCOMET):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)
        assert torch.isclose(
            torch.tensor(metric_xcometlite.corpus_score(hyps, refs, srcs)),
            torch.tensor(0.81435),
        )
        assert torch.isclose(
            torch.tensor(metric_xcometlite.corpus_score(hyps, sources=srcs)),
            torch.tensor(0.87564),
        )
        assert torch.isclose(
            torch.tensor(metric_xcometlite.corpus_score(hyps, references=refs)),
            torch.tensor(0.75123),
        )
