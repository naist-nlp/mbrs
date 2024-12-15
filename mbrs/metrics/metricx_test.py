import pytest
import torch

from .metricx import MetricMetricX

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


class TestMetricMetricX24:
    PAIRWISE_SCORES = torch.Tensor(
        [
            [1.88180, 0.02979, 22.75496],
            [2.09417, 1.20656, 23.77537],
            [7.01185, 7.85715, 25.00000],
            [25.00000, 24.61126, 6.35110],
        ]
    )

    @pytest.fixture(scope="class")
    def metric_metricx(self):
        return MetricMetricX(
            MetricMetricX.Config("google/metricx-24-hybrid-large-v2p6")
        )

    def test_score(self, metric_metricx: MetricMetricX):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    self.PAIRWISE_SCORES[i, j],
                    torch.tensor(metric_metricx.score(hyp, ref, SOURCE)),
                    atol=0.0005 / 100,
                )

    def test_scores(self, metric_metricx: MetricMetricX):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)

        # Reference-based
        ref_scores = [0.23774, 5.31166, 0.10797]
        torch.testing.assert_close(
            metric_metricx.scores(hyps, refs, srcs).cpu().float(),
            torch.FloatTensor(ref_scores),
            atol=0.0005 / 100,
            rtol=1e-6,
        )
        # Reference-free
        qe_scores = [0.80446, 0.24616, 0.24536]
        torch.testing.assert_close(
            metric_metricx.scores(hyps, sources=srcs).cpu().float(),
            torch.FloatTensor(qe_scores),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_pairwise_scores(self, metric_metricx: MetricMetricX):
        pairwise_scores = metric_metricx.pairwise_scores(HYPOTHESES, REFERENCES, SOURCE)
        torch.testing.assert_close(
            pairwise_scores,
            self.PAIRWISE_SCORES.to(metric_metricx.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_pairwise_scores_empty_inputs(self, metric_metricx: MetricMetricX):
        pairwise_scores = metric_metricx.pairwise_scores(
            ["this is a test", ""], ["", "this is a test", ""], SOURCE
        )
        expected_pairwise_scores = [
            [3.74674, 0.02979, 3.74674],
            [5.85574, 4.30336, 5.85574],
        ]

        assert tuple(pairwise_scores.shape) == (2, 3)
        torch.testing.assert_close(
            pairwise_scores,
            torch.tensor(expected_pairwise_scores).to(metric_metricx.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_expected_scores(self, metric_metricx: MetricMetricX):
        expected_scores = metric_metricx.expected_scores(HYPOTHESES, REFERENCES, SOURCE)
        torch.testing.assert_close(
            expected_scores,
            self.PAIRWISE_SCORES.mean(dim=1).to(metric_metricx.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_metricx: MetricMetricX):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)

        # Reference-based
        ref_scores = [0.23774, 5.31166, 0.10797]
        assert torch.isclose(
            torch.tensor(metric_metricx.corpus_score(hyps, refs, srcs)),
            torch.tensor(ref_scores).mean(),
        )
        # Reference-free
        qe_scores = [0.80446, 0.24616, 0.24536]
        assert torch.isclose(
            torch.tensor(metric_metricx.corpus_score(hyps, sources=srcs)),
            torch.tensor(qe_scores).mean(),
        )


class TestMetricMetricX23:
    PAIRWISE_SCORES = torch.Tensor(
        [
            [0.93163, 0.24821, 5.53139],
            [1.53266, 0.99941, 7.54621],
            [4.34711, 5.35249, 9.13149],
            [5.28427, 6.05555, 1.33680],
        ]
    )

    @pytest.fixture(scope="class")
    def metric_metricx(self):
        return MetricMetricX(MetricMetricX.Config("google/metricx-23-large-v2p0"))

    def test_score(self, metric_metricx: MetricMetricX):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    self.PAIRWISE_SCORES[i, j],
                    torch.tensor(metric_metricx.score(hyp, ref, SOURCE)),
                    atol=0.0005 / 100,
                )

    def test_scores(self, metric_metricx: MetricMetricX):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)

        ref_scores = [0.21216, 5.95098, 0.33338]
        torch.testing.assert_close(
            metric_metricx.scores(hyps, refs, srcs).cpu().float(),
            torch.FloatTensor(ref_scores),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_pairwise_scores(self, metric_metricx: MetricMetricX):
        pairwise_scores = metric_metricx.pairwise_scores(HYPOTHESES, REFERENCES, SOURCE)
        torch.testing.assert_close(
            pairwise_scores,
            self.PAIRWISE_SCORES.to(metric_metricx.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_pairwise_scores_empty_inputs(self, metric_metricx: MetricMetricX):
        pairwise_scores = metric_metricx.pairwise_scores(
            ["this is a test", ""], ["", "this is a test", ""], SOURCE
        )
        expected_pairwise_scores = [
            [1.24063, 0.24821, 1.24063],
            [0.00000, 1.23078, 0.00000],
        ]

        assert tuple(pairwise_scores.shape) == (2, 3)
        torch.testing.assert_close(
            pairwise_scores,
            torch.tensor(expected_pairwise_scores).to(metric_metricx.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_expected_scores(self, metric_metricx: MetricMetricX):
        expected_scores = metric_metricx.expected_scores(HYPOTHESES, REFERENCES, SOURCE)
        torch.testing.assert_close(
            expected_scores,
            self.PAIRWISE_SCORES.mean(dim=1).to(metric_metricx.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_metricx: MetricMetricX):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        srcs = [SOURCE] * len(hyps)

        # Reference-based
        ref_scores = [0.21216, 5.95098, 0.33338]
        assert torch.isclose(
            torch.tensor(metric_metricx.corpus_score(hyps, refs, srcs)),
            torch.tensor(ref_scores).mean(),
        )


class TestMetricMetricX23QE:
    QE_SCORES = torch.Tensor([0.34737, 0.45215, 0.79827, 13.10156])

    @pytest.fixture(scope="class")
    def metric_metricx(self):
        return MetricMetricX(MetricMetricX.Config("google/metricx-23-qe-large-v2p0"))

    def test_score(self, metric_metricx: MetricMetricX):
        for i, hyp in enumerate(HYPOTHESES):
            assert torch.isclose(
                self.QE_SCORES[i],
                torch.tensor(metric_metricx.score(hyp, None, SOURCE)),
                atol=0.0005 / 100,
            )
            assert torch.isclose(
                self.QE_SCORES[i],
                torch.tensor(metric_metricx.score(hyp, "dummy", SOURCE)),
                atol=0.0005 / 100,
            )

    def test_scores(self, metric_metricx: MetricMetricX):
        hyps = ["another test", "this is a test", "this is an test"]
        srcs = [SOURCE] * len(hyps)

        # Reference-free
        qe_scores = [0.45215, 0.34737, 0.35742]
        torch.testing.assert_close(
            metric_metricx.scores(hyps, sources=srcs).cpu().float(),
            torch.FloatTensor(qe_scores),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_scores_empty_inputs(self, metric_metricx: MetricMetricX):
        scores = metric_metricx.scores(["this is a test", ""], None, [SOURCE] * 2)
        expected_scores = [0.34737, 0.72682]

        assert tuple(scores.shape) == (2,)
        torch.testing.assert_close(
            scores,
            torch.tensor(expected_scores).to(metric_metricx.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_metricx: MetricMetricX):
        hyps = ["another test", "this is a test", "this is an test"]
        srcs = [SOURCE] * len(hyps)

        # Reference-free
        qe_scores = [0.45215, 0.34737, 0.35742]
        assert torch.isclose(
            torch.tensor(metric_metricx.corpus_score(hyps, sources=srcs)),
            torch.tensor(qe_scores).mean(),
        )
