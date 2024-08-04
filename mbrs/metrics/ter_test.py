import multiprocessing
import sys

import torch

from .ter import MetricTER

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
        [400.0, 0.0, 100.0],
        [200.0, 75.0, 100.0],
        [400.0, 25.0, 100.0],
        [1300.0, 325.0, 100.0],
    ],
)


class TestMetricTER:
    def test_score(self):
        metric = MetricTER(MetricTER.Config())
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES[i, j], torch.tensor(metric.score(hyp, ref)), atol=0.0005
                )

    def test_expected_scores(self):
        metric = MetricTER(MetricTER.Config())
        expected_scores = metric.expected_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            expected_scores,
            SCORES.mean(dim=1),
            atol=0.0005,
            rtol=1e-4,
        )

        if sys.platform == "linux":
            default_method = multiprocessing.get_start_method()
            multiprocessing.set_start_method("spawn", force=True)
            expected_scores = metric.expected_scores(HYPOTHESES, REFERENCES)
            torch.testing.assert_close(
                expected_scores, SCORES.mean(dim=1), atol=0.0005, rtol=1e-4
            )
            multiprocessing.set_start_method(default_method, force=True)

    def test_scores(self):
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

        metric = MetricTER(MetricTER.Config())
        torch.testing.assert_close(
            metric.scores(hyps, refs),
            torch.tensor([metric.score(h, r) for h, r in zip(hyps, refs)]),
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

        metric = MetricTER(MetricTER.Config())
        assert torch.isclose(
            torch.tensor(metric.corpus_score(hyps, refs)), torch.tensor(66.66667)
        )
