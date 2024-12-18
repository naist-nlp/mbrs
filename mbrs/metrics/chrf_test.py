import multiprocessing
import sys

import pytest
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
EXPECTED_SCORES_AGGREGATED = torch.Tensor([23.08157, 10.29285, 16.35524, 29.16298])


class TestMetricChrF:
    @pytest.mark.parametrize("fastchrf", [False, True])
    def test_config(self, fastchrf: bool):
        if fastchrf:
            with pytest.raises(ValueError):
                MetricChrF.Config(word_order=2, fastchrf=fastchrf)
        else:
            MetricChrF(MetricChrF.Config(word_order=2, fastchrf=fastchrf))

    @pytest.mark.parametrize("fastchrf", [False, True])
    def test_score(self, fastchrf: bool):
        metric = MetricChrF(MetricChrF.Config(fastchrf=fastchrf))
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES[i, j], torch.tensor(metric.score(hyp, ref)), atol=0.0005
                )

    @pytest.mark.parametrize("fastchrf", [False, True])
    def test_expected_scores(self, fastchrf: bool):
        metric = MetricChrF(MetricChrF.Config(fastchrf=fastchrf))
        expected_scores = metric.expected_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            expected_scores,
            SCORES.mean(dim=1),
            atol=0.0005,
            rtol=1e-4,
        )
        if not fastchrf and sys.platform == "linux":
            default_method = multiprocessing.get_start_method()
            multiprocessing.set_start_method("spawn", force=True)
            expected_scores = metric.expected_scores(HYPOTHESES, REFERENCES)
            torch.testing.assert_close(
                expected_scores, SCORES.mean(dim=1), atol=0.0005, rtol=1e-4
            )
            multiprocessing.set_start_method(default_method, force=True)

    @pytest.mark.parametrize("fastchrf", [False, True])
    def test_scores(self, fastchrf: bool):
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

        metric = MetricChrF(MetricChrF.Config())
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

        metric = MetricChrF(MetricChrF.Config())
        assert torch.isclose(
            torch.tensor(metric.corpus_score(hyps, refs)), torch.tensor(53.90979)
        )

    @pytest.mark.parametrize("fastchrf", [False, True])
    def test_expected_scores_reference_aggregation(self, fastchrf: bool):
        metric = MetricChrF(MetricChrF.Config(fastchrf=fastchrf))
        expected_scores = metric.expected_scores_reference_aggregation(
            HYPOTHESES, REFERENCES
        )
        torch.testing.assert_close(
            expected_scores, EXPECTED_SCORES_AGGREGATED, atol=0.0005, rtol=1e-4
        )

        if fastchrf:
            with pytest.raises(ValueError):
                metric.expected_scores_reference_aggregation(
                    HYPOTHESES,
                    REFERENCES,
                    reference_lprobs=torch.Tensor([-2.000]).repeat(len(REFERENCES)),
                )
        else:
            expected_scores = metric.expected_scores_reference_aggregation(
                HYPOTHESES,
                REFERENCES,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(REFERENCES)),
            )
            torch.testing.assert_close(
                expected_scores, EXPECTED_SCORES_AGGREGATED, atol=0.0005, rtol=1e-4
            )

    def test_expected_scores_reference_aggregation_empty_inputs(self):
        metric = MetricChrF(MetricChrF.Config())
        hyps = ["thank you", ""]
        refs = ["thank you so much", "", "thank you.", "thank you", ""]
        expected_scores = metric.expected_scores_reference_aggregation(hyps, refs)
        torch.testing.assert_close(
            expected_scores, torch.Tensor([68.2679, 0.0]), atol=0.0005, rtol=1e-4
        )

        expected_scores = metric.expected_scores_reference_aggregation(
            hyps, refs, reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs))
        )
        torch.testing.assert_close(
            expected_scores, torch.Tensor([68.2679, 0.0]), atol=0.0005, rtol=1e-4
        )
