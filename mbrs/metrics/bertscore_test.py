import pytest
import torch

from .bertscore import BERTScoreScoreType, MetricBERTScore

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
        [0.8507761359214783, 1.0000001192092896, 0.7784053683280945],
        [0.89408278465271, 0.9161176681518555, 0.774659276008606],
        [0.8564733862876892, 0.945708692073822, 0.7765841484069824],
        [0.752592146396637, 0.7765471935272217, 0.926070511341095],
    ]
)
PRECISION_SCORES = torch.Tensor(
    [
        [0.8217871189117432, 1.0000001192092896, 0.8424798250198364],
        [0.8671472072601318, 0.9334712028503418, 0.8422600030899048],
        [0.8266336917877197, 0.945708692073822, 0.8379464149475098],
        [0.69720858335495, 0.715325653553009, 0.90543133020401],
    ]
)
RECALL_SCORES = torch.Tensor(
    [
        [0.8818850517272949, 1.0000001192092896, 0.7233883142471313],
        [0.922745406627655, 0.899397611618042, 0.7171036601066589],
        [0.8885480165481567, 0.945708692073822, 0.7235957980155945],
        [0.8175339698791504, 0.8492288589477539, 0.9476726651191711],
    ]
)


@pytest.mark.metrics_bertscore
class TestMetricBERTScore:
    @pytest.fixture(scope="class")
    def metric_bertscore(self):
        return MetricBERTScore(MetricBERTScore.Config(lang="en", batch_size=8))

    def test_score(self, metric_bertscore: MetricBERTScore):
        for i, hyp in enumerate(HYPOTHESES):
            for j, ref in enumerate(REFERENCES):
                assert torch.isclose(
                    SCORES[i, j],
                    torch.tensor(metric_bertscore.score(hyp, ref)),
                    atol=0.0005 / 100,
                )

    def test_scores(self, metric_bertscore: MetricBERTScore):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        torch.testing.assert_close(
            metric_bertscore.scores(hyps, refs).cpu().float(),
            torch.FloatTensor(
                [0.9999998807907104, 0.9457088112831116, 0.949202835559845]
            ),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_pairwise_scores(self, metric_bertscore: MetricBERTScore):
        pairwise_scores = metric_bertscore.pairwise_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            pairwise_scores,
            SCORES.to(metric_bertscore.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_pairwise_scores_empty_inputs(self, metric_bertscore: MetricBERTScore):
        pairwise_scores = metric_bertscore.pairwise_scores(
            ["this is a test", ""], ["", "this is a test", ""]
        )
        assert tuple(pairwise_scores.shape) == (2, 3)
        torch.testing.assert_close(
            pairwise_scores,
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).to(
                metric_bertscore.device
            ),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_expected_scores(self, metric_bertscore: MetricBERTScore):
        expected_scores = metric_bertscore.expected_scores(HYPOTHESES, REFERENCES)
        torch.testing.assert_close(
            expected_scores,
            SCORES.mean(dim=1).to(metric_bertscore.device),
            atol=0.0005 / 100,
            rtol=1e-6,
        )

    def test_corpus_score(self, metric_bertscore: MetricBERTScore):
        hyps = ["another test", "this is a test", "this is an test"]
        refs = ["another test", "this is a fest", "this is a test"]
        assert torch.isclose(
            torch.tensor(metric_bertscore.corpus_score(hyps, [refs])),
            torch.FloatTensor(
                [0.9999998807907104, 0.9457088112831116, 0.949202835559845]
            ).mean(),
        )

    def test_encode_batch(self, metric_bertscore: MetricBERTScore):
        N = metric_bertscore.cfg.batch_size + 2
        hyps = ["this is a test"] * N
        cache = metric_bertscore.encode(hyps)
        assert len(cache.embeddings) == N
        assert len(cache.idf_weights) == N


@pytest.mark.metrics_bertscore
@pytest.mark.parametrize(
    "score_type",
    [
        BERTScoreScoreType.precision,
        BERTScoreScoreType.recall,
        BERTScoreScoreType.f1,
    ],
)
def config_score_type(score_type: BERTScoreScoreType):
    bertscore = MetricBERTScore(
        MetricBERTScore.Config(score_type=score_type, lang="en")
    )
    torch.testing.assert_close(
        bertscore.pairwise_scores(HYPOTHESES, REFERENCES),
        (PRECISION_SCORES, RECALL_SCORES, SCORES)[score_type],
    )
