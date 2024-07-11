import numpy as np
import torch

from mbrs.metrics.chrf import MetricChrF
from mbrs.metrics.comet import MetricCOMET

from .pruning_mbr import DecoderPruningMBR

SOURCE = [
    "これはテストです",
    "これはテストです",
    "これはテストです",
    "これはテストです",
]
HYPOTHESES = [
    ["another test", "this is a test", "this is a fest", "x", "this is test"],
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
    ["Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"],
]
REFERENCES = [
    ["another test", "this is a test", "this is a fest", "x", "this is test"],
    ["this is a test", "ref", "these are tests", "this is the test"],
    ["this is a test"],
    ["producţia de zahăr brut se exprimă în zahăr alb;"],
]

BEST_INDICES = [1, 2, 0, 0]
BEST_SENTENCES = [
    "this is a test",
    "this is a test",
    "this is a test",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SCORES_COMET = np.array([0.84780, 0.85304, 0.99257, 0.78060], dtype=np.float32)
SCORES_CHRF = np.array([48.912, 44.239, 100.0, 46.161], dtype=np.float32)


class TestDecoderPruningMBR:
    def test_decode_chrf(self):
        metric = MetricChrF(MetricChrF.Config())
        decoder = DecoderPruningMBR(
            DecoderPruningMBR.Config(sampling_scheduler=[2, 4, 8]), metric
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert np.allclose(
                np.array(output.score[0], dtype=np.float32), SCORES_CHRF[i], atol=0.0005
            )

            output = decoder.decode(
                hyps,
                refs,
                SOURCE[i],
                nbest=1,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs)),
            )
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert np.allclose(
                np.array(output.score[0], dtype=np.float32), SCORES_CHRF[i], atol=0.0005
            )

    def test_decode_comet(self, metric_comet: MetricCOMET):
        decoder = DecoderPruningMBR(
            DecoderPruningMBR.Config(sampling_scheduler=[2, 4, 8]), metric_comet
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert np.allclose(
                np.array(output.score[0], dtype=np.float32),
                SCORES_COMET[i],
                atol=0.0005,
            )
