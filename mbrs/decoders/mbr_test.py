import numpy as np
import pytest

from mbrs.metrics.bleu import MetricBLEU

from .mbr import DecoderMBR

HYPOTHESES = [
    ["another test", "this is a test", "this is a fest"],
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
    ["Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"],
]
REFERENCES = [
    ["another test", "this is a test", "this is a fest"],
    ["this is a test", "ref"],
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

SCORES_EFFECTIVE_ORDER = {
    True: np.array([58.478, 50.0, 100.0, 8.493], dtype=np.float32),
    False: np.array([58.478, 50.0, 100.0, 8.493], dtype=np.float32),
}


class TestDecoderMBR:
    @pytest.mark.parametrize("effective_order", [True, False])
    def test_decode(self, effective_order: bool):
        decoder = DecoderMBR(
            DecoderMBR.Config(),
            MetricBLEU(MetricBLEU.Config(effective_order=effective_order)),
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert np.allclose(
                np.array(output.score[0], dtype=np.float32),
                SCORES_EFFECTIVE_ORDER[effective_order][i],
                atol=0.0005,
            )
