import numpy as np
import pytest

from mbrs.metrics import get_metric

from .mbr import DecoderMBR

HYPOTHESES = [
    ["another test", "this is a test", "this is an test", "this is a fest"],
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
    ["Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"],
]
REFERENCES = [
    ["another test", "this is a test", "this is an test", "this is a fest"],
    ["this is a test", "ref1 ref2 ref3 ref4"],
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

SCORES = {
    "bleu": np.array([52.697, 50.0, 100.0, 8.493], dtype=np.float32),
    "ter": np.array([50.000, 50.000, 0.000, 100.000], dtype=np.float32),
}


class TestDecoderMBR:
    @pytest.mark.parametrize("metric_type", ["bleu", "ter"])
    def test_decode(self, metric_type: str):
        metric_cls = get_metric(metric_type)
        decoder = DecoderMBR(DecoderMBR.Config(), metric_cls(metric_cls.Config()))
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert np.allclose(
                np.array(output.score[0], dtype=np.float32),
                SCORES[metric_type][i],
                atol=0.0005,
            )
