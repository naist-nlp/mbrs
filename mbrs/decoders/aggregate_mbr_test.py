import pytest
import torch

from mbrs.metrics import get_metric

from .aggregate_mbr import DecoderAggregateMBR

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

BEST_INDICES_CHRF = [2, 2, 0, 0]
BEST_SENTENCES_CHRF = [
    "this is an test",
    "this is a test",
    "this is a test",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SCORES_CHRF = torch.Tensor([63.75969, 40.86618, 100.0, 46.16121])

BEST_INDICES_BLEU = [1, 2, 0, 0]
BEST_SENTENCES_BLEU = [
    "this is a test",
    "this is a test",
    "this is a test",
    "Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;",
]
SCORES_BLEU = {
    True: torch.Tensor([42.36951, 50.0, 100.0, 8.49310]),
    False: torch.Tensor([42.36951, 50.0, 100.0, 8.49310]),
}


class TestDecoderAggregateMBR:
    @pytest.mark.parametrize("nbest", [1, 2])
    def test_decode_chrf(self, nbest: int):
        metric_type = "chrf"
        metric_cls = get_metric(metric_type)
        decoder = DecoderAggregateMBR(
            DecoderAggregateMBR.Config(), metric_cls(metric_cls.Config())
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, nbest=nbest, reference_lprobs=None)
            assert output.idx[0] == BEST_INDICES_CHRF[i]
            assert output.sentence[0] == BEST_SENTENCES_CHRF[i]
            assert len(output.sentence) == min(nbest, len(hyps))
            assert len(output.score) == min(nbest, len(hyps))
            torch.testing.assert_close(
                torch.tensor(output.score[0]),
                SCORES_CHRF[i],
                atol=0.0005,
                rtol=1e-4,
            )

            output = decoder.decode(
                hyps,
                refs,
                nbest=1,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs)),
            )
            assert output.idx[0] == BEST_INDICES_CHRF[i]
            assert output.sentence[0] == BEST_SENTENCES_CHRF[i]
            torch.testing.assert_close(
                torch.tensor(output.score[0]),
                SCORES_CHRF[i],
                atol=0.0005,
                rtol=1e-4,
            )

    @pytest.mark.parametrize("effective_order", [True, False])
    @pytest.mark.parametrize("nbest", [1, 2])
    def test_decode_bleu(self, nbest: int, effective_order: bool):
        metric_type = "bleu"
        metric_cls = get_metric(metric_type)
        decoder = DecoderAggregateMBR(
            DecoderAggregateMBR.Config(),
            metric_cls(metric_cls.Config(effective_order=effective_order)),
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, nbest=nbest, reference_lprobs=None)
            assert output.idx[0] == BEST_INDICES_BLEU[i]
            assert output.sentence[0] == BEST_SENTENCES_BLEU[i]
            assert len(output.sentence) == min(nbest, len(hyps))
            assert len(output.score) == min(nbest, len(hyps))
            torch.testing.assert_close(
                torch.tensor(output.score[0]),
                SCORES_BLEU[effective_order][i],
                atol=0.0005,
                rtol=1e-4,
            )

            output = decoder.decode(
                hyps,
                refs,
                nbest=1,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs)),
            )
            assert output.idx[0] == BEST_INDICES_BLEU[i]
            assert output.sentence[0] == BEST_SENTENCES_BLEU[i]
            torch.testing.assert_close(
                torch.tensor(output.score[0]),
                SCORES_BLEU[effective_order][i],
                atol=0.0005,
                rtol=1e-4,
            )
