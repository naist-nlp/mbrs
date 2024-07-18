import pytest
import torch

from mbrs.decoders.aggregate_mbr import DecoderAggregateMBR
from mbrs.metrics.comet import MetricCOMET

from .centroid_mbr import DecoderCentroidMBR

SOURCE = [
    "これはテストです",
    "これはテストです",
    "これはテストです",
    "これはテストです",
]
HYPOTHESES = [
    ["another test", "this is a test", "this is a fest"],
    ["another test", "this is a fest", "this is a test"],
    ["this is a test"],
    ["Producția de zahăr primă va fi exprimată în ceea ce privește zahărul alb;"],
]
REFERENCES = [
    ["another test", "this is a test", "this is a fest"],
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
SCORES = torch.Tensor([0.88974, 0.76127, 0.99257, 0.78060])

NCENTROIDS = 2


class TestDecoderCBMBR:
    @pytest.mark.parametrize("kmeanspp", [True, False])
    def test_decode(self, metric_comet: MetricCOMET, kmeanspp: bool):
        decoder = DecoderCentroidMBR(
            DecoderCentroidMBR.Config(ncentroids=NCENTROIDS, kmeanspp=kmeanspp), metric_comet
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert torch.isclose(
                torch.tensor(output.score[0]),
                SCORES[i],
                atol=0.0005 / 100,
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
            assert torch.isclose(
                torch.tensor(output.score[0]),
                SCORES[i],
                atol=0.0005 / 100,
            )

    @pytest.mark.parametrize("kmeanspp", [True, False])
    def test_decode_equivalent_with_aggregate(
        self, metric_comet: MetricCOMET, kmeanspp: bool
    ):
        decoder_cbmbr = DecoderCentroidMBR(
            DecoderCentroidMBR.Config(ncentroids=1, kmeanspp=kmeanspp), metric_comet
        )
        decoder_aggregate = DecoderAggregateMBR(
            DecoderAggregateMBR.Config(), metric_comet
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output_cbmbr = decoder_cbmbr.decode(hyps, refs, SOURCE[i], nbest=1)
            output_aggregate = decoder_aggregate.decode(hyps, refs, SOURCE[i], nbest=1)
            assert output_cbmbr.idx[0] == output_aggregate.idx[0]
            assert output_cbmbr.sentence[0] == output_aggregate.sentence[0]
            assert torch.isclose(
                torch.tensor(output_cbmbr.score[0]),
                torch.tensor(output_aggregate.score[0]),
                atol=0.0005 / 100,
            )
