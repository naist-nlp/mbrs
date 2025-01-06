import pytest
import torch

from mbrs.decoders.aggregate_mbr import DecoderAggregateMBR
from mbrs.metrics.comet import MetricCOMET
from mbrs.modules.kmeans import Kmeans
from mbrs.selectors.base import Selector

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
            DecoderCentroidMBR.Config(
                Kmeans.Config(ncentroids=NCENTROIDS, kmeanspp=kmeanspp)
            ),
            metric_comet,
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

    @pytest.mark.parametrize("nbest", [1, 2])
    @pytest.mark.parametrize("kmeanspp", [True, False])
    def test_decode_equivalent_with_aggregate(
        self, metric_comet: MetricCOMET, nbest: int, kmeanspp: bool, selector: Selector
    ):
        decoder_cbmbr = DecoderCentroidMBR(
            DecoderCentroidMBR.Config(Kmeans.Config(ncentroids=1, kmeanspp=kmeanspp)),
            metric_comet,
            selector=selector,
        )
        decoder_aggregate = DecoderAggregateMBR(
            DecoderAggregateMBR.Config(), metric_comet, selector=selector
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output_cbmbr = decoder_cbmbr.decode(hyps, refs, SOURCE[i], nbest=nbest)
            output_aggregate = decoder_aggregate.decode(
                hyps, refs, SOURCE[i], nbest=nbest
            )
            assert output_cbmbr.idx == output_aggregate.idx
            assert output_cbmbr.sentence == output_aggregate.sentence
            torch.testing.assert_close(
                torch.tensor(output_cbmbr.score),
                torch.tensor(output_aggregate.score),
                rtol=1e-6,
                atol=0.0005 / 100,
            )

    @pytest.mark.parametrize("nbest", [1, 2])
    def test_decode_selector(
        self, metric_comet: MetricCOMET, nbest: int, selector: Selector
    ):
        decoder = DecoderCentroidMBR(
            DecoderCentroidMBR.Config(Kmeans.Config(ncentroids=NCENTROIDS)),
            metric_comet,
            selector=selector,
        )
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, SOURCE[i], nbest=nbest)
            assert len(output.sentence) == min(nbest, len(hyps))
            assert len(output.score) == min(nbest, len(hyps))

            output = decoder.decode(
                hyps,
                refs,
                SOURCE[i],
                nbest=nbest,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs)),
            )
            assert len(output.sentence) == min(nbest, len(hyps))
            assert len(output.score) == min(nbest, len(hyps))
