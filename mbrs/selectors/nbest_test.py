import pytest
import torch

from .nbest import SelectorNbest


class TestSelectorNbest:
    @pytest.mark.parametrize("nbest", [1, 3, 5])
    @pytest.mark.parametrize("maximize", [True, False])
    def test_select(self, nbest: int, maximize: bool):
        selector = SelectorNbest(SelectorNbest.Config())
        scores = torch.Tensor([0.5, 0.9, 0.8, 0.2, 0.4])
        sentences = ["a", "b", "c", "d", "e"]
        output = selector.select(sentences, scores, nbest=nbest, maximize=maximize)
        if maximize:
            expected_scores, expected_indices = scores.sort(descending=True)
        else:
            expected_scores, expected_indices = scores.sort()
        expected_sentences = [sentences[i] for i in expected_indices.tolist()]
        assert output.idx == expected_indices.tolist()[:nbest]
        assert output.sentence == expected_sentences[:nbest]
        assert torch.allclose(torch.tensor(output.score), expected_scores[:nbest])
