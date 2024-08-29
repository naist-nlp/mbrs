from typing import Type

import pytest
import torch

from mbrs.metrics import Metric, MetricEnum, Metrics, get_metric

from .diverse import SelectorDiverse


class TestSelectorDiverse:
    @pytest.mark.parametrize("diversity_metric", [Metrics.bleu, Metrics.chrf])
    def test_init(self, diversity_metric: MetricEnum):
        diversity_metric_type: Type[Metric] = get_metric(diversity_metric)
        diversity_metric_config = diversity_metric_type.Config()
        selector = SelectorDiverse(
            SelectorDiverse.Config(
                diversity_metric=diversity_metric,
                diversity_metric_config=diversity_metric_config,
            )
        )
        assert isinstance(selector.diversity_metric, diversity_metric_type)
        assert isinstance(selector.diversity_metric.cfg, diversity_metric_type.Config)

    @pytest.mark.parametrize("diversity_lambda", [0.0, 0.1, 0.5, 1.0, 10.0])
    def test_compute_objective(self, diversity_lambda: float):
        selector = SelectorDiverse(
            SelectorDiverse.Config(diversity_lambda=diversity_lambda)
        )
        expected_scores = torch.arange(5.0)
        hypothesis_dissimilarities = torch.arange(5.0).repeat(5, 1).fill_diagonal_(0.0)
        mask = torch.BoolTensor([True, True, False, True, False])
        obj = selector.compute_objective(
            expected_scores, hypothesis_dissimilarities, mask
        )

        expected_score = (expected_scores[mask].sum() / mask.sum().float()).item()
        _hypothesis_dissimilarities = hypothesis_dissimilarities.clone()
        _hypothesis_dissimilarities[~mask] = 0.0
        _hypothesis_dissimilarities[:, ~mask] = 0.0
        k = mask.sum().item()
        hypothesis_dissimilarity = (
            _hypothesis_dissimilarities.sum() / k / max(k - 1, 1)
        ).item()
        torch.testing.assert_close(obj.expected_score, expected_score)
        torch.testing.assert_close(obj.diversity_score, hypothesis_dissimilarity)
        torch.testing.assert_close(
            obj.score, expected_score + diversity_lambda * hypothesis_dissimilarity
        )

    @pytest.mark.parametrize("maximize", [True, False])
    def test_search_greedy_best_first(self, maximize: bool):
        selector = SelectorDiverse(SelectorDiverse.Config(diversity_lambda=1.1))
        expected_scores = torch.arange(4.0) + 1.0
        hypothesis_dissimilarities = (
            torch.arange(4.0, 0.0, -1.0).repeat(4, 1).fill_diagonal_(0.0)
        )

        mask_patterns = [
            [False, False, False, True],
            [False, False, True, False],
            [False, True, False, False],
            [True, False, False, False],
        ]
        objs = [
            selector.compute_objective(
                expected_scores, hypothesis_dissimilarities, torch.tensor(mask)
            ).score
            for mask in mask_patterns
        ]
        topk_scores, topk_indices = torch.tensor(objs).topk(k=1, largest=maximize)
        selections = selector.search_greedy_best_first(
            expected_scores, hypothesis_dissimilarities, nbest=1, maximize=maximize
        )
        torch.testing.assert_close(
            selections, torch.tensor(mask_patterns[topk_indices.item()])
        )

        mask_patterns = [
            [False, False, True, True],
            [False, True, False, True],
            [True, False, False, True],
            [False, True, True, False],
            [True, False, True, False],
            [True, True, False, False],
        ]
        mask_patterns = [
            mask
            for mask in mask_patterns
            if (torch.tensor(mask) * selections).sum() >= 1
        ]
        objs = [
            selector.compute_objective(
                expected_scores, hypothesis_dissimilarities, torch.tensor(mask)
            ).score
            for mask in mask_patterns
        ]
        topk_scores, topk_indices = torch.tensor(objs).topk(k=1, largest=maximize)
        selections = selector.search_greedy_best_first(
            expected_scores, hypothesis_dissimilarities, nbest=2, maximize=maximize
        )
        torch.testing.assert_close(
            selections, torch.tensor(mask_patterns[topk_indices.item()])
        )

    @pytest.mark.parametrize("maximize", [True, False])
    def test_search_local(self, maximize: bool):
        selector = SelectorDiverse(
            SelectorDiverse.Config(diversity_lambda=1.1, local_search_neighbors=1)
        )
        expected_scores = torch.arange(4.0) + 1.0
        hypothesis_dissimilarities = (
            torch.arange(4.0, 0.0, -1.0).repeat(4, 1).fill_diagonal_(0.0)
        )

        mask_patterns = [
            [False, False, True, True],
            [False, True, False, True],
            [True, False, False, True],
            [False, True, True, False],
            [True, False, True, False],
            [True, True, False, False],
        ]
        solutions = []
        for mask in mask_patterns[:3] if maximize else mask_patterns[3:]:
            selections = selector.search_local(
                expected_scores,
                hypothesis_dissimilarities,
                torch.tensor(mask),
                nbest=2,
                maximize=maximize,
            )
            solutions.append(selections)
        assert torch.unique(torch.stack(solutions), dim=0).size(0) == 1
        objs = [
            selector.compute_objective(
                expected_scores, hypothesis_dissimilarities, torch.tensor(mask)
            ).score
            for mask in mask_patterns
        ]
        topk_scores, topk_indices = torch.tensor(objs).topk(k=1, largest=maximize)
        torch.testing.assert_close(
            torch.unique(torch.stack(solutions), dim=0)[0],
            torch.tensor(mask_patterns[topk_indices.item()]),
        )

    @pytest.mark.parametrize("diversity_metric", [Metrics.bleu, Metrics.chrf])
    @pytest.mark.parametrize("nbest", [1, 3, 5])
    @pytest.mark.parametrize("maximize", [True, False])
    def test_select(self, diversity_metric: MetricEnum, nbest: int, maximize: bool):
        diversity_metric_type: Type[Metric] = get_metric(diversity_metric.value)
        diversity_metric_config = diversity_metric_type.Config()
        selector = SelectorDiverse(
            SelectorDiverse.Config(
                diversity_metric=diversity_metric,
                diversity_metric_config=diversity_metric_config,
                diversity_lambda=0.5,
                local_search_iterations=100,
            )
        )
        scores = torch.Tensor([0.5, 0.9, 0.8, 0.2, 0.4]) * 100.0
        sentences = ["a", "a", "a", "b", "b"]
        output = selector.select(sentences, scores, nbest=nbest, maximize=maximize)
        if maximize:
            if nbest < 5:
                expected_indices = [1, 2, 4, 0, 3]
            else:
                expected_indices = [1, 2, 0, 4, 3]
        else:
            if nbest < 5:
                expected_indices = [3, 4, 0, 2, 1]
            else:
                expected_indices = [3, 4, 0, 2, 1]
        expected_sentences = [sentences[i] for i in expected_indices]
        assert output.idx == expected_indices[:nbest]
        assert output.sentence == expected_sentences[:nbest]
