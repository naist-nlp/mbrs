from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import timer
from mbrs.metrics import Metric, Metrics, get_metric
from mbrs.selectors import Selector, register


@register("diverse")
class SelectorDiverse(Selector):
    def __init__(self, cfg: SelectorDiverse.Config) -> None:
        super().__init__(cfg)
        self.diversity_metric: Metric = get_metric(cfg.diversity_metric)(
            cfg.diversity_metric_config
        )

    @dataclass
    class Config(Selector.Config):
        """Configuration for the selector."""

        diversity_metric: Metrics = Metrics.bleu
        diversity_metric_config: Metric.Config | None = None
        diversity_lambda: float = 0.1
        local_search_iterations: int = 20
        local_search_neighbors: int = 1
        seed: int = 0

        def __post_init__(self):
            if self.diversity_metric_config is None:
                self.diversity_metric_config = get_metric(
                    self.diversity_metric
                ).Config()

    cfg: Config

    @dataclass
    class Output(Selector.Output):
        """
        - idx (list[int]): Index numbers of the n-best hypotheses.
        - sentence (list[str]): Sentences of the n-best hypotheses.
        - score (list[float]): Scores of the n-best hypotheses.
        - nbest_objective_score: (float): Objective score for the n-best list.
        - nbest_expected_score (float): Expected score for the n-best list.
        - nbest_diversity_score: (float): Diversity score for the n-best list.
        """

        nbest_objective_score: float
        nbest_expected_score: float
        nbest_diversity_score: float

    @dataclass
    class Objective:
        """
        - score (float): The objective score.
        - expected_score (float): Expected score for the n-best list.
        - diversity_score (float): Diversity score for the n-best list.
        """

        score: float
        expected_score: float
        diversity_score: float

    def compute_objective(
        self, expected_scores: Tensor, hypothesis_dissimilarities: Tensor, mask: Tensor
    ) -> Objective:
        """Compute the objective function.

        Args:
            expected_scores (Tensor): The expected scores for each hypothesis. The shape is `(H,)`.
            hypothesis_dissimilarities (Tensor): The pairwise dissimilarities for all hypotheses. The shape is `(H, H)`.
            mask (Tensor): Boolean tensor of shape `(H,)`. The positions of True elements are calculated
              in the objective and the others are discarded.

        Returns:
            Objective: The objective scores that contain the expected score, diversity score, and the sum of them.
        """
        hypothesis_dissimilarities = hypothesis_dissimilarities.clone().float()
        hypothesis_dissimilarities = hypothesis_dissimilarities.fill_diagonal_(0.0)
        k = mask.sum().item()
        expected_score = expected_scores[mask].float().sum() / k
        hypothesis_dissimilarity = (
            mask.float() @ hypothesis_dissimilarities @ mask.float() / k / max(k - 1, 1)
        )
        objective = (
            expected_score + self.cfg.diversity_lambda * hypothesis_dissimilarity
        ).item()
        return self.Objective(
            objective, expected_score.item(), hypothesis_dissimilarity.item()
        )

    def search_greedy_best_first(
        self,
        expected_scores: Tensor,
        hypothesis_dissimilarities: Tensor,
        nbest: int = 1,
        maximize: bool = True,
    ) -> Tensor:
        """Search the solution by greedy best first search.

        Args:
            expected_scores (Tensor): The expected scores for each hypothesis. The shape is `(H,)`.
            hypothesis_dissimilarities (Tensor): The pairwise dissimilarities for all hypotheses. The shape is `(H, H)`.
            nbest (int): The number of final outputs.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            Tensor: Boolean tensor of shape `(H,)` where True positions indicate that they are selected.
        """
        H = expected_scores.size(0)
        selections = torch.zeros(H, dtype=torch.bool, device=expected_scores.device)
        for k in range(nbest):
            best = float("-inf") if maximize else float("inf")
            best_i = -1
            for i in range(H):
                if selections[i]:
                    continue
                selection_candidate = selections.clone()
                selection_candidate[i] = True
                objective = self.compute_objective(
                    expected_scores, hypothesis_dissimilarities, selection_candidate
                )

                if self.superior(objective.score, best, maximize=maximize):
                    best = objective.score
                    best_i = i
            selections[best_i] = True

        return selections

    def search_local(
        self,
        expected_scores: Tensor,
        hypothesis_dissimilarities: Tensor,
        initial_selections: Tensor,
        nbest: int = 1,
        maximize: bool = True,
    ) -> Tensor:
        """Search the solution by greedy best first search.

        Args:
            expected_scores (Tensor): The expected scores for each hypothesis. The shape is `(H,)`.
            hypothesis_dissimilarities (Tensor): The pairwise dissimilarities for all hypotheses. The shape is `(H, H)`.
            initial_selections (Tensor): Boolean tensor of shape `(H,)` where True positions indicate that they are selected.
            nbest (int): The number of final outputs.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            Tensor: Boolean tensor of shape `(H,)` where True positions indicate that they are selected.
        """
        rng = torch.Generator(device=expected_scores.device).manual_seed(self.cfg.seed)
        H = initial_selections.size(0)
        selections = initial_selections.clone()

        num_neighbors = min(self.cfg.local_search_neighbors, nbest)

        for i in range(self.cfg.local_search_iterations):
            prev_selections = selections.clone()
            selection_indices = selections.nonzero(as_tuple=True)[0]
            removed_candidates = torch.randperm(
                nbest, generator=rng, device=rng.device
            )[:num_neighbors]

            for k in range(num_neighbors):
                selections[selection_indices[removed_candidates[k]]] = False

            for k in range(num_neighbors):
                best = float("-inf") if maximize else float("inf")
                best_i = -1
                for i in range(H):
                    if selections[i]:
                        continue
                    selection_candidate = selections.clone()
                    selection_candidate[i] = True
                    objective = self.compute_objective(
                        expected_scores, hypothesis_dissimilarities, selection_candidate
                    )
                    if self.superior(objective.score, best, maximize=maximize):
                        best = objective.score
                        best_i = i
                selections[best_i] = True

            prev_objective = self.compute_objective(
                expected_scores, hypothesis_dissimilarities, prev_selections
            )
            new_objective = self.compute_objective(
                expected_scores, hypothesis_dissimilarities, selections
            )
            if self.superior(
                prev_objective.score, new_objective.score, maximize=maximize
            ):
                selections = prev_selections

        return selections

    def select(
        self,
        hypotheses: list[str],
        expected_scores: Tensor,
        nbest: int = 1,
        source: Optional[str] = None,
        maximize: bool = True,
        **kwargs,
    ) -> SelectorDiverse.Output:
        """Select the final output list.

        Args:
            hypotheses (list[str]): Hypotheses.
            expected_scores (Tensor): The expected scores for each hypothesis.
            nbest (int): Return the n-best hypotheses based on the selection rule.
            source (str, optional): A source.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            Selector.Output: Selected hypotheses.
        """
        nbest = min(len(hypotheses), nbest)
        with timer.measure("dissimilarity_calculation"):
            hypothesis_dissimilarities = self.diversity_metric.pairwise_scores(
                hypotheses,
                hypotheses,
                source=source,
            ).to(expected_scores)
            if maximize:
                hypothesis_dissimilarities *= -1
        with timer.measure("search/greedy_best_first"):
            selections = self.search_greedy_best_first(
                expected_scores,
                hypothesis_dissimilarities,
                nbest=nbest,
                maximize=maximize,
            )
        with timer.measure("search/local"):
            selections = self.search_local(
                expected_scores,
                hypothesis_dissimilarities,
                selections,
                nbest=nbest,
                maximize=maximize,
            )
        objective = self.compute_objective(
            expected_scores, hypothesis_dissimilarities, selections
        )
        topk_scores, topk_order = self.topk(
            expected_scores[selections].float(), k=nbest, maximize=maximize
        )
        selected_idx_set = [
            i
            for i, selection in zip(range(len(hypotheses)), selections.tolist())
            if selection
        ]
        topk_indices = [selected_idx_set[i] for i in topk_order]
        return self.Output(
            idx=topk_indices,
            sentence=[hypotheses[idx] for idx in topk_indices],
            score=topk_scores,
            nbest_objective_score=objective.score,
            nbest_expected_score=objective.expected_score,
            nbest_diversity_score=objective.diversity_score,
        )
