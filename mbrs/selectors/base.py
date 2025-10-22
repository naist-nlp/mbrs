from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mbrs import registry


class Selector(abc.ABC):
    """Selector base class."""

    def __init__(self, cfg: Selector.Config) -> None:
        self.cfg = cfg

    @dataclass
    class Config:
        """Configuration for the selector."""

    @dataclass
    class Output:
        """
        - idx (list[int]): Index numbers of the n-best hypotheses.
        - sentence (list[str]): Sentences of the n-best hypotheses.
        - score (list[float]): Scores of the n-best hypotheses.
        """

        idx: list[int]
        sentence: list[str]
        score: list[float]

    def topk(
        self, x: Tensor, k: int = 1, maximize: bool = True
    ) -> tuple[list[float], list[int]]:
        """Return the top-k best elements and corresponding indices.

        Args:
            x (Tensor): Input 1-D array.
            k (int): Return the top-k values and indices.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            tuple[list[float], list[int]]
              - list[float]: The top-k values.
              - list[int]: The top-k indices.
        """
        values, indices = torch.topk(x, k=min(k, len(x)), largest=maximize)
        return values.tolist(), indices.tolist()

    def argbest(self, x: Tensor, maximize: bool = True) -> Tensor:
        """Return the index of the best element.

        Args:
            x (Tensor): Input 1-D array.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            Tensor: A scalar tensor of the best index.
        """
        if maximize:
            return torch.argmax(x)
        return torch.argmin(x)

    def superior(self, a: float, b: float, maximize: bool = True) -> bool:
        """Return whether the score `a` is superior to the score `b`.

        Args:
            a (float): A score.
            b (float): A score.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            bool: Return True when `a` is superior to `b`.
        """
        if maximize:
            return a > b
        return a < b

    @abc.abstractmethod
    def select(
        self,
        hypotheses: list[str],
        expected_scores: Tensor,
        nbest: int = 1,
        source: Optional[str] = None,
        maximize: bool = True,
        *args,
        **kwargs,
    ) -> Selector.Output:
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


register, get_selector = registry.Registry(Selector).get_closure()
