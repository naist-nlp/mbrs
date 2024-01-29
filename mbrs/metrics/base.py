from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


class Metric(abc.ABC):
    """Base metric class."""

    def __init__(self, cfg: Config):
        pass

    @dataclass
    class Config: ...

    @abc.abstractmethod
    def score(
        self, hypothesis: str, reference: str, source: Optional[str] = None
    ) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            reference (str): A reference.
            source (str, optional): A source.

        Returns:
            float: The score of the given hypothesis.
        """

    def pairwise_score(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None
    ) -> npt.NDArray[np.float32]:
        """Calculate the pairwise scores for each hypothesis.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            NDArray[np.float32]: A score matrix of shape `(H, R)`, where
              - H: the number of hypotheses
              - R: the number of references
        """

        H, R = len(hypotheses), len(references)
        scores = np.zeros((H, R), dtype=np.float32)
        for i, hyp in enumerate(hypotheses):
            for j, ref in enumerate(references):
                scores[i, j] = self.score(hyp, ref)
        return scores
