from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from mbrs.modules import topk


class Metric(abc.ABC):
    """Base metric class."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config: ...

    def topk(
        self, x: npt.NDArray[np.float32], k: int = 1
    ) -> tuple[list[float], list[int]]:
        """Return the top-k best elements and corresponding indices.

        Args:
            x (NDArray[np.float32]): Input 1-D array.
            k (int): Return the top-k values and indices.

        Returns:
            tuple[list[float], list[int]]
              - list[float]: The top-k values.
              - list[int]: The top-k indices.
        """
        values, indices = topk(x, k=min(k, len(x)), largest=self.HIGHER_IS_BETTER)
        return values.tolist(), indices.tolist()

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

    def expected_scores(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> npt.NDArray[np.float32]:
        """Calculate the expected scores for each hypothesis.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            NDArray[np.float32]: The expected scores for each hypothesis.
        """
        return np.array(
            [
                [self.score(hyp, ref, source) for ref in references]
                for hyp in hypotheses
            ],
            dtype=np.float32,
        ).mean(axis=1)


class MetricNeural(Metric, metaclass=abc.ABCMeta):
    """Base metric class for neural network.

    This class supports to cache intermediate representations of the encoder."""

    @dataclass
    class IR:
        """Intermediate representations."""

    @abc.abstractmethod
    def encode(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> IR:
        """Encode the given sentences into their intermediate representations.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            IR: Intermediate representations.
        """

    @abc.abstractmethod
    def out_proj(self, ir: IR) -> torch.Tensor:
        """Forward the output projection layer.

        Args:
            ir (MetricNeural.IR): Intermediate representations
              computed by the `encode` method.

        Returns:
            torch.Tensor: H x R score matrix, where
              - H: the number of hypotheses
              - R: the number of references
        """

    def score(
        self,
        hypothesis: str,
        reference: str,
        source: Optional[str] = None,
    ) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            reference (str): A reference.
            source (str, optional): A source.

        Returns:
            float: The score of the given hypothesis.
        """
        return self.out_proj(self.encode([hypothesis], [reference], source)).item()

    def expected_scores(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> npt.NDArray[np.float32]:
        """Calculate the expected scores for each hypothesis.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            NDArray[np.float32]: The expected scores for each hypothesis.
        """

        ir = self.encode(hypotheses, references, source)
        return self.out_proj(ir).mean(dim=1).cpu().float().numpy()


class MetricReferenceless(abc.ABC):
    """Base class for reference-less metrics like quality estimation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    HIGHER_IS_BETTER: bool = True

    @dataclass
    class Config: ...

    def topk(
        self, x: npt.NDArray[np.float32], k: int = 1
    ) -> tuple[list[float], list[int]]:
        """Return the top-k best elements and corresponding indices.

        Args:
            x (NDArray[np.float32]): Input 1-D array.
            k (int): Return the top-k values and indices.

        Returns:
            tuple[list[float], list[int]]
              - list[float]: The top-k values.
              - list[int]: The top-k indices.
        """
        values, indices = topk(x, k=min(k, len(x)), largest=self.HIGHER_IS_BETTER)
        return values.tolist(), indices.tolist()

    @abc.abstractmethod
    def score(self, hypothesis: str, source: str) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            source (str): A source.

        Returns:
            float: The score of the given hypothesis.
        """

    def scores(self, hypotheses: list[str], source: str) -> npt.NDArray[np.float32]:
        """Calculate the scores of hypotheses.

        Args:
            hypotheses (list[str]): Hypotheses.
            source (str): A source.

        Returns:
            NDArray[np.float32]: The scores of hypotheses.
        """
        return np.array([self.score(hyp, source) for hyp in hypotheses])
