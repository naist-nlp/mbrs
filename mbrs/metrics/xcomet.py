from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from comet import download_model, load_from_checkpoint
from comet.models import XCOMETMetric
from torch import Tensor

from mbrs import timer, utils

from . import Metric, register


@register("xcomet")
class MetricXCOMET(Metric):
    """XCOMET metric class."""

    scorer: XCOMETMetric

    @dataclass
    class Config(Metric.Config):
        """XCOMET metric configuration.

        - model (str): Model name or path.
        - batch_size (int): Batch size.
        - fp16 (bool): Use float16 for the forward computation.
        - bf16 (bool): Use bfloat16 for the forward computation.
        - cpu (bool): Use CPU for the forward computation.
        """

        model: str = "Unbabel/XCOMET-XL"
        batch_size: int = 8
        fp16: bool = False
        bf16: bool = False
        cpu: bool = False

    def __init__(self, cfg: MetricXCOMET.Config):
        self.cfg = cfg
        self.scorer = load_from_checkpoint(download_model(cfg.model))
        self.scorer.eval()
        for param in self.scorer.parameters():
            param.requires_grad = False

        if not cfg.cpu and torch.cuda.is_available():
            if cfg.fp16:
                self.scorer = self.scorer.half()
            elif cfg.bf16:
                self.scorer = self.scorer.bfloat16()
            self.scorer = self.scorer.cuda()

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.scorer.device

    def score(
        self,
        hypothesis: str,
        reference: Optional[str] = None,
        source: Optional[str] = None,
    ) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            reference (str, optional): A reference.
            source (str, optional): A source.

        Returns:
            float: The score of the given hypothesis.
        """
        inputs = {"mt": hypothesis}
        if reference is not None:
            inputs["ref"] = reference
        if source is not None:
            inputs["src"] = source

        batch = self.scorer.prepare_for_inference([inputs])
        batch = utils.to_device(batch, self.device)
        model_output = self.scorer.predict_step(batch)
        return model_output.scores.item()

    def scores(
        self,
        hypotheses: list[str],
        references: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
    ) -> Tensor:
        """Calculate the scores of the given hypothesis.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str], optional): N references.
            sources (list[str], optional): N sources.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """
        inputs = [{"mt": hyp} for hyp in hypotheses]
        if references is not None:
            for d, ref in zip(inputs, references):
                d["ref"] = ref
        if sources is not None:
            for d, src in zip(inputs, sources):
                d["src"] = src

        scores = []
        with timer.measure("score") as t:
            t.set_delta_ncalls(len(inputs))
            for i in range(0, len(inputs), self.cfg.batch_size):
                batch = self.scorer.prepare_for_inference(
                    inputs[i : i + self.cfg.batch_size]
                )
                batch = utils.to_device(batch, self.device)
                model_output = self.scorer.predict_step(batch)
                scores.append(model_output.scores)
        return torch.cat(scores).view(len(hypotheses))

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> Tensor:
        """Calculate the pairwise scores.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        data = [
            {"src": source, "mt": hyp, "ref": ref}
            for hyp in hypotheses
            for ref in references
        ]
        scores = []
        with timer.measure("score") as t:
            t.set_delta_ncalls(len(data))
            for i in range(0, len(data), self.cfg.batch_size):
                batch = self.scorer.prepare_for_inference(
                    data[i : i + self.cfg.batch_size]
                )
                batch = utils.to_device(batch, self.device)
                model_output = self.scorer.predict_step(batch)
                scores.append(model_output.scores)
        return torch.cat(scores).view(len(hypotheses), len(references))

    def corpus_score(
        self,
        hypotheses: list[str],
        references: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
    ) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str], optional): References.
            sources (list[str], optional): Sources.

        Returns:
            float: The corpus score.
        """
        scores = []
        for i in range(0, len(hypotheses), self.cfg.batch_size):
            scores.append(
                self.scores(
                    hypotheses[i : i + self.cfg.batch_size],
                    references[i : i + self.cfg.batch_size] if references is not None else None,
                    sources[i : i + self.cfg.batch_size] if sources is not None else None,
                )
                .float()
                .cpu()
            )
        return torch.cat(scores).mean().item()
