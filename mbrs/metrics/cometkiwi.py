from __future__ import annotations

from dataclasses import dataclass

import torch
from comet import download_model, load_from_checkpoint

from mbrs import utils

from . import MetricReferenceless, register


@register("cometkiwi")
class MetricCOMETkiwi(MetricReferenceless):
    """COMETkiwi metric class."""

    @dataclass
    class Config(MetricReferenceless.Config):
        """COMETkiwi metric configuration.

        - model (str): Model name or path.
        - batch_size (int): Batch size.
        - fp16 (bool): Use float16 for the forward computation.
        - bf16 (bool): Use bfloat16 for the forward computation.
        - cpu (bool): Use CPU for the forward computation.
        """

        model: str = "Unbabel/wmt22-cometkiwi-da"
        batch_size: int = 64
        fp16: bool = False
        bf16: bool = False
        cpu: bool = False

    def __init__(self, cfg: MetricCOMETkiwi.Config):
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

    def score(self, hypothesis: str, source: str) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            source (str): A source.

        Returns:
            float: The score of the given hypothesis.
        """
        return self.scores([hypothesis], [source]).item()

    def scores(self, hypotheses: list[str], sources: list[str]) -> torch.Tensor:
        """Calculate the scores of hypotheses.

        Args:
            hypotheses (list[str]): N hypotheses.
            source (list[str]): N sources.

        Returns:
            torch.Tensor: N scores of the given hypotheses.
        """
        data = [{"src": src, "mt": hyp} for hyp, src in zip(hypotheses, sources)]
        scores = []
        for i in range(0, len(data), self.cfg.batch_size):
            batch = self.scorer.prepare_for_inference(data[i : i + self.cfg.batch_size])
            batch = utils.to_device(batch, self.device)
            model_output = self.scorer.predict_step(batch)
            scores.append(model_output.scores)
        return torch.cat(scores).view(len(hypotheses))

    def corpus_score(self, hypotheses: list[str], sources: list[str]) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            source (list[str]): Sources.

        Returns:
            float: The corpus score.
        """
        scores = []
        for i in range(0, len(hypotheses), self.cfg.batch_size):
            scores.append(
                self.scores(
                    hypotheses[i : i + self.cfg.batch_size],
                    sources[i : i + self.cfg.batch_size],
                )
                .float()
                .cpu()
            )
        return torch.cat(scores).mean().item()
