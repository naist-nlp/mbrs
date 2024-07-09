from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from bleurt_pytorch import (
    BleurtForSequenceClassification,
    BleurtTokenizer,
)
from torch import Tensor

from mbrs import timer

from . import Metric, register


@register("bleurt")
class MetricBLEURT(Metric):
    """BLEURT metric class.

    We employ the PyTorch port version to implement this metric instead of the original version:
    https://github.com/lucadiliello/bleurt-pytorch
    (thanks to @lucadiliello)

    Available checkpoints:
    - lucadiliello/BLEURT-20
    - lucadiliello/BLEURT-20-D12
    - lucadiliello/BLEURT-20-D3
    - lucadiliello/BLEURT-20-D6
    - lucadiliello/bleurt-base-128
    - lucadiliello/bleurt-base-512
    - lucadiliello/bleurt-large-128
    - lucadiliello/bleurt-large-512
    - lucadiliello/bleurt-tiny-128
    - lucadiliello/bleurt-tiny-512
    """

    scorer: BleurtForSequenceClassification

    @dataclass
    class Config(Metric.Config):
        """BLEURT metric configuration.

        - model (str): Model name or path.
        - batch_size (int): Batch size.
        - fp16 (bool): Use float16 for the forward computation.
        - bf16 (bool): Use bfloat16 for the forward computation.
        - cpu (bool): Use CPU for the forward computation.
        """

        model: str = "lucadiliello/BLEURT-20-D12"
        batch_size: int = 64
        fp16: bool = False
        bf16: bool = False
        cpu: bool = False

    def __init__(self, cfg: MetricBLEURT.Config):
        self.cfg = cfg
        self.scorer = BleurtForSequenceClassification.from_pretrained(cfg.model)
        self.tokenizer = BleurtTokenizer.from_pretrained(cfg.model)
        self.max_length = self.tokenizer.max_model_input_sizes[
            self.tokenizer.name_or_path
        ]
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

    def score(self, hypothesis: str, reference: str, *_) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            reference (str): A reference.

        Returns:
            float: The score of the given hypothesis.
        """
        batch = self.tokenizer(
            [reference],
            [hypothesis],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        model_output = self.scorer(**batch)
        return model_output.logits.flatten().tolist()[0]

    def scores(self, hypotheses: list[str], references: list[str], *_) -> Tensor:
        """Calculate the scores of the given hypothesis.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str]): N references.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """

        scores = []
        with timer.measure("score") as t:
            t.set_delta_ncalls(len(hypotheses))
            for i in range(0, len(hypotheses), self.cfg.batch_size):
                batch = self.tokenizer(
                    references[i : i + self.cfg.batch_size],
                    hypotheses[i : i + self.cfg.batch_size],
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                model_output = self.scorer(**batch)
                scores.append(model_output.logits.flatten())
        return torch.cat(scores).view(len(hypotheses))

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], *_
    ) -> Tensor:
        """Calculate the pairwise scores.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        scores = []
        pairwise_iter = itertools.product(hypotheses, references)
        while batch := tuple(
            zip(*itertools.islice(pairwise_iter, self.cfg.batch_size))
        ):
            with timer.measure("score") as t:
                hyps, refs = batch
                t.set_delta_ncalls(len(hyps))
                batch = self.tokenizer(
                    refs,
                    hyps,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                model_output = self.scorer(**batch)
                scores.append(model_output.logits.flatten())
        return torch.cat(scores).view(len(hypotheses), len(references))

    def corpus_score(self, hypotheses: list[str], references: list[str]) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            float: The corpus score.
        """
        scores = []
        for i in range(0, len(hypotheses), self.cfg.batch_size):
            scores.append(
                self.scores(
                    hypotheses[i : i + self.cfg.batch_size],
                    references[i : i + self.cfg.batch_size],
                )
                .float()
                .cpu()
            )
        return torch.cat(scores).mean().item()
