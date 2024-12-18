from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import Optional

import torch
import transformers
from bert_score import BERTScorer
from simple_parsing.helpers.fields import choice
from torch import Tensor

from mbrs import timer

from . import Metric, register

transformers.logging.set_verbosity_error()


class BERTScoreScoreType(int, enum.Enum):
    precision = 0
    recall = 1
    f1 = 2


@register("bertscore")
class MetricBERTScore(Metric):
    """BERTScore metric class."""

    scorer: BERTScorer

    @dataclass
    class Config(Metric.Config):
        """BERTScore metric configuration.

        - score_type (BERTScoreScoreType): The output score type, i.e.,
            precision, recall, or f1.
        - model_type (str): Contexual embedding model specification, default using the
            suggested model for the target langauge; has to specify at least one of
            `model_type` or `lang`.
        - num_layers (int): The layer of representation to use. Default using the number
            of layer tuned on WMT16 correlation data.
        - idf (bool): A booling to specify whether to use idf or not. (This should be
            True even if `idf_sents` is given.)
        - idf_sents (list[str]): List of sentences used to compute the idf weights.
        - batch_size (int): Bert score processing batch size
        - nthreads (int): Number of threads.
        - lang (str): Language of the sentences; has to specify at least one of
            `model_type` or `lang`. `lang` needs to be specified when
            `rescale_with_baseline` is True.
        - rescale_with_baseline (bool): Rescale bertscore with pre-computed baseline.
        - baseline_path (str): Customized baseline file.
        - use_fast_tokenizer (bool): `use_fast` parameter passed to HF tokenizer.
        - fp16 (bool): Use float16 for the forward computation.
        - bf16 (bool): Use bfloat16 for the forward computation.
        - cpu (bool): Use CPU for the forward computation.
        """

        score_type: BERTScoreScoreType = choice(
            BERTScoreScoreType, default=BERTScoreScoreType.f1
        )
        model_type: Optional[str] = None
        num_layers: Optional[int] = None
        batch_size: int = 64
        nthreads: int = 4
        all_layers: bool = False
        idf: bool = False
        idf_sents: Optional[list[str]] = None
        lang: Optional[str] = None
        rescale_with_baseline: bool = False
        baseline_path: Optional[str] = None
        use_fast_tokenizer: bool = False
        fp16: bool = False
        bf16: bool = False
        cpu: bool = False

    def __init__(self, cfg: MetricBERTScore.Config):
        self.cfg = cfg
        self.scorer = BERTScorer(
            model_type=cfg.model_type,
            num_layers=cfg.num_layers,
            batch_size=cfg.batch_size,
            nthreads=cfg.nthreads,
            all_layers=cfg.all_layers,
            idf=cfg.idf,
            idf_sents=cfg.idf_sents,
            device="cpu" if cfg.cpu else None,
            lang=cfg.lang,
            rescale_with_baseline=cfg.rescale_with_baseline,
            baseline_path=cfg.baseline_path,
            use_fast_tokenizer=cfg.use_fast_tokenizer,
        )
        self.scorer._model.eval()
        for param in self.scorer._model.parameters():
            param.requires_grad = False

        if not cfg.cpu and torch.cuda.is_available():
            if cfg.fp16:
                self.scorer._model = self.scorer._model.half()
            elif cfg.bf16:
                self.scorer._model = self.scorer._model.bfloat16()
            self.scorer._model = self.scorer._model.cuda()

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.scorer._model.device

    def _choose_output_score(self, triplet: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Choose the output score from the triplet of precision, recall, and f1 scores.

        Args:
            triplet (tuple[Tensor, Tensor, Tensor]): A triplet of precision, recall, and f1 scores.

        Returns:
            Tensor: Output score.
        """
        return triplet[self.cfg.score_type]

    def score(self, hypothesis: str, reference: str, *_, **__) -> float:
        """Calculate the score of the given hypothesis.

        Args:
            hypothesis (str): A hypothesis.
            reference (str): A reference.

        Returns:
            float: The score of the given hypothesis.
        """
        return self._choose_output_score(
            self.scorer.score(
                [hypothesis],
                [reference],
                batch_size=self.cfg.batch_size,
            )
        ).item()

    def scores(self, hypotheses: list[str], references: list[str], *_, **__) -> Tensor:
        """Calculate the scores of the given hypothesis.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str]): N references.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """

        with timer.measure("score") as t:
            t.set_delta_ncalls(len(hypotheses))
            return self._choose_output_score(
                self.scorer.score(
                    hypotheses,
                    references,
                    batch_size=self.cfg.batch_size,
                )
            ).view(len(hypotheses))

    def pairwise_scores(
        self, hypotheses: list[str], references: list[str], *_, **__
    ) -> Tensor:
        """Calculate the pairwise scores.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            Tensor: Score matrix of shape `(H, R)`, where `H` is the number
              of hypotheses and `R` is the number of references.
        """
        hyps, refs = tuple(zip(*itertools.product(hypotheses, references)))
        with timer.measure("score") as t:
            t.set_delta_ncalls(len(hypotheses) * len(references))
            return self._choose_output_score(
                self.scorer.score(hyps, refs, batch_size=self.cfg.batch_size)
            ).view(len(hypotheses), len(references))

    def corpus_score(
        self, hypotheses: list[str], references: list[str], *_, **__
    ) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.

        Returns:
            float: The corpus score.
        """
        return self.scores(hypotheses, references).mean().item()
