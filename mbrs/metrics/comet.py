from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from comet import download_model, load_from_checkpoint

from . import Metric, register


@register("comet")
class MetricCOMET(Metric):
    """COMET metric class."""

    @dataclass
    class Config(Metric.Config):
        """COMET metric configuration.

        - model (str): Model name or path.
        - batch_size (int): Batch size.
        - float16 (bool): Use float16 for the forward computation.
        - cpu (bool): Use CPU for the forward computation.
        """

        model: str = "Unbabel/wmt22-comet-da"
        batch_size: int = 64
        float16: bool = False
        cpu: bool = False

    def __init__(self, cfg: MetricCOMET.Config):
        self.cfg = cfg
        self.scorer = load_from_checkpoint(download_model(cfg.model))
        self.scorer.eval()
        for param in self.scorer.parameters():
            param.requires_grad = False

        if not cfg.cpu and torch.cuda.is_available():
            self.scorer = self.scorer.cuda()
            if cfg.float16:
                self.scorer = self.scorer.half()

    @property
    def device(self) -> torch.device:
        return self.scorer.device

    def compute_sentence_embedding(
        self, sentences: list[str], batch_size: int
    ) -> torch.Tensor:
        """Compute sentence embedding vectors of the given sentences.

        Args:
            sentences (list[str]): Input sentences.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Sentence embeddings of shape `(N, D)`, where
              - N: the number of sentences
              - D: size of the embedding dimmension
        """
        batches = [
            self.scorer.encoder.prepare_sample(sentences[i : i + batch_size])
            for i in range(0, len(sentences), batch_size)
        ]
        embeds = []
        for batch in batches:
            embeds.append(
                self.scorer.get_sentence_embedding(**batch.to(self.scorer.device))
            )
        embeds = torch.vstack(embeds)
        return embeds

    def compute_output_projection(
        self,
        hyp_embeds: torch.Tensor,
        ref_embeds: torch.Tensor,
        src_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scores of the given triplets of vectors.

        Args:
            hyp_embeds (torch.Tensor): Hypothesis embeddings of shape `(N, D)`.
            ref_embeds (torch.Tensor): Reference embeddings of shape `(N, D)`.
            src_embeds (torch.Tensor, optional): Source embeddings of shape `(N, D)`.

        Returns:
            torch.Tensor: N scores.
        """
        return self.scorer.estimate(src_embeds, hyp_embeds, ref_embeds)["score"]

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
        assert source is not None
        hyp_embed = self.compute_sentence_embedding([hypothesis], 1)
        ref_embed = self.compute_sentence_embedding([reference], 1)
        src_embed = self.compute_sentence_embedding([source], 1)
        return self.scorer.estimate(src_embed, hyp_embed, ref_embed)["score"][0].item()

    def pairwise_score(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
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
        scores = torch.zeros((H, R), dtype=torch.float32, device=self.scorer.device)

        assert source is not None
        hyp_embeds = self.compute_sentence_embedding(hypotheses, self.cfg.batch_size)
        ref_embeds = self.compute_sentence_embedding(references, self.cfg.batch_size)
        src_embeds = self.compute_sentence_embedding(
            [source], self.cfg.batch_size
        ).repeat(R, 1)

        for i in range(H):
            scores[i, :] = self.compute_output_projection(
                hyp_embeds[i, :].repeat(R, 1), ref_embeds, src_embeds
            )
        return scores.cpu().float().numpy()
