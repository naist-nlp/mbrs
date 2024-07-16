from __future__ import annotations

from dataclasses import dataclass

import torch
from comet import download_model, load_from_checkpoint
from torch import Tensor

from . import MetricAggregatable, MetricCacheable, register


@register("comet")
class MetricCOMET(MetricCacheable, MetricAggregatable):
    """COMET metric class."""

    @dataclass
    class Config(MetricCacheable.Config):
        """COMET metric configuration.

        - model (str): Model name or path.
        - batch_size (int): Batch size.
        - fp16 (bool): Use float16 for the forward computation.
        - bf16 (bool): Use bfloat16 for the forward computation.
        - cpu (bool): Use CPU for the forward computation.
        """

        model: str = "Unbabel/wmt22-comet-da"
        batch_size: int = 64
        fp16: bool = False
        bf16: bool = False
        cpu: bool = False

    def __init__(self, cfg: MetricCOMET.Config):
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
    def embed_dim(self) -> int:
        """Return the size of embedding dimension."""
        return self.scorer.encoder.output_units

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.scorer.device

    def encode(self, sentences: list[str]) -> torch.Tensor:
        """Compute sentence embedding vectors of the given sentences.

        Args:
            sentences (list[str]): Input sentences.

        Returns:
            torch.Tensor: Sentence embeddings of shape `(N, D)`, where
              - N: the number of sentences
              - D: size of the embedding dimmension
        """
        batches = [
            self.scorer.encoder.prepare_sample(sentences[i : i + self.cfg.batch_size])
            for i in range(0, len(sentences), self.cfg.batch_size)
        ]
        embeds = []
        for batch in batches:
            emb = self.scorer.get_sentence_embedding(**batch.to(self.scorer.device))
            if self.scorer.device.type != "cpu":
                if self.cfg.fp16:
                    emb = emb.half()
                elif self.cfg.bf16:
                    emb = emb.bfloat16()
                else:
                    emb = emb.float()
            embeds.append(emb)
        embeds = torch.vstack(embeds)
        return embeds

    def out_proj(
        self, hypotheses_ir: Tensor, references_ir: Tensor, sources_ir: Tensor
    ) -> Tensor:
        """Forward the output projection layer.

        Args:
            hypotheses_ir (Tensor): Intermediate representations of hypotheses.
            references_ir (Tensor): Intermediate representations of references.
            sources_ir (Tensor): Intermediate representations of sources.

        Returns:
            Tensor: N scores.
        """
        return self.scorer.estimate(sources_ir, hypotheses_ir, references_ir)["score"]

    def corpus_score(
        self, hypotheses: list[str], references: list[str], sources: list[str]
    ) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (list[str]): Sources.

        Returns:
            float: The corpus score.
        """
        scores = []
        for i in range(0, len(hypotheses), self.cfg.batch_size):
            scores.append(
                self.scores(
                    hypotheses[i : i + self.cfg.batch_size],
                    references[i : i + self.cfg.batch_size],
                    sources[i : i + self.cfg.batch_size],
                )
                .float()
                .cpu()
            )
        return torch.cat(scores).mean().item()
