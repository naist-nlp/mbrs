from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from comet import download_model, load_from_checkpoint

from mbrs.metrics.base import MetricNeural

from . import MetricNeural, register


@register("comet")
class MetricCOMET(MetricNeural):
    """COMET metric class."""

    @dataclass
    class Config(MetricNeural.Config):
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

    @dataclass
    class IR(MetricNeural.IR):
        """Intermediate representations."""

        hyp: torch.Tensor
        ref: torch.Tensor
        src: torch.Tensor

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.scorer.device

    def embed_sentences(self, sentences: list[str], batch_size: int) -> torch.Tensor:
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

    def encode(
        self, hypotheses: list[str], references: list[str], source: Optional[str] = None
    ) -> IR:
        """Encode the given sentences into their intermediate representations.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.

        Returns:
            MetricCOMET.IR: Intermediate representations.
        """
        assert source is not None
        hyp_embed = self.embed_sentences(hypotheses, self.cfg.batch_size)
        ref_embed = (
            self.embed_sentences(references, self.cfg.batch_size)
            if references != hypotheses
            else hyp_embed
        )
        src_embed = self.embed_sentences([source], self.cfg.batch_size)
        return self.IR(hyp_embed, ref_embed, src_embed)

    def out_proj(self, ir: IR) -> torch.Tensor:
        """Forward the output projection layer.

        Args:
            ir (MetricCOMET.IR): Intermediate representations
              computed by the `encode` method.

        Returns:
            torch.Tensor: H x R score matrix, where
              - H: the number of hypotheses
              - R: the number of references
        """
        H, D = ir.hyp.size()
        R, _ = ir.ref.size()
        src = ir.src.repeat(R, 1)
        scores = []
        for i in range(H):
            scores.append(
                self.scorer.estimate(src, ir.hyp[i, :].repeat(R, 1), ir.ref)["score"]
            )
        return torch.vstack(scores)
