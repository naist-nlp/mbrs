from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from comet import download_model, load_from_checkpoint
from torch import Tensor

from mbrs.modules.kmeans import Kmeans

from . import MetricAggregatableCache, register


@register("comet")
class MetricCOMET(MetricAggregatableCache):
    """COMET metric class."""

    @dataclass
    class Config(MetricAggregatableCache.Config):
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

    @dataclass
    class Cache(MetricAggregatableCache.Cache):
        """Intermediate representations of sentences.

        - embeddings (Tensor): Sentence embeddings of shape `(N, D)`, where `N`
            is the number of sentences and `D` is a size of the embedding
            dimension.
        """

        embeddings: Tensor

        def __len__(self) -> int:
            """Return the length of cache."""
            return len(self.embeddings)

        def __getitem__(
            self, key: int | Sequence[int] | slice | Tensor
        ) -> MetricCOMET.Cache:
            """Get the items."""
            return type(self)(self.embeddings[key])

        def repeat(self, n: int) -> MetricCOMET.Cache:
            """Repeat the representations by n times.

            Args:
                n (int): The number of repetition.

            Returns:
                Cache: The repeated cache.
            """
            return type(self)(self.embeddings.repeat((n, 1)))

        def aggregate(
            self, reference_lprobs: Optional[Tensor] = None
        ) -> MetricCOMET.Cache:
            """Aggregate the cached representations.

            Args:
                reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
                  The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

            Returns:
                Cache: An aggregated representation.
            """
            if reference_lprobs is not None:
                aggregated_embedding = (
                    self.embeddings
                    * reference_lprobs.to(self.embeddings)
                    .softmax(dim=-1, dtype=torch.float32)
                    .to(self.embeddings)[:, None]
                ).sum(dim=0, keepdim=True)
            else:
                aggregated_embedding = self.embeddings.mean(dim=0, keepdim=True)
            return type(self)(aggregated_embedding)

        def cluster(
            self, kmeans: Kmeans
        ) -> tuple[MetricAggregatableCache.Cache, Tensor]:
            """Cluster the cached representations.

            Args:
                kmeans (Kmeans): k-means class to perform clustering.

            Returns:
                tuple[Cache, Tensor]:
                  - Cache: Centroid representations.
                  - Tensor: N assigned IDs.
            """
            centroids, assigns = kmeans.train(self.embeddings)
            return type(self)(centroids), assigns

    @property
    def embed_dim(self) -> int:
        """Return the size of embedding dimension."""
        return self.scorer.encoder.output_units

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.scorer.device

    def encode(self, sentences: list[str]) -> Cache:
        """Encode the given sentences into their intermediate representations.

        Args:
            sentences (list[str]): Input sentences.

        Returns:
            MetricCOMET.Cache: Intermediate representations.
        """
        batches = [
            self.scorer.encoder.prepare_sample(sentences[i : i + self.cfg.batch_size])
            for i in range(0, len(sentences), self.cfg.batch_size)
        ]
        embeddings = []
        for batch in batches:
            emb = self.scorer.get_sentence_embedding(**batch.to(self.scorer.device))
            if self.scorer.device.type != "cpu":
                if self.cfg.fp16:
                    emb = emb.half()
                elif self.cfg.bf16:
                    emb = emb.bfloat16()
                else:
                    emb = emb.float()
            embeddings.append(emb)
        return self.Cache(torch.vstack(embeddings))

    def out_proj(
        self, hypotheses_ir: Cache, references_ir: Cache, sources_ir: Cache
    ) -> Tensor:
        """Forward the output projection layer.

        Args:
            hypotheses_ir (Cache): N intermediate representations of hypotheses.
            references_ir (Cache): N intermediate representations of references.
            sources_ir (Cache, optional): N intermediate representations of sources.

        Returns:
            Tensor: N scores.
        """
        return self.scorer.estimate(
            sources_ir.embeddings, hypotheses_ir.embeddings, references_ir.embeddings
        )["score"]

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
