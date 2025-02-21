from __future__ import annotations

import enum
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import bert_score
import bert_score.utils
import torch
import transformers
from bert_score import BERTScorer
from simple_parsing.helpers.fields import choice
from torch import Tensor
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.models.roberta import RobertaTokenizer
from transformers.tokenization_utils import (
    BatchEncoding,
    EncodedInput,
    PreTrainedTokenizerBase,
)

from mbrs.metrics.base import MetricCacheable

from . import Metric, register

transformers.logging.set_verbosity_error()


class BERTScoreScoreType(int, enum.Enum):
    precision = 0
    recall = 1
    f1 = 2


@register("bertscore")
class MetricBERTScore(MetricCacheable):
    """BERTScore metric class."""

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
        idf: bool = False
        idf_sents: Optional[list[str]] = None
        lang: Optional[str] = None
        rescale_with_baseline: bool = False
        baseline_path: Optional[str] = None
        use_fast_tokenizer: bool = False
        fp16: bool = False
        bf16: bool = False
        cpu: bool = False

    @dataclass
    class Cache(MetricCacheable.Cache):
        """Intermediate representations of sentences.

        - embeddings (list[Tensor]): A list of token embeddings of shape `(T, D)`,
            where `T` is the length of sequence, and `D` is a size of the embedding.
        - idf_weights (list[Tensor]): A list of IDF weights of shape `(T,)`.
        """

        embeddings: list[Tensor]
        idf_weights: list[Tensor]

        def __len__(self) -> int:
            """Return the length of cache."""
            return len(self.embeddings)

        def __getitem__(
            self, key: int | Sequence[int] | slice | Tensor
        ) -> MetricBERTScore.Cache:
            """Get the items."""
            if isinstance(key, Tensor):
                dtype = key.dtype
                key = key.tolist()
                if dtype == torch.bool:
                    return type(self)(
                        [self.embeddings[k] for k in key if k],
                        [self.idf_weights[k] for k in key if k],
                    )
                return type(self)(
                    [self.embeddings[k] for k in key],
                    [self.idf_weights[k] for k in key],
                )
            elif isinstance(key, Sequence):
                return type(self)(
                    [self.embeddings[k] for k in key],
                    [self.idf_weights[k] for k in key],
                )
            elif isinstance(key, slice):
                return type(self)(self.embeddings[key], self.idf_weights[key])
            else:
                return type(self)([self.embeddings[key]], [self.idf_weights[key]])

        def repeat(self, n: int) -> MetricBERTScore.Cache:
            """Repeat the representations by n times.

            Args:
                n (int): The number of repetition.

            Returns:
                Cache: The repeated cache.
            """
            return type(self)(self.embeddings * n, self.idf_weights * n)

    cfg: MetricBERTScore.Config

    def __init__(self, cfg: MetricBERTScore.Config):
        super().__init__(cfg)
        self.scorer: BERTScorer = BERTScorer(
            model_type=cfg.model_type,
            num_layers=cfg.num_layers,
            batch_size=cfg.batch_size,
            nthreads=cfg.nthreads,
            all_layers=False,
            idf=cfg.idf,
            idf_sents=cfg.idf_sents,
            device="cpu" if cfg.cpu else None,
            lang=cfg.lang,
            rescale_with_baseline=cfg.rescale_with_baseline,
            baseline_path=cfg.baseline_path,
            use_fast_tokenizer=cfg.use_fast_tokenizer,
        )
        self.tokenizer: PreTrainedTokenizerBase = self.scorer._tokenizer
        self.model = self.scorer._model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if not cfg.cpu and torch.cuda.is_available():
            if cfg.fp16:
                self.model = self.model.half()
            elif cfg.bf16:
                self.model = self.model.bfloat16()
            self.model = self.model.cuda()

        self.idf_dict: dict[int, float]
        if cfg.idf and self.scorer._idf_dict is not None:
            self.idf_dict = self.scorer._idf_dict
        else:
            self.idf_dict = defaultdict(lambda: 1.0)
            if (
                sep_token_id := getattr(self.tokenizer, "sep_token_id", None)
            ) is not None:
                self.idf_dict[sep_token_id] = 0.0
            if (
                cls_token_id := getattr(self.tokenizer, "cls_token_id", None)
            ) is not None:
                self.idf_dict[cls_token_id] = 0.0

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return self.model.device

    @property
    def embed_dim(self) -> int:
        """Return the size of embedding dimension."""
        return self.model.config.hidden_size

    def _tokenize(self, sentence: str) -> list[int]:
        """Tokenize a sentence and encode it to the token IDs.

        Args:
            sentence (str): An input sentence.

        Returns:
            list[int]: The token IDs.
        """
        tokenizer_kwargs = {}
        if isinstance(self.tokenizer, (GPT2Tokenizer, RobertaTokenizer)):
            tokenizer_kwargs["add_prefix_space"] = True

        return self.tokenizer.encode(
            sentence,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            **tokenizer_kwargs,
        )

    def _collate(self, batch_ids: list[EncodedInput]) -> BatchEncoding:
        """Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs (list[EncodedInputPair]): List of tokenized input ids.

        Returns:
            BatchEncoding: A mini-batch.
        """
        batch = {}
        for ids in batch_ids:
            example = self.tokenizer.prepare_for_model(
                ids,
                add_special_tokens=False,
                padding=False,
                pad_to_multiple_of=None,
                return_attention_mask=False,
                return_tensors=None,
            )

            for key, value in example.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        return self.tokenizer.pad(
            batch,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

    def encode(self, sentences: list[str]) -> MetricBERTScore.Cache:
        """Encode the given sentences into their intermediate representations.

        Args:
            sentences (list[str]): Input sentences.

        Returns:
            Tensor: Intermediate representations of shape `(N, D)` where `N` is the
              number of hypotheses and `D` is a size of the embedding dimension.
        """
        sequences = [self._tokenize(sentence) for sentence in sentences]
        embeddings = []
        for i in range(0, len(sentences), self.cfg.batch_size):
            batch = self._collate(sequences[i : i + self.cfg.batch_size])
            attention_mask = batch.attention_mask.bool()
            embs = self.model(**batch.to(self.device))[0].cpu()
            for j in range(len(embs)):
                embeddings.append(embs[j, attention_mask[j]])
        idf_weights = [
            torch.Tensor([self.idf_dict.get(token, 1.0) for token in seq])
            for seq in sequences
        ]

        return self.Cache(embeddings, idf_weights)

    def _choose_output_score(self, triplet: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Choose the output score from the triplet of precision, recall, and f1 scores.

        Args:
            triplet (tuple[Tensor, Tensor, Tensor]): A triplet of precision, recall, and f1 scores.

        Returns:
            Tensor: Output score.
        """
        return triplet[self.cfg.score_type]

    def pad_sequence(self, tensors: list[Tensor]) -> Tensor:
        match tensors[0].dtype:
            case torch.bool:
                padding_value = False
            case torch.float32:
                padding_value = torch.finfo(torch.float32).eps
            case torch.float16:
                padding_value = torch.finfo(torch.float16).eps
            case torch.bfloat16:
                padding_value = torch.finfo(torch.bfloat16).eps
            case _:
                padding_value = 0.0

        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=padding_value
        ).to(self.device)

    def out_proj(
        self,
        hypotheses_ir: Cache,
        references_ir: Cache,
        sources_ir: Optional[Cache] = None,
    ) -> Tensor:
        """Forward the output projection layer.

        Args:
            hypotheses_ir (Cache): N intermediate representations of hypotheses.
            references_ir (Cache): N intermediate representations of references.
            sources_ir (Cache, optional): N intermediate representations of sources.

        Returns:
            Tensor: N scores.
        """

        hypotheses_embeddings = self.pad_sequence(hypotheses_ir.embeddings)
        references_embeddings = self.pad_sequence(references_ir.embeddings)
        hypotheses_token_masks = self.pad_sequence(
            [torch.BoolTensor([True] * len(emb)) for emb in hypotheses_ir.embeddings]
        )
        references_token_masks = self.pad_sequence(
            [torch.BoolTensor([True] * len(emb)) for emb in references_ir.embeddings]
        )
        hypotheses_idf_weights = self.pad_sequence(hypotheses_ir.idf_weights)
        references_idf_weights = self.pad_sequence(references_ir.idf_weights)

        scores = self._choose_output_score(
            bert_score.utils.greedy_cos_idf(
                references_embeddings,
                references_token_masks,
                references_idf_weights,
                hypotheses_embeddings,
                hypotheses_token_masks,
                hypotheses_idf_weights,
                all_layers=False,
            )
        )
        if self.cfg.rescale_with_baseline:
            scores = (scores - self.scorer.baseline_vals) / (
                1 - self.scorer.baseline_vals
            )
        return scores.view(len(hypotheses_embeddings))

    def scores(self, hypotheses: list[str], references: list[str], *_, **__) -> Tensor:
        """Calculate the scores of the given hypothesis.

        Args:
            hypotheses (list[str]): N hypotheses.
            references (list[str]): N references.

        Returns:
            Tensor: The N scores of the given hypotheses.
        """
        return super().scores(hypotheses, references)

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
        return super().pairwise_scores(hypotheses, references)

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
