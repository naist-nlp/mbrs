from __future__ import annotations

import copy
import enum
import itertools
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import transformers
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.models.mt5.modeling_mt5 import (
    __HEAD_MASK_WARNING_MSG,
    MT5Config,
    MT5PreTrainedModel,
    MT5Stack,
)
from transformers.tokenization_utils import BatchEncoding, EncodedInput

from mbrs import timer

from . import Metric, register

transformers.logging.set_verbosity_error()


@dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None


class MT5ForRegression(MT5PreTrainedModel):
    """MT5 model for regression.

    This implementation is copied from https://github.com/google-research/metricx
    """

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor] | MT5ForRegressionOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask,
        # decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Create 1 step of dummy input for the decoder.
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)
        if torch.cuda.is_available():
            decoder_input_ids = decoder_input_ids.to(torch.device("cuda"))

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See
            # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        # 250089 = <extra_id_10>
        predictions = lm_logits[:, 0, 250089]

        # Clip to 0 to 25
        predictions = torch.clamp(predictions, 0, 25)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            # move labels to correct device to enable PP
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return MT5ForRegressionOutput(loss=loss, predictions=predictions)


@register("metricx")
class MetricMetricX(Metric):
    """MetricX metric class.

    References:
    - MetricX-23: https://aclanthology.org/2023.wmt-1.63
    - MetricX-24: https://aclanthology.org/2024.wmt-1.35

    Available checkpoints:

    - google/metricx-24-hybrid-xxl-v2p6
    - google/metricx-24-hybrid-xl-v2p6
    - google/metricx-24-hybrid-large-v2p6
    - google/metricx-23-xxl-v2p0
    - google/metricx-23-xl-v2p0
    - google/metricx-23-large-v2p0
    - google/metricx-23-qe-xxl-v2p0
    - google/metricx-23-qe-xl-v2p0
    - google/metricx-23-qe-large-v2p0
    """

    HIGHER_IS_BETTER: bool = False

    scorer: MT5ForRegression

    @dataclass
    class Config(Metric.Config):
        """MetricX metric configuration.

        - model (str): Model name or path.
        - batch_size (int): Batch size.
        - fp16 (bool): Use float16 for the forward computation.
        - bf16 (bool): Use bfloat16 for the forward computation.
        - cpu (bool): Use CPU for the forward computation.
        """

        model: str = "google/metricx-24-hybrid-xxl-v2p6"
        batch_size: int = 8
        fp16: bool = False
        bf16: bool = False
        cpu: bool = False

    class MetricXVersion(str, enum.Enum):
        metricx_24 = "metricx_24"
        metricx_23 = "metricx_23"

    METRICX_VERSION_MAP = {
        "google/metricx-24-hybrid-xxl-v2p6": MetricXVersion.metricx_24,
        "google/metricx-24-hybrid-xl-v2p6": MetricXVersion.metricx_24,
        "google/metricx-24-hybrid-large-v2p6": MetricXVersion.metricx_24,
        "google/metricx-23-xxl-v2p0": MetricXVersion.metricx_23,
        "google/metricx-23-xl-v2p0": MetricXVersion.metricx_23,
        "google/metricx-23-large-v2p0": MetricXVersion.metricx_23,
        "google/metricx-23-qe-xxl-v2p0": MetricXVersion.metricx_23,
        "google/metricx-23-qe-xl-v2p0": MetricXVersion.metricx_23,
        "google/metricx-23-qe-large-v2p0": MetricXVersion.metricx_23,
    }
    METRICX_INPUT_LENGTH_MAP = {
        MetricXVersion.metricx_24: 1536,
        MetricXVersion.metricx_23: 1024,
    }
    METRICX23_QE_MODELS = {
        "google/metricx-23-qe-xxl-v2p0",
        "google/metricx-23-qe-xl-v2p0",
        "google/metricx-23-qe-large-v2p0",
    }

    @dataclass
    class InputPrefix:
        hypothesis: str
        reference: str
        source: str

    METRICX_INPUT_PREFIX_MAP = {
        MetricXVersion.metricx_24: InputPrefix(
            " candidate: ", " reference: ", "source: "
        ),
        MetricXVersion.metricx_23: InputPrefix(
            "candidate: ", " reference: ", " source: "
        ),
    }

    def __init__(self, cfg: MetricMetricX.Config):
        super().__init__(cfg)
        self.scorer = MT5ForRegression.from_pretrained(cfg.model)
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/mt5-xl", legacy=False, use_fast=False
        )
        self.metricx_version = self.METRICX_VERSION_MAP[cfg.model]
        self.max_length = self.METRICX_INPUT_LENGTH_MAP[self.metricx_version]
        self.input_prefix = self.METRICX_INPUT_PREFIX_MAP[self.metricx_version]

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

    def _encode_hypothesis(self, hypothesis: str) -> list[int]:
        """Encode a hypothesis.

        Args:
            hypothesis (str): A hypothesis.

        Returns:
            list[int]: Token IDs of a hypothesis.
        """
        return self.tokenizer.encode(
            self.input_prefix.hypothesis + hypothesis, add_special_tokens=False
        )

    def _encode_reference(self, reference: str) -> list[int]:
        """Encode a reference.

        Args:
            reference (str): A reference.

        Returns:
            list[int]: Token IDs of a reference.
        """
        return self.tokenizer.encode(
            self.input_prefix.reference + reference, add_special_tokens=False
        )

    def _encode_source(self, source: str) -> list[int]:
        """Encode a source.

        Args:
            source (str): A source.

        Returns:
            list[int]: Token IDs of a source.
        """
        return self.tokenizer.encode(
            self.input_prefix.source + source, add_special_tokens=False
        )

    def _concatenate_inputs(
        self,
        hypothesis_ids: list[int],
        reference_ids: Optional[list[int]] = None,
        source_ids: Optional[list[int]] = None,
    ) -> list[int]:
        """Prepare a model input for MetricX.

        Args:
            hypothesis_ids (str): Hypothesis token IDs.
            reference_ids (str, optional): Reference token IDs.
            source_ids (str, optional): Source token IDs.

        Returns:
            str: Input string.
        """
        input_ids: list[int] = []
        match self.metricx_version:
            case self.MetricXVersion.metricx_24:
                if source_ids is None:
                    raise ValueError("MetricX-24 requires the source text.")
                input_ids += source_ids + hypothesis_ids
                if reference_ids is not None:
                    input_ids += reference_ids
            case self.MetricXVersion.metricx_23:
                input_ids += hypothesis_ids
                if self.cfg.model in self.METRICX23_QE_MODELS:
                    if source_ids is None:
                        raise ValueError("MetricX-23-QE requires the source text.")
                    input_ids += source_ids
                else:
                    if reference_ids is None:
                        raise ValueError("MetricX-23 requires the reference text.")
                    input_ids += reference_ids
        return input_ids

    def _collate(self, batch_ids: list[EncodedInput]) -> BatchEncoding:
        """Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids (list[EncodedInput]): List of tokenized input ids.
        """

        batch = {}
        for input_ids in batch_ids:
            example = self.tokenizer.prepare_for_model(
                input_ids,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                pad_to_multiple_of=None,
                return_attention_mask=False,
                return_tensors=None,
            )

            for key, value in example.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        return batch

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

        batch = self._collate(
            [
                self._concatenate_inputs(
                    self._encode_hypothesis(hypothesis),
                    self._encode_reference(reference)
                    if reference is not None
                    else None,
                    self._encode_source(source) if source is not None else None,
                )
            ]
        ).to(self.device)
        model_output = self.scorer(**batch)
        return model_output.predictions.flatten().tolist()[0]

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
        examples: list[list[int]] = []
        for i, hyp in enumerate(hypotheses):
            examples.append(
                self._concatenate_inputs(
                    self._encode_hypothesis(hyp),
                    self._encode_reference(references[i])
                    if references is not None
                    else None,
                    self._encode_source(sources[i]) if sources is not None else None,
                )
            )

        scores = []
        with timer.measure("score") as t:
            t.set_delta_ncalls(len(examples))
            for i in range(0, len(examples), self.cfg.batch_size):
                batch = self._collate(examples[i : i + self.cfg.batch_size]).to(
                    self.device
                )
                model_output = self.scorer(**batch)
                scores.append(model_output.predictions.flatten())
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
        scores = []
        hypotheses_ids = [self._encode_hypothesis(hyp) for hyp in hypotheses]
        references_ids = [self._encode_reference(ref) for ref in references]
        source_ids = self._encode_source(source) if source is not None else None
        pairwise_iter = itertools.product(hypotheses_ids, references_ids)

        while batch := list(itertools.islice(pairwise_iter, self.cfg.batch_size)):
            with timer.measure("score") as t:
                t.set_delta_ncalls(len(batch))
                batch = self._collate(
                    [
                        self._concatenate_inputs(hyp_ids, ref_ids, source_ids)
                        for hyp_ids, ref_ids in batch
                    ]
                ).to(self.device)
                model_output = self.scorer(**batch)
                scores.append(model_output.predictions.flatten())
        return torch.cat(scores).view(len(hypotheses), len(references))

    def corpus_score(
        self,
        hypotheses: list[str],
        references_lists: Optional[list[list[str]]] = None,
        sources: Optional[list[str]] = None,
    ) -> float:
        """Calculate the corpus-level score.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[list[str]], optional): Lists of references.
            sources (list[str], optional): Sources.

        Returns:
            float: The corpus score.
        """
        scores: list[Tensor] = []
        if references_lists is None:
            if sources is None:
                raise ValueError(
                    "`sources` must be given when `references_lists` is None."
                )

            for i in range(0, len(hypotheses), self.cfg.batch_size):
                scores.append(
                    self.scores(
                        hypotheses[i : i + self.cfg.batch_size],
                        None,
                        sources[i : i + self.cfg.batch_size],
                    )
                    .float()
                    .cpu()
                )
        else:
            for references in references_lists:
                for i in range(0, len(hypotheses), self.cfg.batch_size):
                    scores.append(
                        self.scores(
                            hypotheses[i : i + self.cfg.batch_size],
                            references[i : i + self.cfg.batch_size],
                            sources[i : i + self.cfg.batch_size]
                            if sources is not None
                            else None,
                        )
                        .float()
                        .cpu()
                    )
        return torch.cat(scores).mean().item()
