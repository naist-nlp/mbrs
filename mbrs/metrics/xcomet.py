from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import comet.encoders
import torch
from comet import download_model, load_from_checkpoint
from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder
from comet.models import XCOMETMetric
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.deberta_v2 import modeling_deberta_v2

from mbrs import timer, utils

from . import Metric, register


class DeBERTaEncoder(BERTEncoder):
    """DeBERTa encoder.

    Args:
        pretrained_model (str): Pretrained model from hugging face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face
        local_files_only (bool): Whether or not to only look at local files.
    """

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super(Encoder, self).__init__()
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, local_files_only=local_files_only
        )
        if load_pretrained_weights:
            self.model = AutoModel.from_pretrained(pretrained_model)
        else:
            self.model = AutoModel.from_config(
                AutoConfig.from_pretrained(
                    pretrained_model, local_files_only=local_files_only
                ),
            )
        self.model.encoder.output_hidden_states = True

        self.model.encoder.layer = nn.ModuleList(
            [
                modeling_deberta_v2.DebertaV2Layer(
                    AutoConfig.from_pretrained(pretrained_model)
                )
                for _ in range(self.model.config.num_hidden_layers)
            ]
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face
            local_files_only (bool): Whether or not to only look at local files.

        Returns:
            DeBERTaEncoder: DeBERTaEncoder object.
        """
        return DeBERTaEncoder(
            pretrained_model, load_pretrained_weights, local_files_only=local_files_only
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        return {
            "sentemb": model_output.last_hidden_state[:, 0, :],
            "wordemb": model_output.last_hidden_state,
            "all_layers": model_output.hidden_states,
            "attention_mask": attention_mask,
        }


class XCOMETLiteMetric(XCOMETMetric, PyTorchModelHubMixin):
    """xCOMET-Lite model."""

    def __init__(
        self,
        encoder_model="DeBERTa",
        pretrained_model="microsoft/mdeberta-v3-base",
        word_layer=8,
        validation_data=[],
        word_level_training=True,
        hidden_sizes=(3072, 1024),
        load_pretrained_weights=False,
        *args,
        **kwargs,
    ):
        comet.encoders.str2encoder["DeBERTa"] = DeBERTaEncoder
        super().__init__(
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            word_layer=word_layer,
            validation_data=validation_data,
            word_level_training=word_level_training,
            hidden_sizes=hidden_sizes,
            load_pretrained_weights=load_pretrained_weights,
            *args,
            **kwargs,
        )


@register("xcomet")
class MetricXCOMET(Metric):
    """XCOMET metric class.

    Both XCOMET (Guerreiro et al., 2024) and XCOMET-lite (Larionov et al., 2024) are supported.

    Supported models:
        - Unbabel/XCOMET-XL
        - Unbabel/XCOMET-XXL
        - myyycroft/XCOMET-lite
    """

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
        if cfg.model == "myyycroft/XCOMET-lite":
            self.scorer = XCOMETLiteMetric.from_pretrained(cfg.model)
        else:
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
                    references[i : i + self.cfg.batch_size]
                    if references is not None
                    else None,
                    sources[i : i + self.cfg.batch_size]
                    if sources is not None
                    else None,
                )
                .float()
                .cpu()
            )
        return torch.cat(scores).mean().item()
