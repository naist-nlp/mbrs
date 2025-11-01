#!/usr/bin/env python3

import dataclasses
import enum
import json
import logging
import os
import sys
import typing
from argparse import FileType, Namespace
from dataclasses import asdict, dataclass, fields, make_dataclass
from typing import Sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

import torch
from simple_parsing import choice, field, flag
from tabulate import tabulate, tabulate_formats
from tqdm import tqdm

from mbrs import registry, timer
from mbrs.args import ArgumentParser, DataclassWrapper
from mbrs.decoders import (
    DecoderBase,
    DecoderReferenceBased,
    DecoderReferenceless,
    get_decoder,
)
from mbrs.metrics import Metric, MetricEnum, MetricReferenceless, get_metric
from mbrs.selectors import Selector, get_selector


class Format(enum.Enum):
    plain = "plain"
    json = "json"

    def output_results(self, res: DecoderBase.Output, output: typing.TextIO):
        match self:
            case self.plain:
                for sent in res.sentence:
                    print(sent, file=output)
            case self.json:
                for i, (sent, idx, score) in enumerate(
                    zip(res.sentence, res.idx, res.score)
                ):
                    print(
                        json.dumps(
                            {
                                "rank": i,
                                "sentence": sent,
                                "selected_idx": idx,
                                "expected_score": score,
                                **{
                                    k: v
                                    for k, v in asdict(res).items()
                                    if k not in {"sentence", "idx", "score"}
                                },
                            },
                            ensure_ascii=False,
                        ),
                        file=output,
                    )


@dataclass
class CommonArguments:
    """Common arguments."""

    # Hypotheses file.
    hypotheses: str = field(positional=True)
    # Number of candidates.
    num_candidates: int = field(alias=["-n"])
    # Source file.
    source: str | None = field(default=None, alias=["-s"])
    # References file.
    references: str | None = field(default=None, alias=["-r"])
    # References log-probabilities file.
    reference_lprobs: str | None = field(default=None)
    # Output file.
    output: FileType("w", encoding="utf-8") = field(default="-", alias=["-o"])
    # Output format.
    format: Format = choice(Format, default=Format.plain)
    # Number of references for each sentence.
    num_references: int | None = field(default=None)
    # Type of the decoder.
    decoder: str = field(
        default="mbr",
        metadata={
            "choices": registry.get_registry(
                DecoderReferenceBased | DecoderReferenceless
            )
        },
    )
    # Type of the metric.
    metric: str = field(
        default="bleu",
        metadata={"choices": registry.get_registry(Metric | MetricReferenceless)},
    )
    # Type of the selector.
    selector: str = field(
        default="nbest", metadata={"choices": registry.get_registry(Selector)}
    )
    # Return the n-best hypotheses.
    nbest: int = field(default=1)
    # No verbose information and report.
    quiet: bool = flag(default=False)
    # Report file.
    report: FileType("w") = field(default="-")
    # Report runtime statistics with the given format.
    report_format: str = choice(*tabulate_formats, default="rounded_outline")
    # Number of digits for values of float point.
    width: int = field(default=1, alias=["-w"])

    def get_decoder_type(self) -> type[DecoderReferenceBased | DecoderReferenceless]:
        return get_decoder(self.decoder)

    def get_metric_type(self) -> type[Metric | MetricReferenceless]:
        return get_metric(self.metric)

    def get_selector_type(self) -> type[Selector]:
        return get_selector(self.selector)

    def output_results(self, res: DecoderBase.Output):
        return self.format.output_results(res, self.output)


def get_argparser(args: Sequence[str] | None = None) -> ArgumentParser:
    """Gets an argument parser.

    Args:
        args (Sequence[str], optional): Command-line arguments. If not specified,
          `sys.argv` will be parsed.

    Returns:
        ArgumentParser: An argument parser.

    Todo:
        - Improve the logic or use other libraries.
    """
    def build_parser(
        known_args: Namespace | None = None, partial: bool = False
    ) -> ArgumentParser:
        parser = ArgumentParser(add_help=not partial, add_config_path_arg=True)
        parser.add_arguments(
            CommonArguments, "common", dataclass_wrapper_class=DataclassWrapper
        )
        if known_args is not None:
            common_args: CommonArguments = known_args.common
            for prefix, m in [
                ("metric", common_args.get_metric_type()),
                ("decoder", common_args.get_decoder_type()),
                ("selector", common_args.get_selector_type()),
            ]:
                parser.add_arguments(m.Config, prefix, prefix=f"{prefix}.")
        if partial:
            for f in parser._wrappers[0].fields:
                f.required = False
        return parser

    def parse_partial(common_args: CommonArguments, known_args: Namespace) -> Namespace:
        for prefix, m in [
            ("metric", common_args.get_metric_type()),
            ("decoder", common_args.get_decoder_type()),
            ("selector", common_args.get_selector_type()),
        ]:
            field_types = typing.get_type_hints(m.Config)
            for f in fields(m.Config):
                ftype = field_types[f.name]
                if isinstance(ftype, type) and issubclass(ftype, MetricEnum):
                    if (cfg_dict := getattr(known_args, prefix, None)) is not None:
                        if dataclasses.is_dataclass(cfg_dict):
                            cfg_dict = asdict(cfg_dict)
                        config_type = get_metric(cfg_dict[f.name]).Config
                    else:
                        config_type = get_metric(f.default).Config
                    m.Config = make_dataclass(
                        m.Config.__name__,
                        fields=[
                            (f.name, ftype, dataclasses.field(default=f.default)),
                            (
                                f.name + "_config",
                                config_type,
                                dataclasses.field(default_factory=config_type),
                            ),
                        ],
                        bases=(m.Config,),
                    )
        (known_args, _) = build_parser(
            known_args=known_args, partial=True
        ).parse_known_args(args=args)
        return known_args

    # 1. Determines component classes.
    known_args, _ = build_parser(partial=True).parse_known_args(args=args)
    common_args: CommonArguments = known_args.common
    # 2. Parses default values.
    known_args = parse_partial(common_args, known_args)
    # 3. Prepares configuration classes via partially parsed settings.
    known_args = parse_partial(common_args, known_args)
    if known_args.config_path is not None:
        known_args = parse_partial(common_args, known_args)
    # 4. Builds a full configuration parser.
    return build_parser(known_args=known_args)


def format_argparser() -> ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser


def main(args: Namespace) -> None:
    if not args.common.quiet:
        logger.info(args)

    common_args: CommonArguments = args.common

    sources = None
    if common_args.source is not None:
        with open(common_args.source, mode="r") as f:
            sources = f.readlines()

    with open(common_args.hypotheses, mode="r") as f:
        hypotheses = f.readlines()

    references = None
    if common_args.references is not None:
        with open(common_args.references, mode="r") as f:
            references = f.readlines()
    else:
        references = hypotheses

    reference_lprobs = None
    if common_args.reference_lprobs is not None:
        with open(common_args.reference_lprobs, mode="r") as f:
            reference_lprobs = f.readlines()
        assert len(references) == len(reference_lprobs)

    metric = common_args.get_metric_type()(args.metric)
    selector = common_args.get_selector_type()(args.selector)
    decoder = common_args.get_decoder_type()(args.decoder, metric, selector)

    num_cands = common_args.num_candidates
    num_refs = common_args.num_references or num_cands
    num_sents = len(hypotheses) // num_cands
    assert num_sents * num_cands == len(hypotheses)

    if isinstance(decoder, DecoderReferenceless):
        for i in tqdm(range(num_sents)):
            src = sources[i].strip()
            hyps = [h.strip() for h in hypotheses[i * num_cands : (i + 1) * num_cands]]
            with timer.measure("total"):
                res = decoder.decode(hyps, src, common_args.nbest)
            common_args.output_results(res)
    else:
        for i in tqdm(range(num_sents)):
            src = sources[i].strip() if sources is not None else None
            hyps = [h.strip() for h in hypotheses[i * num_cands : (i + 1) * num_cands]]
            refs = [r.strip() for r in references[i * num_refs : (i + 1) * num_refs]]
            ref_lprobs = None
            if reference_lprobs is not None:
                ref_lprobs = [
                    float(r.strip())
                    for r in reference_lprobs[i * num_refs : (i + 1) * num_refs]
                ]

                # Deduplicate the same elements for the model-based estimation.
                # Note that we regard pairs of `(reference, log-prob of reference)` that are
                # equal as the same samples.
                # We use `dict.fromkeys()` instead of `set()`; thus, the order is kept.
                uniq_refs, uniq_ref_lprobs = tuple(
                    zip(*dict.fromkeys(zip(refs, ref_lprobs)).keys())
                )

                refs = list(uniq_refs)
                ref_lprobs = torch.tensor(
                    list(uniq_ref_lprobs), dtype=torch.float32, device=metric.device
                )

            with timer.measure("total"):
                res = decoder.decode(
                    hyps,
                    refs,
                    src,
                    nbest=common_args.nbest,
                    reference_lprobs=ref_lprobs,
                )
            common_args.output_results(res)

    if not common_args.quiet:
        statistics = timer.aggregate().result(num_sents)
        table = tabulate(
            statistics,
            headers="keys",
            tablefmt=common_args.report_format,
            floatfmt=f".{common_args.width}f",
        )
        print(table, file=common_args.report)


def cli_main():
    args = get_argparser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
