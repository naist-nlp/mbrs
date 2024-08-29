#!/usr/bin/env python3

import dataclasses
import enum
import json
import logging
import os
import sys
from argparse import FileType, Namespace
from dataclasses import asdict, dataclass, fields, make_dataclass
from typing import Sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

import simple_parsing
import torch
from simple_parsing import choice, field, flag
from simple_parsing.wrappers import dataclass_wrapper
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
from mbrs.metrics import Metric, MetricEnum, get_metric
from mbrs.selectors import Selector, get_selector

simple_parsing.parsing.logger.setLevel(logging.ERROR)
dataclass_wrapper.logger.setLevel(logging.ERROR)


class Format(enum.Enum):
    plain = "plain"
    json = "json"


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
    decoder: str = choice(*registry.get_registry("decoder").keys(), default="mbr")
    # Type of the metric.
    metric: str = choice(*registry.get_registry("metric").keys(), default="bleu")
    # Type of the selector.
    selector: str = choice(*registry.get_registry("selector").keys(), default="nbest")
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


def get_argparser(args: Sequence[str] | None = None) -> ArgumentParser:
    meta_parser = ArgumentParser(add_help=False, add_config_path_arg=True)
    meta_parser.add_arguments(
        CommonArguments, "common", dataclass_wrapper_class=DataclassWrapper
    )
    for _field in meta_parser._wrappers[0].fields:
        _field.required = False
    known_args, _ = meta_parser.parse_known_args(args=args)
    metric_type = get_metric(known_args.common.metric)
    decoder_type = get_decoder(known_args.common.decoder)
    selector_type = get_selector(known_args.common.selector)

    parser = ArgumentParser(add_help=False, add_config_path_arg=True)
    parser.add_arguments(
        CommonArguments, "common", dataclass_wrapper_class=DataclassWrapper
    )
    parser.add_arguments(metric_type.Config, "metric", prefix="metric.")
    parser.add_arguments(decoder_type.Config, "decoder", prefix="decoder.")
    parser.add_arguments(selector_type.Config, "selector", prefix="selector.")
    for _field in parser._wrappers[0].fields:
        _field.required = False

    known_args, _ = parser.parse_known_args(args=args)
    for cfg, m in [
        (known_args.metric, metric_type),
        (known_args.decoder, decoder_type),
        (known_args.selector, selector_type),
    ]:
        for _field in fields(cfg):
            field_name = _field.name
            field_attr = getattr(cfg, field_name)
            if isinstance(field_attr, MetricEnum):
                config_type = get_metric(field_attr).Config
                m.Config = make_dataclass(
                    m.Config.__name__,
                    fields=[
                        (
                            field_name,
                            type(field_attr),
                            dataclasses.field(default=field_attr),
                        ),
                        (
                            field_name + "_config",
                            config_type,
                            dataclasses.field(default_factory=config_type),
                        ),
                    ],
                    bases=(m.Config,),
                )

    parser = ArgumentParser(add_help=True, add_config_path_arg=True)
    parser.add_arguments(
        CommonArguments, "common", dataclass_wrapper_class=DataclassWrapper
    )
    parser.add_arguments(metric_type.Config, "metric", prefix="metric.")
    parser.add_arguments(decoder_type.Config, "decoder", prefix="decoder.")
    parser.add_arguments(selector_type.Config, "selector", prefix="selector.")
    return parser


def format_argparser() -> ArgumentParser:
    parser = get_argparser()
    parser.preprocess_parser()
    return parser


def main(args: Namespace) -> None:
    if not args.common.quiet:
        logger.info(args)

    sources = None
    if args.common.source is not None:
        with open(args.common.source, mode="r") as f:
            sources = f.readlines()

    with open(args.common.hypotheses, mode="r") as f:
        hypotheses = f.readlines()

    references = None
    if args.common.references is not None:
        with open(args.common.references, mode="r") as f:
            references = f.readlines()
    else:
        references = hypotheses

    reference_lprobs = None
    if args.common.reference_lprobs is not None:
        with open(args.common.reference_lprobs, mode="r") as f:
            reference_lprobs = f.readlines()
        assert len(references) == len(reference_lprobs)

    metric_type = get_metric(args.common.metric)
    metric: Metric = metric_type(args.metric)

    selector_type = get_selector(args.common.selector)
    selector: Selector = selector_type(args.selector)

    decoder_type = get_decoder(args.common.decoder)
    decoder: DecoderReferenceBased | DecoderReferenceless = decoder_type(
        args.decoder, metric, selector
    )

    num_cands = args.common.num_candidates
    num_refs = args.common.num_references or num_cands
    num_sents = len(hypotheses) // num_cands
    assert num_sents * num_cands == len(hypotheses)

    def output_results(res: DecoderBase.Output):
        if args.common.format == Format.plain:
            for sent in res.sentence:
                print(sent, file=args.common.output)
        elif args.common.format == Format.json:
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
                    file=args.common.output,
                )

    if isinstance(decoder, DecoderReferenceless):
        for i in tqdm(range(num_sents)):
            src = sources[i].strip()
            hyps = [h.strip() for h in hypotheses[i * num_cands : (i + 1) * num_cands]]
            with timer.measure("total"):
                output = decoder.decode(hyps, src, args.common.nbest)
            output_results(output)
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
                output = decoder.decode(
                    hyps,
                    refs,
                    src,
                    nbest=args.common.nbest,
                    reference_lprobs=ref_lprobs,
                )
            output_results(output)

    if not args.common.quiet:
        statistics = timer.aggregate().result(num_sents)
        table = tabulate(
            statistics,
            headers="keys",
            tablefmt=args.common.report_format,
            floatfmt=f".{args.common.width}f",
        )
        print(table, file=args.common.report)


def cli_main():
    args = get_argparser().parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
