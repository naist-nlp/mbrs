#!/usr/bin/env python3

import enum
import json
import logging
import os
import sys
from argparse import FileType, Namespace
from dataclasses import dataclass

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

import simple_parsing
import torch
from simple_parsing import ArgumentParser, choice, field, flag
from simple_parsing.wrappers import dataclass_wrapper
from tabulate import tabulate, tabulate_formats
from tqdm import tqdm

from mbrs import registry, timer
from mbrs.decoders import (
    DecoderBase,
    DecoderReferenceBased,
    DecoderReferenceless,
    get_decoder,
)
from mbrs.metrics import Metric, get_metric

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


def parse_args() -> Namespace:
    meta_parser = ArgumentParser(add_help=False)
    meta_parser.add_arguments(CommonArguments, "common")
    known_args, _ = meta_parser.parse_known_args()
    metric_type = get_metric(known_args.common.metric)
    decoder_type = get_decoder(known_args.common.decoder)

    parser = ArgumentParser(add_help=True)
    parser.add_arguments(CommonArguments, "common")
    parser.add_arguments(metric_type.Config, "metric", prefix="metric.")
    parser.add_arguments(decoder_type.Config, "decoder", prefix="decoder.")
    return parser.parse_args()


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

    decoder_type = get_decoder(args.common.decoder)
    decoder: DecoderReferenceBased | DecoderReferenceless = decoder_type(
        args.decoder, metric
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
                ref_lprobs = torch.Tensor(
                    [
                        float(r.strip())
                        for r in reference_lprobs[i * num_refs : (i + 1) * num_refs]
                    ]
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
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
