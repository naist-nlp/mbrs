#!/usr/bin/env python3

import enum
import json
import logging
import os
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass

import simple_parsing
from simple_parsing import ArgumentParser, choice, field, flag
from simple_parsing.wrappers import dataclass_wrapper

from mbrs import registry
from mbrs.metrics import Metric, get_metric
from mbrs.metrics.base import MetricReferenceless

simple_parsing.parsing.logger.setLevel(logging.ERROR)
dataclass_wrapper.logger.setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class Format(enum.Enum):
    plain = "plain"
    json = "json"


@dataclass
class CommonArguments:
    """Common arguments."""

    # Hypotheses file.
    hypotheses: str = field(positional=True)
    # Source file.
    source: str | None = field(default=None, alias=["-s"])
    # References file.
    references: str | None = field(default=None, alias=["-r"])
    # Output format.
    format: Format = choice(Format, default=Format.json)
    # Type of the metric.
    metric: str = choice(*registry.get_registry("metric").keys(), default="bleu")
    # No verbose information and report.
    quiet: bool = flag(default=False)
    # Number of digits for values of float point.
    width: int = field(default=1, alias=["-w"])


def parse_args() -> Namespace:
    meta_parser = ArgumentParser(add_help=False)
    meta_parser.add_arguments(CommonArguments, "common")
    known_args, _ = meta_parser.parse_known_args()
    metric_type = get_metric(known_args.common.metric)

    parser = ArgumentParser(add_help=True)
    parser.add_arguments(CommonArguments, "common")
    parser.add_arguments(metric_type.Config, "metric", prefix="metric.")
    return parser.parse_args()


def main(args: Namespace) -> None:
    with open(args.common.hypotheses, mode="r") as f:
        hypotheses = f.readlines()
    num_sents = len(hypotheses)

    sources = None
    if args.common.source is not None:
        with open(args.common.source, mode="r") as f:
            sources = f.readlines()
        assert num_sents == len(sources)

    references = None
    if args.common.references is not None:
        with open(args.common.references, mode="r") as f:
            references = f.readlines()
        assert num_sents == len(references)

    metric_type = get_metric(args.common.metric)
    metric: Metric = metric_type(args.metric)

    if isinstance(metric, MetricReferenceless):
        assert sources is not None
        corpus_score = metric.corpus_score(hypotheses, sources=sources)
    else:
        assert references is not None
        corpus_score = metric.corpus_score(hypotheses, references, sources)

    score = {
        "name": args.common.metric,
        "score": float(f"{corpus_score:.{args.common.width}f}"),
        "metric_cfg": asdict(args.metric),
    }

    if args.common.format == Format.plain:
        print(score["score"])
    elif args.common.format == Format.json:
        print(json.dumps(score, ensure_ascii=False, indent=2))


def cli_main():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
