<p align="center">
<img src="https://raw.githubusercontent.com/naist-nlp/mbrs/main/docs/icon.svg" height="240px">
</p>

<p align="center">
<i>mbrs</i> is a library for minimum Bayes risk (MBR) decoding.
</p>

<p align="center">
<a href="https://pypi.org/project/mbrs"><img alt="PyPi" src="https://img.shields.io/pypi/v/mbrs"></a>
<a href="https://github.com/naist-nlp/mbrs/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/naist-nlp/mbrs.svg"></a>
<a href=""><img src="https://github.com/naist-nlp/mbrs/actions/workflows/ci.yaml/badge.svg"></a>
</p>
<p align="center">
<b>
      <a href="https://aclanthology.org/2024.emnlp-demo.37">Paper</a> |
      <a href="https://mbrs.readthedocs.io">Reference docs</a> |
      <a href="https://github.com/naist-nlp/mbrs#citation">Citation</a>
</b>
</p>

## Installation

You can install from PyPi:

``` bash
pip install mbrs
```

For developers, it can be installed from the source.

``` bash
git clone https://github.com/naist-nlp/mbrs.git
cd mbrs/
pip install ./
```

## Quick start

mbrs provides two interfaces: command-line interface (CLI) and Python
API.

### Command-line interface

Command-line interface can run MBR decoding from command-line. Before
running MBR decoding, you can generate hypothesis sentences with
`mbrs-generate`:

``` bash
mbrs-generate \
  sources.txt \
  --output hypotheses.txt \
  --lang_pair en-de \
  --model facebook/m2m100_418M \
  --num_candidates 1024 \
  --sampling eps --epsilon 0.02 \
  --batch_size 8 --sampling_size 8 --fp16 \
  --report_format rounded_outline
```

Beam search can also be used by replacing
`--sampling eps --epsilon 0.02` with `--beam_size 10`.

Next, MBR decoding and other decoding methods can be executed with
`mbrs-decode`. This example regards the hypothesis set as the
pseudo-reference set.

``` bash
mbrs-decode \
  hypotheses.txt \
  --num_candidates 1024 \
  --nbest 1 \
  --source sources.txt \
  --references hypotheses.txt \
  --output translations.txt \
  --report report.txt --report_format rounded_outline \
  --decoder mbr \
  --metric comet \
  --metric.model Unbabel/wmt22-comet-da \
  --metric.batch_size 64 --metric.fp16 true
```

You can pass the arguments using a configuration yaml file via
`--config_path` option. See
[docs](https://mbrs.readthedocs.io/en/latest/yaml_config.html) for the
details.

Finally, you can evaluate the score with `mbrs-score`:

``` bash
mbrs-score \
  hypotheses.txt \
  --sources sources.txt \
  --references hypotheses.txt \
  --format json \
  --metric bleurt \
  --metric.batch_size 64 --metric.fp16 true
```

### Python API

This is the example of COMET-MBR via Python API.

``` python
from mbrs.metrics import MetricCOMET
from mbrs.decoders import DecoderMBR

SOURCE = "ありがとう"
HYPOTHESES = ["Thanks", "Thank you", "Thank you so much", "Thank you.", "thank you"]

# Setup COMET.
metric_cfg = MetricCOMET.Config(
  model="Unbabel/wmt22-comet-da",
  batch_size=64,
  fp16=True,
)
metric = MetricCOMET(metric_cfg)

# Setup MBR decoding.
decoder_cfg = DecoderMBR.Config()
decoder = DecoderMBR(decoder_cfg, metric)

# Decode by COMET-MBR.
# This example regards the hypotheses themselves as the pseudo-references.
# Args: (hypotheses, pseudo-references, source)
output = decoder.decode(HYPOTHESES, HYPOTHESES, source=SOURCE, nbest=1)

print(f"Selected index: {output.idx}")
print(f"Output sentence: {output.sentence}")
print(f"Expected score: {output.score}")
```

## List of implemented methods

### Metrics

Currently, the following metrics are supported:

-   BLEU [(Papineni et al., 2002)](https://aclanthology.org/P02-1040):
    `bleu`
-   TER [(Snover et al.,
    2006)](https://aclanthology.org/2006.amta-papers.25): `ter`
-   chrF [(Popović et al., 2015)](https://aclanthology.org/W15-3049):
    `chrf`
-   COMET [(Rei et al.,
    2020)](https://aclanthology.org/2020.emnlp-main.213): `comet`
-   COMETkiwi [(Rei et al.,
    2022)](https://aclanthology.org/2022.wmt-1.60): `cometkiwi`
-   XCOMET [(Guerreiro et al., 2023)](https://doi.org/10.1162/tacl_a_00683):
    `xcomet`
-   XCOMET-lite [(Larionov et al., 2024)](https://aclanthology.org/2024.emnlp-main.1223):
    `xcomet` with `--metric.model="myyycroft/XCOMET-lite"`
-   BLEURT [(Sellam et al.,
    2020)](https://aclanthology.org/2020.acl-main.704): `bleurt` (thanks
    to [\@lucadiliello](https://github.com/lucadiliello/bleurt-pytorch))
-   MetricX ([Juraska et al., 2023](https://aclanthology.org/2023.wmt-1.63);
    [Juraska et al., 2024](https://aclanthology.org/2024.wmt-1.35)): `metricx`

### Decoders

The following decoding methods are implemented:

-   N-best reranking: `rerank`
-   MBR decoding: `mbr`

Specifically, the following methods of MBR decoding are included:

-   Expectation estimation:
    -   Monte Carlo estimation ([Eikema and Aziz,
        2020](https://aclanthology.org/2020.coling-main.398); [Eikema
        and Aziz, 2022](https://aclanthology.org/2022.emnlp-main.754))
    -   Model-based estimation [(Jinnai et al.,
        2024)](https://proceedings.mlr.press/v235/jinnai24a.html): `--reference_lprobs`
        option
-   Efficient methods:
    -   Confidence-based pruning [(Cheng and Vlachos,
        2023)](https://aclanthology.org/2023.emnlp-main.767) :
        `pruning_mbr`
    -   Reference aggregation ([DeNero et al.,
        2009](https://aclanthology.org/P09-1064); [Vamvas and Sennrich,
        2024](https://aclanthology.org/2024.acl-short.71)): `aggregate_mbr`
        -   N-gram aggregation on BLEU [(DeNero et al.,
            2009)](https://aclanthology.org/P09-1064)
        -   N-gram aggregation on chrF [(Vamvas and Sennrich,
            2024)](https://aclanthology.org/2024.acl-short.71)
        -   Embedding aggregation on COMET ([Vamvas and Sennrich,
            2024](https://aclanthology.org/2024.acl-short.71); [Deguchi et al.,
            2024](https://aclanthology.org/2024.findings-acl.654))
    -   Centroid-based MBR [(Deguchi et al.,
        2024)](https://aclanthology.org/2024.findings-acl.654): `centroid_mbr`
    -   Probabilistic MBR [(Trabelsi et al.,
        2024)](https://arxiv.org/abs/2406.02832): `probabilistic_mbr`

### Selectors

The final output list is selected according to these selectors:

-   N-best selection: `nbest`
-   Diverse selection [(Jinnai et al., 2024)](https://aclanthology.org/2024.findings-acl.503): `diverse`

## Related projects

-   [mbr](https://github.com/ZurichNLP/mbr)
    -   Highly integrated with [huggingface
        transformers](https://huggingface.co/transformers) by
        customizing `generate()` method of model
        implementation.
    -   If you are looking for an MBR decoding library that is fully
        integrated into transformers, this might be a good choice.
    -   Our mbrs works standalone; thus, not only
        [transformers](https://huggingface.co/transformers) but also
        [fairseq](https://github.com/facebookresearch/fairseq) or LLM
        outputs via API can be used.

## Citation

If you use this software, please cite:

``` bibtex
@inproceedings{deguchi-etal-2024-mbrs,
    title = "mbrs: A Library for Minimum {B}ayes Risk Decoding",
    author = "Deguchi, Hiroyuki  and
      Sakai, Yusuke  and
      Kamigaito, Hidetaka  and
      Watanabe, Taro",
    editor = "Hernandez Farias, Delia Irazu  and
      Hope, Tom  and
      Li, Manling",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-demo.37",
    pages = "351--362",
}
```

## License

This library is mainly developed by [Hiroyuki
Deguchi](https://sites.google.com/view/hdeguchi) and published under the
MIT-license.
