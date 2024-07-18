mbrs
####

*mbrs* is a library for minimum bayes risk (MBR) decoding.

Installation
============

We recommend to install from the source.

.. code:: bash

    git clone https://github.com/naist-nlp/mbrs.git
    cd mbrs/
    pip install ./

Also, you can install from PyPi:

.. code:: bash

    pip install mbrs


Quick start
===========

mbrs provides two interfaces: command-line interface (CLI) and Python API.

Command-line interface
----------------------

Command-line interface can run MBR decoding from command-line.
Before running MBR decoding, you can generate hypothesis sentences with :code:`mbrs-generate`:

.. code:: bash

    mbrs-generate \
      sources.txt \
      --output hypotheses.txt \
      --lang_pair en-de \
      --model facebook/m2m100_418M \
      --num_candidates 1024 \
      --sampling eps --epsilon 0.02 \
      --batch_size 8 --sampling_size 8 --fp16 \
      --report_format rounded_outline

Beam search can also be used by replacing :code:`--sampling eps --epsilon 0.02` with :code:`--beam_size 10`.

Next, MBR decoding and other decoding methods can be executed with :code:`mbrs-decode`.
This example regards the hypothesis set as the pseudo-reference set.

.. code:: bash

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

Finally, you can evaluate the score with :code:`mbrs-score`:

.. code:: bash

    mbrs-score \
      hypotheses.txt \
      --sources sources.txt \
      --references hypotheses.txt \
      --format json \
      --metric bleurt \
      --metric.batch_size 64 --metric.fp16 true


Python API
----------
This is the example of COMET-MBR via Python API.

.. code:: python

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

List of implemented methods
===========================

Currently, the following metrics are supported:

- BLEU `(Papineni et al., 2002) <https://aclanthology.org/P02-1040>`_: :code:`bleu`
- TER `(Snover et al., 2006) <https://aclanthology.org/2006.amta-papers.25>`_: :code:`ter`
- chrF `(Popović et al., 2015) <https://aclanthology.org/W15-3049>`_: :code:`chrf`
- COMET `(Rei et al., 2020) <https://aclanthology.org/2020.emnlp-main.213>`_: :code:`comet`
- COMETkiwi `(Rei et al., 2022) <https://aclanthology.org/2022.wmt-1.60>`_: :code:`cometkiwi`
- XCOMET `(Guerreiro et al., 2023) <https://arxiv.org/abs/2310.10482>`_: :code:`xcomet`
- BLEURT `(Sellam et al., 2020) <https://aclanthology.org/2020.acl-main.704>`_: :code:`bleurt` (thanks to `@lucadiliello <https://github.com/lucadiliello/bleurt-pytorch>`_)

The following decoding methods are implemented:

- N-best reranking: :code:`rerank`
- MBR decoding: :code:`mbr`

Specifically, the following methods of MBR decoding are included:

- Expectation estimation:

  - Monte Carlo estimation (`Eikema and Aziz, 2020 <https://aclanthology.org/2020.coling-main.398>`_; `Eikema and Aziz, 2022 <https://aclanthology.org/2022.emnlp-main.754>`_)
  - Model-based estimation `(Jinnai et al., 2024) <https://arxiv.org/abs/2311.05263>`_: :code:`--reference_lprobs` option

- Efficient methods:

  - Confidence-based pruning `(Cheng and Vlachos, 2023) <https://aclanthology.org/2023.emnlp-main.767>`_ : :code:`pruning_mbr`
  - Reference aggregation (`DeNero et al., 2009 <https://aclanthology.org/P09-1064>`_; `Vamvas and Sennrich, 2024 <https://arxiv.org/abs/2402.04251>`_): :code:`aggregate_mbr`

    - N-gram aggregation on BLEU `(DeNero et al., 2009) <https://aclanthology.org/P09-1064>`_
    - N-gram aggregation on chrF `(Vamvas and Sennrich, 2024) <https://arxiv.org/abs/2402.04251>`_
    - Embedding aggregation on COMET (`Vamvas and Sennrich, 2024 <https://arxiv.org/abs/2402.04251>`_; `Deguchi et al., 2024 <https://arxiv.org/abs/2402.11197>`_)

  - Centroid-based MBR `(Deguchi et al., 2024) <https://arxiv.org/abs/2402.11197>`_: :code:`centroid_mbr`
  - Probabilistic MBR `(Trabelsi et al., 2024) <https://arxiv.org/abs/2406.02832>`_: :code:`probabilistic_mbr`

Related projects
================

- `mbr <https://github.com/ZurichNLP/mbr>`_

  - Highly integrated with `huggingface transformers <https://huggingface.co/transformers>`_ by customizing `generate()` method of model implementation.
  - If you are looking for an MBR decoding library that is fully integrated into transformers, this might be a good choice.
  - Our mbrs works standalone; thus, not only `transformers <https://huggingface.co/transformers>`_ but also `fairseq <https://github.com/facebookresearch/fairseq>`_ or LLM outputs via API can be used.

Citation
========
If you use this software, please cite:

.. code:: bibtex

   @software{Deguchi_mbrs_2024,
     author = {Deguchi, Hiroyuki},
     month = jun,
     title = {{mbrs}},
     url = {https://github.com/naist-nlp/mbrs},
     year = {2024}
   }

License
=======
This library is mainly developed by `Hiroyuki Deguchi <https://sites.google.com/view/hdeguchi>`_ and published under the MIT-license.
