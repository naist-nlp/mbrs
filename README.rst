mbrs
####

*mbrs* is a library for minimum bayes risk (MBR) decoding.

Installation
============

.. code:: bash

    git clone https://github.com/naist-nlp/mbrs.git
    cd mbrs/
    pip install ./

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
      --metric.batch_size 64 --metric.fp16

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

License
=======
This library is published under the MIT-license.
