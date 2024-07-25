Command-line interface
======================

mbrs provides useful command-line interface (CLI) scripts.

.. seealso::

   :doc:`Manual of CLI options <./cli_help>`
        Detailed documentation of CLI options.

Overview
--------

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
