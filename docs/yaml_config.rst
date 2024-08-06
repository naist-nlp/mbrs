YAML Configuration
==================

Command-line arguments can be passed by a configuration yaml file via :code:`--config_path`.

.. seealso::

   :doc:`Command-line interface <./cli>`
        Overview of the command-line interface.

This is an example of COMET-MBR.
:code:`hypotheses.txt` are also used as the pseudo-references.

.. code-block:: bash

    mbrs-decode \
      hypotheses.txt \
      --num_candidates 1024 \
      --nbest 1 \
      --source sources.txt \
      --references hypotheses.txt \
      --output translations.txt \
      --report report.txt --report_format tsv \
      --decoder mbr \
      --metric comet \
      --metric.model Unbabel/wmt22-comet-da \
      --metric.batch_size 64 --metric.fp16 true

All arguments can be passed via :code:`--config_path`,

.. code-block:: bash

   mbrs-decode --config_path comet_mbr.yaml

with a configuration yaml:

.. code-block:: yaml
   :caption: comet_mbr.yaml

   common:
     hypotheses: hypotheses.txt
     num_candidates: 1024
     nbest: 1
     source: sources.txt
     references: hypotheses.txt
     output: translations.txt
     report: report.txt
     report_format: tsv
     decoder: mbr
     metric: comet

   metric:
     model: Unbabel/wmt22-comet-da
     batch_size: 64
     fp16: true

The arguments with dot-prefixes are loaded from each key in the yaml, and others are loaded from the :code:`common:` key.
In other words, :code:`--metric.` and :code:`--decoder.` are loaded from each corresponding key in the yaml, i.e., :code:`metric:` or :code:`decoder:`.

Of course, you can override the values via command-line arguments, for example:

.. code-block:: bash

   mbrs-decode --config_path comet_mbr.yaml --nbest 1024
