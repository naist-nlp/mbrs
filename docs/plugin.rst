Plug-in loader
==============

:code:`mbrs-decode` and :code:`mbrs-score` load plug-in modules via the :code:`--plugin_dir` option.

Examples
~~~~~~~~

This tutorial explains how to load a user defined modules.

.. seealso::

   :doc:`How to define a new metric <./custom_metric>`
        Detailed documentation of the metric customization.

   :doc:`How to define a new decoder <./custom_decoder>`
        Detailed documentation of the decoder customization.

1. Define a new metric, decoder, or selector with :code:`@register` decorator.

   .. code-block:: python
      :emphasize-lines: 4

      from mbrs.metrics import register, Metric, MetricBLEU


      @register("my_bleu")
      class MetricMyBLEU(MetricBLEU):
          ...

2. Prepare :code:`__init__.py` to specify classes to be loaded.

   .. code-block:: python
      :emphasize-lines: 1

      from .new import MetricNew

3. Then, load the modules with :code:`--plugin_dir` option with a path to the directory containing the above :code:`__init__.py`.

   .. code-block:: bash
      :emphasize-lines: 2,6

      mbrs-decode \
        --plugin_dir path/to/plugins/ \
        hypotheses.txt \
        --num_candidates 1024 \
        --decoder mbr \
        --metric my_bleu

   :code:`mbrs-score` also supports plug-in loading.

   .. code-block:: bash
      :emphasize-lines: 2,5

      mbrs-score \
        --plugin_dir path/to/plugins/ \
        hypotheses.txt \
        -r references.txt \
        --metric my_bleu
