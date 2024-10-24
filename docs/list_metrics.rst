Supported metrics
=================

Supported metrics are listed below.

.. note::

   All metric classes can be imported from :code:`mbrs.metrics`

.. list-table:: List of supported metrics
   :header-rows: 1
   :stub-columns: 1

   * - Metric
     - CLI name :code:`--metric=<name>`
     - Class
     - Reference
   * - BLEU
     - :code:`bleu`
     - :doc:`MetricBLEU <./source/mbrs.metrics.bleu>`
     - `(Papineni et al., 2002) <https://aclanthology.org/P02-1040>`_
   * - TER
     - :code:`ter`
     - :doc:`MetricTER <./source/mbrs.metrics.ter>`
     - `(Snover et al., 2006) <https://aclanthology.org/2006.amta-papers.25>`_
   * - chrF
     - :code:`chrf`
     - :doc:`MetricChrF <./source/mbrs.metrics.chrf>`
     - `(Popović et al., 2015) <https://aclanthology.org/W15-3049>`_
   * - COMET
     - :code:`comet`
     - :doc:`MetricCOMET <./source/mbrs.metrics.comet>`
     - `(Rei et al., 2020) <https://aclanthology.org/2020.emnlp-main.213>`_
   * - COMETkiwi
     - :code:`cometkiwi`
     - :doc:`MetricCOMETkiwi <./source/mbrs.metrics.cometkiwi>`
     - `(Rei et al., 2022) <https://aclanthology.org/2022.wmt-1.60>`_
   * - XCOMET
     - :code:`xcomet`
     - :doc:`MetricXCOMET <./source/mbrs.metrics.xcomet>`
     - `(Guerreiro et al., 2023) <https://doi.org/10.1162/tacl_a_00683>`_
   * - BLEURT
     - :code:`bleurt`
     - :doc:`MetricBLEURT <./source/mbrs.metrics.bleurt>`
     - `(Sellam et al., 2020) <https://aclanthology.org/2020.acl-main.704>`_
