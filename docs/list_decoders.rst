Supported decoders
==================

Supported decoders are listed below.

.. note::

   All decoders classes can be imported from :code:`mbrs.decoders`

Decoding strategies
-------------------
.. list-table:: List of supported decoding strategies
   :header-rows: 1
   :stub-columns: 1

   * - Strategy
     - CLI name :code:`--decoder=<name>`
     - Class
     - Supported metrics
     - Reference
   * - MBR decoding
     - :code:`mbr`
     - :doc:`DecoderMBR <./source/mbrs.decoders.mbr>`
     - all
     - (`Eikema and Aziz, 2020 <https://aclanthology.org/2020.coling-main.398>`_; `Eikema and Aziz, 2022 <https://aclanthology.org/2022.emnlp-main.754>`_)
   * - N-best reranking
     - :code:`rerank`
     - :doc:`DecoderRerank <./source/mbrs.decoders.rerank>`
     - COMETkiwi
     - --

Expectation estimations
-----------------------

.. list-table:: List of supported expectation estimations
   :header-rows: 1
   :stub-columns: 1

   * - Estimation
     - CLI option
     - Reference
   * - Monte Carlo
     - --
     - (`Eikema and Aziz, 2020 <https://aclanthology.org/2020.coling-main.398>`_; `Eikema and Aziz, 2022 <https://aclanthology.org/2022.emnlp-main.754>`_)
   * - Model-based
     - :code:`--reference_lprobs=lprobs.txt`
     - `(Jinnai et al., 2024) <https://arxiv.org/abs/2311.05263>`_

Efficient MBR decoders
----------------------

.. list-table:: List of supported efficient MBR decoders
   :header-rows: 1
   :stub-columns: 1

   * - Decoder
     - CLI name :code:`--decoder=<name>`
     - Class
     - Supported metrics
     - Reference
   * - Confidence-based pruning
     - :code:`pruning_mbr`
     - :doc:`DecoderPruningMBR <./source/mbrs.decoders.pruning_mbr>`
     - all
     - `(Cheng and Vlachos, 2023) <https://aclanthology.org/2023.emnlp-main.767>`_
   * - Reference aggregation
     - :code:`aggregate_mbr`
     - :doc:`DecoderAggregateMBR <./source/mbrs.decoders.aggregate_mbr>`
     - .. list-table::

         * - BLEU: N-gram and length aggregation `(DeNero et al., 2009) <https://aclanthology.org/P09-1064>`_
         * - chrF: N-gram aggregation `(Vamvas and Sennrich, 2024) <https://arxiv.org/abs/2402.04251>`_
         * - COMET: Embedding aggregation (`Vamvas and Sennrich, 2024 <https://arxiv.org/abs/2402.04251>`_; `Deguchi et al., 2024 <https://arxiv.org/abs/2402.11197>`_)

     - (`DeNero et al., 2009 <https://aclanthology.org/P09-1064>`_; `Vamvas and Sennrich, 2024 <https://arxiv.org/abs/2402.04251>`_)
   * - Centroid-based aggregation
     - :code:`centroid_mbr`
     - :doc:`DecoderCentroidMBR <./source/mbrs.decoders.centroid_mbr>`
     - COMET
     - `(Deguchi et al., 2024) <https://arxiv.org/abs/2402.11197>`_
   * - Low-Rank Matrix Completion (Probabilistic MBR)
     - :code:`probabilistic_mbr`
     - :doc:`DecoderProbabilisticMBR <./source/mbrs.decoders.probabilistic_mbr>`
     - all
     - `(Trabelsi et al., 2024) <https://arxiv.org/abs/2406.02832>`_
