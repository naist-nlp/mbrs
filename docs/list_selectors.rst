Supported selectors
===================

:code:`Selector` class selects the final output list from hypotheses.

Supported selectors are listed below.

.. note::

   All selectors classes can be imported from :code:`mbrs.selectors`

Selection strategies
--------------------
.. list-table:: List of supported selection strategies
   :header-rows: 1
   :stub-columns: 1

   * - Strategy
     - CLI name :code:`--selector=<name>`
     - Class
     - Reference
     - Note
   * - N-best
     - :code:`nbest`
     - :doc:`SelectorNbest <./source/mbrs.selectors.nbest>`
     - --
     - --
   * - Diverse
     - :code:`diverse`
     - :doc:`SelectorDiverse <./source/mbrs.selectors.diverse>`
     - `(Jinnai et al., 2024) <https://aclanthology.org/2024.findings-acl.503>`_
     - :doc:`DecoderPruningMBR <./source/mbrs.decoders.pruning_mbr>` cannot use this selector.
