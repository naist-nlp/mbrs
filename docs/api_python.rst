Python interface
================

mbrs is implemented in Python and PyTorch.

.. seealso::

   :doc:`References of Python API <./source/mbrs>`
        Detailed documentation of Python API.

Examples
--------
This is a Python API example of COMET-MBR.

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
