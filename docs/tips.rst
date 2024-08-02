Tips
====

Oracle selection
----------------
If you have true references, you can get the oracle outputs.

The below is an example of COMET oracle selection.

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         from mbrs.metrics import MetricCOMET
         from mbrs.decoders import DecoderMBR

         SOURCE = "ありがとう"
         TRUE_REFERENCES = ["Thank you"]
         HYPOTHESES = ["Thanks", "Thank you", "Thank you so much", "Thank you.", "thank you"]

         metric_cfg = MetricCOMET.Config(model="Unbabel/wmt22-comet-da")
         metric = MetricCOMET(metric_cfg)
         decoder_cfg = DecoderMBR.Config()
         decoder = DecoderMBR(decoder_cfg, metric)

         output = decoder.decode(HYPOTHESES, TRUE_REFERENCES, source=SOURCE, nbest=1)

         print(f"Selected index: {output.idx}")
         print(f"Output sentence: {output.sentence}")
         print(f"Expected score: {output.score}")

   .. tab-item:: CLI

      .. code-block:: shell

         mbrs-decode \
           hypotheses.txt \
           --num_candidates 1024 \
           --references true_references.txt \
           --num_references 1 \
           --source sources.txt \
           --output oracle_translations.txt \
           --decoder mbr \
           --metric comet --metric.model "Unbabel/wmt22-comet-da"
