Measuring speed and ncalls
==========================

:doc:`timer <source/mbrs.timer>` module measures the cumulative processing time, averaged processing time, and the number of calls of scopes.

This document consists of two parts:

- :ref:`stopwatch`
- :ref:`collect and aggregate`

.. _stopwatch:

Stopwatch class
~~~~~~~~~~~~~~~

Stopwatch class measures the elapsed time and the number of calls using :code:`with` block.

This is a minimal example of :code:`Stopwatch`.

.. code:: python

   >>> import mbrs.timer
   >>> timer = mbrs.timer.Stopwatch()
   >>> for i in range(10):
   ...     with timer():
   ...         time.sleep(1)
   >>> print(f"{timer.elapsed_time:.3f}")
   10.000
   >>> print(f"{timer.ncalls}")
   10

:code:`Stopwatch.__call__()` can be used with the :code:`with` block.
It automatically measures the elapsed time within the scope and and the number of calling the scope.

- :code:`elpased_time` property returns the accumulated elapsed time within the scope.
- :code:`ncalls` property returns the number of calling the scope.

You can manually adjust the :code:`ncalls` by :code:`.set_delta_ncalls()`.
This method is convenient for measuring the batch computation which cannot be traced from python codes.

   >>> import mbrs.timer
   >>> timer = mbrs.timer.Stopwatch()
   >>> for i in range(10):
   ...     with timer() as t:
   ...         time.sleep(2)
   ...         t.set_delta_ncalls(2)
   >>> print(f"{timer.elapsed_time:.3f}")
   20.000
   >>> print(f"{timer.ncalls}")
   20

.. _collect and aggregate:

Collect and aggregate all statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To collect and aggregate all running statistics, we provide the global timer object, :code:`mbrs.timer.measure`.
We do not need to prepare :code:`Stopwatch` instance for each scope, instead, just call :code:`with mbrs.timer.measure("...")`.

.. code:: python

   >>> from mbrs import timer
   >>> for hyps, src in zip(hypotheses, sources):
   ...     with timer.measure("encode/hypotheses") as t:
   ...         h = metric.encode(hyps)
   ...     t.set_delta_ncalls(len(hyps))
   ...     with timer.measure("encode/source"):
   ...         s = metric.encode([src])
   ...     with timer.measure("score"):
   ...         scores = metric.score(h, s)
   >>> res = timer.aggregate().result()  # return the result table

:code:`mbrs.timer.measure` provides some powerful features:

- All measured statistics can be managed via a single object.
- The statistics that have the same parent names in the scope key can be automatically aggregated.

The above example has three measured scopes, :code:`"encode/hypotheses"`, :code:`"encode/source"`, and :code:`"source"`.
The two scopes have shared parent name in the key, i.e., :code:`"encode/..."`.
Then, they are automatically aggregated as :code:`"encode"`.

Thus, the result table of the above example has four statistics:

- Measured statistics: :code:`"encode/hypotheses"`, :code:`"encode/source"`, and :code:`"source"`
- Aggregated statistics: :code:`"encode"`
