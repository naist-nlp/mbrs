"""
Timer module including some useful global objects.

Example:
    >>> from mbrs import timer
    >>> for hyps, src in zip(hypotheses, sources):
            with timer.measure("encode/hypotheses") as t:
                h = metric.encode(hyps)
            t.set_delta_ncalls(len(hyps))
            with timer.measure("encode/source"):
                s = metric.encode([src])
            with timer.measure("score"):
                scores = metric.score(h, s)
    >>> res = timer.aggregate().result()  # return the result table
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass, field


class Stopwatch:
    """Stopwatch class to measure the elapsed time.

    Example:
        >>> timer = Stopwatch()
        >>> for i in range(10):
                with timer():
                    time.sleep(1)
        >>> print(f"{timer.elapsed_time:.3f}")
        10.000

        >>> timer = Stopwatch()
        >>> for i in range(10):
                with timer() as t:
                    time.sleep(2)
                    t.set_delta_ncalls(2)
        >>> print(f"{timer.elapsed_time:.3f}")
        20.000
        >>> print(f"{timer.ncalls}")
        20
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the stopwatch."""
        self._acc_time = 0.0
        self._acc_ncalls = 0
        self._delta_ncalls = 1

    @contextlib.contextmanager
    def __call__(self):
        """Measure the time."""
        _acc_time = self._acc_time
        _acc_ncalls = self._acc_ncalls
        start = time.perf_counter()
        try:
            self.set_delta_ncalls(1)  # Set to 1 in the default
            yield self
        finally:
            # Treat nest calling
            if _acc_time == self._acc_time:
                self._acc_time = _acc_time + time.perf_counter() - start
            if _acc_ncalls == self._acc_ncalls:
                self._acc_ncalls = _acc_ncalls + self._delta_ncalls

    def set_delta_ncalls(self, delta: int = 1):
        """Set delta for counting the number of calls."""
        self._delta_ncalls = delta

    @property
    def elpased_time(self) -> float:
        """Return the total elapsed time."""
        return self._acc_time

    @property
    def ncalls(self) -> int:
        """Return the number of calls."""
        return self._acc_ncalls


class StopwatchDict(defaultdict[str, Stopwatch]):
    """A dictionary of the :class:`Stopwatch` class.

    Example:
        >>> timers = StopwatchDict()
        >>> for i in range(10):
                with timers("A"):
                    time.sleep(1)
            for i in range(3):
                with timers("B"):
                    time.sleep(1)
        >>> print(f"{timers.total}")
        {"A": 10.000, "B": 3.000}
    """

    def __init__(self) -> None:
        super().__init__(Stopwatch)

    def reset(self) -> None:
        """Reset all stopwatches."""
        for t in self.values():
            t.reset()

    @contextlib.contextmanager
    def __call__(self, name: str):
        """Measure the time."""
        with self[name]() as timer:
            try:
                yield timer
            finally:
                pass

    @property
    def elapsed_time(self) -> dict[str, float]:
        """Return the total elapsed time."""
        return {k: v.elpased_time for k, v in self.items()}

    @property
    def ncalls(self) -> dict[str, int]:
        """Return the number of calls."""
        return {k: v.ncalls for k, v in self.items()}


measure = StopwatchDict()


@dataclass
class ProfileTree:
    elapsed_time: float = -1.0
    ncalls: int = -1
    children: dict[str, ProfileTree] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """Return whether the node is leaf or not."""
        return len(self.children) == 0

    def aggregate(self):
        if self.is_leaf:
            if self.elapsed_time < 0.0:
                raise RuntimeError("Missing elapsed_time")
            if self.ncalls < 0:
                raise RuntimeError("Missing ncalls")
        else:
            for child in self.children.values():
                child.aggregate()
            if self.elapsed_time < 0.0:
                self.elapsed_time = 0.0
                for child in self.children.values():
                    self.elapsed_time += child.elapsed_time
            if self.ncalls < 0:
                self.ncalls = 0
                for child in self.children.values():
                    self.ncalls += child.ncalls

    @classmethod
    def build(cls, timers: StopwatchDict, separetor: str = "/"):
        root = cls()
        for name, timer in timers.items():
            prefix = name.split(separetor)
            node = root
            for path in prefix:
                if path not in node.children:
                    node.children[path] = ProfileTree()
                node = node.children[path]
            node.elapsed_time = timer.elpased_time
            node.ncalls = timer.ncalls
        root.aggregate()
        return root

    def result(self, nsentences: int = -1) -> list[dict[str, str | int | float]]:
        def _result(name: str, node: ProfileTree) -> list[dict[str, str | int | float]]:
            stat = {
                "name": name.strip("/"),
                "acctime": node.elapsed_time,
                "acccalls": node.ncalls,
                "ms/call": node.elapsed_time * 1000 / node.ncalls,
            }
            if nsentences > 0:
                stat["ms/sentence"] = node.elapsed_time * 1000 / nsentences
                stat["calls/sentence"] = node.ncalls / nsentences
            res = [stat]
            for path, child in node.children.items():
                res += _result(name + "/" + path, child)
            return res

        res = _result("", self)
        return res[1:]  # Remove the root node


def aggregate() -> ProfileTree:
    """Aggregate the timers.

    Returns:
        ProfileTree: The root of the profile tree.
    """
    return ProfileTree.build(measure)
