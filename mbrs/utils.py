import contextlib
import time


class Stopwatch:
    """Stopwatch class to measure the elapsed time.

    Example:
        >>> timer = Stopwatch()
        >>> for i in range(10):
                with timer.measure():
                    time.sleep(1)
        >>> print(f"{timer.total:.3f}")
        10.000
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the stopwatch."""
        self._acc_time = 0.0

    @contextlib.contextmanager
    def measure(self):
        """Start to measure the time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._acc_time += time.perf_counter() - start

    @property
    def total(self) -> float:
        """Return the total time."""
        return self._acc_time
