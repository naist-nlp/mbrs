import time

import pytest

from .utils import Stopwatch


class TestStopwatch:
    def test__init__(self) -> None:
        timer = Stopwatch()
        assert timer._acc_time == 0.0

    def test_reset(self) -> None:
        timer = Stopwatch()
        assert timer._acc_time == 0.0
        timer._acc_time = 1.0
        assert timer._acc_time != 0.0
        timer.reset()
        assert timer._acc_time == 0.0

    def test_measure(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timer = Stopwatch()
        for _ in range(10):
            with timer.measure():
                t += 1.0
        assert timer.total == 10.0
