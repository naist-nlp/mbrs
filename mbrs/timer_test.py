import time

import pytest

from .timer import ProfileTree, Stopwatch, StopwatchDict


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

    def test___call__(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timer = Stopwatch()
        for _ in range(10):
            with timer():
                t += 1.0
        assert timer.elpased_time == 10.0

    def test_nest_duplicate_measurement(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timer = Stopwatch()
        with timer():
            t += 10.0
            for _ in range(10):
                with timer():
                    t += 1.0
        assert timer.elpased_time == 10.0

    def test_ncalls(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timer = Stopwatch()
        for _ in range(20):
            with timer() as ts:
                t += 2.0
                ts.set_delta_ncalls(2)
        assert timer.elpased_time == 40.0
        assert timer.ncalls == 40


class TestStopwatchDict:
    def test__init__(self) -> None:
        timers = StopwatchDict()
        assert len(timers) == 0

    def test_reset(self) -> None:
        timers = StopwatchDict()
        names = ["A", "B"]
        assert all([timers[name]._acc_time == 0.0 for name in names])
        for t in timers.values():
            t._acc_time = 1.0
        assert all([timers[name]._acc_time != 0.0 for name in names])
        timers.reset()
        assert all([timers[name]._acc_time == 0.0 for name in names])

    def test___call__(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timers = StopwatchDict()
        for _ in range(10):
            with timers("A"):
                t += 1.0
        for i in range(3):
            with timers("B") as ts:
                t += 3.0
                ts.set_delta_ncalls(i + 1)
        assert timers.elapsed_time == {"A": 10.0, "B": 9.0}
        assert timers.ncalls == {"A": 10, "B": 6}

    def test_nest(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timers = StopwatchDict()
        for _ in range(10):
            with timers("A"):
                for _ in range(3):
                    with timers("B") as ts:
                        t += 3.0
                        ts.set_delta_ncalls(3)
        assert timers.elapsed_time == {"A": 90.0, "B": 90.0}
        assert timers.ncalls == {"A": 10, "B": 90}


class TestProfileTree:
    def test_is_leaf(self):
        root = ProfileTree()
        assert root.is_leaf

        root.children["A"] = ProfileTree()
        root.children["B"] = ProfileTree()
        assert not root.is_leaf
        assert root.children["A"].is_leaf
        assert root.children["B"].is_leaf

    def test_aggregate(self):
        root = ProfileTree()
        for i, name in enumerate(["A", "B", "C"], start=1):
            root.children[name] = ProfileTree(float(i) ** 2, i)
        root.children["D"] = ProfileTree()
        root.children["D"].children["a"] = ProfileTree(7.0, 7)
        root.children["D"].children["b"] = ProfileTree(13.0, 13)
        root.aggregate()
        assert root.elapsed_time == 34.0
        assert root.ncalls == 26

        root = ProfileTree()
        root.children["A"] = ProfileTree(5.0, 5)
        root.children["A"].children["a"] = ProfileTree(7.0, 7)
        root.children["A"].children["b"] = ProfileTree(13.0, 13)
        root.aggregate()
        assert root.elapsed_time == 5.0
        assert root.ncalls == 5
        assert root.children["A"].elapsed_time == 5.0
        assert root.children["A"].ncalls == 5

        root = ProfileTree()
        root.children["A"] = ProfileTree()
        root.children["A"].children["a"] = ProfileTree(ncalls=13)
        with pytest.raises(RuntimeError):
            root.aggregate()
        root = ProfileTree()
        root.children["A"] = ProfileTree()
        root.children["A"].children["a"] = ProfileTree(13.0)
        with pytest.raises(RuntimeError):
            root.aggregate()

    def test_build(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timers = StopwatchDict()
        for _ in range(10):
            with timers("A"):
                t += 1.0
        for i in range(3):
            with timers("B") as ts:
                t += 3.0
                ts.set_delta_ncalls(i + 1)
        for i in range(5):
            with timers("C/a") as ts:
                t += 3.0
                ts.set_delta_ncalls(3)
            with timers("C/b"):
                t += 1.0

        root = ProfileTree.build(timers)

        assert root.children["A"].elapsed_time == 10.0
        assert root.children["A"].ncalls == 10
        assert root.children["B"].elapsed_time == 9.0
        assert root.children["B"].ncalls == 6
        C = root.children["C"]
        assert C.children["a"].elapsed_time == 15.0
        assert C.children["a"].ncalls == 15
        assert C.children["b"].elapsed_time == 5.0
        assert C.children["b"].ncalls == 5
        assert C.elapsed_time == 20.0
        assert C.ncalls == 20
        assert root.elapsed_time == 39.0
        assert root.ncalls == 36

    def test_result(self, monkeypatch: pytest.MonkeyPatch):
        t = 0.0

        def perf_counter_mock() -> float:
            return t

        monkeypatch.setattr(time, "perf_counter", perf_counter_mock)

        timers = StopwatchDict()
        for _ in range(10):
            with timers("A"):
                t += 1.0
        for i in range(3):
            with timers("B") as ts:
                t += 3.0
                ts.set_delta_ncalls(i + 1)
        for i in range(5):
            with timers("C/a") as ts:
                t += 3.0
                ts.set_delta_ncalls(3)
            with timers("C/b"):
                t += 1.0
            with timers("C/c/1"):
                t += 1.0
            with timers("C/c/2"):
                t += 2.0

        root = ProfileTree.build(timers)
        res = root.result()
        expected = [
            # {"name": "/", "elapsed_time": 54.0, "ncalls": 46, "average_time": 1173.9},
            {"name": "A", "acctime": 10.0, "acccalls": 10, "ms/call": 1000.0},
            {"name": "B", "acctime": 9.0, "acccalls": 6, "ms/call": 1500.0},
            {"name": "C", "acctime": 35.0, "acccalls": 30, "ms/call": 35.0 * 1000 / 30},
            {
                "name": "C/a",
                "acctime": 15.0,
                "acccalls": 15,
                "ms/call": 1000.0,
            },
            {"name": "C/b", "acctime": 5.0, "acccalls": 5, "ms/call": 1000.0},
            {
                "name": "C/c",
                "acctime": 15.0,
                "acccalls": 10,
                "ms/call": 1500.0,
            },
            {
                "name": "C/c/1",
                "acctime": 5.0,
                "acccalls": 5,
                "ms/call": 1000.0,
            },
            {
                "name": "C/c/2",
                "acctime": 10.0,
                "acccalls": 5,
                "ms/call": 2000.0,
            },
        ]
        for r, e in zip(res, expected):
            assert r == e
