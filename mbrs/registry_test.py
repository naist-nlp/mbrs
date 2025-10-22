import pytest

from . import registry


class MockBase: ...


def test_registry():
    mock_registry = registry.Registry(MockBase)
    assert "mock1" not in mock_registry

    @mock_registry.register("mock1")
    class MockClass1(MockBase): ...

    assert "mock1" in mock_registry
    assert issubclass(mock_registry["mock1"], MockClass1)
    assert issubclass(mock_registry.get_cls("mock1"), MockClass1)

    @mock_registry.register("mock2")
    class MockClass2(MockBase): ...

    assert "mock2" in mock_registry
    assert issubclass(mock_registry["mock2"], MockClass2)
    assert issubclass(mock_registry.get_cls("mock2"), MockClass2)

    assert registry.get_registry(MockBase) == {"mock1": MockClass1, "mock2": MockClass2}


def test_check_duplicate_registration():
    mock_registry = registry.Registry(MockBase)

    @mock_registry.register("mock1")
    class MockClass1(MockBase): ...

    with pytest.raises(ValueError):

        @mock_registry.register("mock1")
        class MockClass3(MockBase): ...


def test_check_base_type():
    mock_registry = registry.Registry(MockBase)

    @mock_registry.register("mock1")
    class MockClass1(MockBase): ...

    @mock_registry.register("mock_child")
    class MockClassChild(MockClass1): ...

    with pytest.raises(ValueError):

        @mock_registry.register("no_base")
        class MockClassNoBase: ...

    class MockBase2: ...

    with pytest.raises(ValueError):

        @mock_registry.register("base2")
        class MockClassBase2(MockBase2): ...


def test_regsiter_union():
    class MockBase2: ...

    mock_registry = registry.Registry(MockBase | MockBase2)

    @mock_registry.register("mock1")
    class MockClass1(MockBase): ...

    @mock_registry.register("mock2")
    class MockClass2(MockBase2): ...

    assert registry.get_registry(MockBase | MockBase2) is mock_registry
    assert registry.get_registry(MockBase2 | MockBase) is mock_registry
