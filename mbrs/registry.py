from typing import Callable, TypeVar

T = TypeVar("T")


class Registry(dict[str, type[T]]):
    """Registry that maps a name to its corresponding type."""

    def __init__(self, base_type: type[T]):
        super().__init__()
        REGISTRIES[base_type] = self
        self._base_type = base_type

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Register a type as the given name.

        Args:
            name (str): The name of a type.

        Returns:
            Callable[[type[T]], type[T]]: Register decorator function.

        Raises:
            ValueError: The type is already registered.
        """

        def _register(cls: type[T]) -> type[T]:
            if not issubclass(cls, self._base_type):
                raise ValueError(f"`{cls.__name__}` must inherit `{self._base_type}`.")

            if (registered := self.get(name)) is not None:
                raise ValueError(
                    f"{cls.__name__}: `{name}` already registered as `{registered.__name__}`."
                )
            self[name] = cls
            return cls

        return _register

    def get_cls(self, name: str) -> type[T]:
        """Get a class type.

        Args:
            name: A registered name.

        Returns:
            type[T]: Class type.
        """
        return self.__getitem__(name)

    def get_closure(
        self,
    ) -> tuple[Callable[[str], Callable[[type[T]], type[T]]], Callable[[str], type[T]]]:
        """Get closure functions: `register()` and `get_cls()`.

        Returns:
            tuple:
              - Callable[[str], Callable[[type[T]], type[T]]]: `register()` function.
              - Callable[[str], type[T]]: `get_cls()` function.
        """
        return (self.register, self.get_cls)


REGISTRIES: dict[type, Registry] = {}


def get_registry(base_type: type[T]) -> Registry[T]:
    """Get registry of the given base class type.

    Args:
        base_type (type[T]): Base class type that associated with the registry to be returned.

    Returns:
        Registry[T]: Class mapper from registered name to its corresponding class.
    """
    return REGISTRIES[base_type]
