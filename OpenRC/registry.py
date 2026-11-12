"""String-keyed factory registry for OpenRC extension points.

Each extensible component family defines a :class:`Registry` instance.
Implementations register via :meth:`Registry.register` or :meth:`Registry.add`::

    from OpenRC.compression.base import COMPRESSORS, Compressor

    @COMPRESSORS.register("my_method", doc="One-line description for --help")
    class MyCompressor(Compressor):
        ...

Registered entries are resolved by ``get_<kind>("my_method")`` and listed by
``python scripts/demo.py list``.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, List, Tuple, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a short string key to a factory (usually a class).

    Parameters
    ----------
    kind:
        Human readable name of the family, only used in error messages
        (``"compressor"``, ``"case"``, ...).
    """

    def __init__(self, kind: str) -> None:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("registry kind must be a non-empty string")
        self.kind = kind
        self._items: Dict[str, T] = {}
        self._docs: Dict[str, str] = {}

    # ------------------------------------------------------------------ write
    def register(self, name: str, *, doc: str = "") -> Callable[[T], T]:
        """Decorator form: ``@REGISTRY.register("name", doc="...")``."""

        def _decorator(obj: T) -> T:
            self.add(name, obj, doc=doc)
            return obj

        return _decorator

    def add(self, name: str, obj: T, *, doc: str = "") -> None:
        """Register ``obj`` under ``name`` imperatively (e.g. for factory functions)."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.kind} name must be a non-empty string")
        if not isinstance(doc, str):
            raise ValueError(f"{self.kind} doc must be a string, got {type(doc).__name__}")
        if name in self._items:
            raise KeyError(f"{self.kind} {name!r} is already registered")
        self._items[name] = obj
        # Default to the first line of obj.__doc__ when doc is omitted.
        self._docs[name] = doc or (getattr(obj, "__doc__", "") or "").strip().split("\n")[0]

    # ------------------------------------------------------------------- read
    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except (KeyError, TypeError):
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}"
            ) from None

    def names(self) -> List[str]:
        return sorted(self._items)

    def describe(self) -> List[Tuple[str, str]]:
        """``[(name, one-line doc), ...]`` sorted by name - used by ``demo.py list``."""
        return [(n, self._docs.get(n, "")) for n in self.names()]

    def __contains__(self, name: object) -> bool:
        try:
            return name in self._items
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<Registry {self.kind}: {', '.join(self.names())}>"
