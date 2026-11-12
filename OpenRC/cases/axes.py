"""Coordinate-axis definitions for security regions.

Each dimension of a security region corresponds to an active (``"p"``, MW) or
reactive (``"q"``, Mvar) power injection offset at a point-of-connection (POC)
bus, represented as a ``(bus, quantity)`` pair in :class:`RegionAxis`.
Integer specifications such as ``[1, 3, 6]`` default to active power; mixed
active/reactive specifications such as ``["1p", "1q", "3p"]`` are supported
throughout (e.g. Method 1 shifts ``"p"`` axes by ``r * l_max`` and ``"q"`` axes
by ``x * l_max``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple, Union

# Accepts 3, "3", "3p", "3q" and the label form this module emits, "b3p".
_AXIS_PATTERN = re.compile(r"^b?(?P<bus>\d+)(?P<quantity>[pq])?$", re.IGNORECASE)

AxisSpec = Union[int, str, "RegionAxis", Tuple[int, str]]


@dataclass(frozen=True)
class RegionAxis:
    """One dimension of a region: active or reactive injection at one bus."""

    bus: int
    quantity: str = "p"  # "p" (MW) or "q" (Mvar)

    def __post_init__(self) -> None:
        if (isinstance(self.bus, bool) or type(self.bus).__name__ == "bool_"
                or not (isinstance(self.bus, int) or hasattr(self.bus, "__index__"))
                or int(self.bus) < 0):
            raise ValueError(f"bus must be a non-negative integer, got {self.bus!r}")
        object.__setattr__(self, "bus", int(self.bus))
        if self.quantity not in ("p", "q"):
            raise ValueError(f"quantity must be 'p' or 'q', got {self.quantity!r}")

    @property
    def unit(self) -> str:
        return "MW" if self.quantity == "p" else "Mvar"

    @property
    def label(self) -> str:
        """Compact, filename-safe identifier such as ``b3p``."""
        return f"b{self.bus}{self.quantity}"

    @property
    def pretty(self) -> str:
        """Axis label for plots, e.g. ``P @ bus 3 (MW)``."""
        return f"{self.quantity.upper()} @ bus {self.bus} ({self.unit})"

    @classmethod
    def parse(cls, spec: AxisSpec) -> "RegionAxis":
        """Accept ``3``, ``"3"``, ``"3p"``, ``"3q"``, ``"b3p"``, ``(3, "q")`` or an axis.

        The ``"b3p"`` form is what :attr:`label` emits, so a saved region round-trips.
        """
        if isinstance(spec, RegionAxis):
            return spec
        if isinstance(spec, bool) or type(spec).__name__ == "bool_":
            raise ValueError(f"cannot parse axis {spec!r}; bus index cannot be a boolean")
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError(f"axis tuple must be (bus, quantity), got {spec!r}")
            bus, quantity = spec
            if (isinstance(bus, bool) or type(bus).__name__ == "bool_"
                    or not (isinstance(bus, int) or hasattr(bus, "__index__"))):
                raise ValueError(f"axis bus must be an integer, got {bus!r}")
            return cls(int(bus), str(quantity).lower())
        if isinstance(spec, (int,)) or (hasattr(spec, "__index__") and not isinstance(spec, (str, bytes))):
            return cls(int(spec), "p")
        if not isinstance(spec, str):
            raise ValueError(
                f"cannot parse axis {spec!r}; expected forms are 3, '3', '3p', '3q', 'b3p'"
            )
        match = _AXIS_PATTERN.match(spec.strip())
        if not match:
            raise ValueError(
                f"cannot parse axis {spec!r}; expected forms are 3, '3', '3p', '3q', 'b3p'"
            )
        return cls(int(match.group("bus")), (match.group("quantity") or "p").lower())


class RegionAxes(Sequence[RegionAxis]):
    """An ordered, duplicate-free list of :class:`RegionAxis`.

    Behaves like a tuple, so ``len(axes)`` is the region dimension and
    ``axes[0]`` is the first coordinate.
    """

    def __init__(self, axes: Sequence[AxisSpec]) -> None:
        if isinstance(axes, (str, bytes, bytearray)):
            raise TypeError(
                "RegionAxes expects a sequence of axis specifications, not str or bytes"
            )
        parsed = [RegionAxis.parse(a) for a in axes]
        if not parsed:
            raise ValueError("a region needs at least one axis")
        seen = set()
        for axis in parsed:
            if axis in seen:
                raise ValueError(f"duplicate axis {axis.label} in {[a.label for a in parsed]}")
            seen.add(axis)
        self._axes: Tuple[RegionAxis, ...] = tuple(parsed)

    # ------------------------------------------------------- Sequence protocol
    def __getitem__(self, item):
        if isinstance(item, slice):
            return RegionAxes(self._axes[item])
        if isinstance(item, bool) or type(item).__name__ == "bool_":
            raise TypeError("RegionAxes indices must be integers or slices, not bool")
        return self._axes[item]

    def __len__(self) -> int:
        return len(self._axes)

    def __iter__(self) -> Iterator[RegionAxis]:
        return iter(self._axes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RegionAxes) and self._axes == other._axes

    def __hash__(self) -> int:
        return hash(self._axes)

    # ------------------------------------------------------------- convenience
    @property
    def dim(self) -> int:
        return len(self._axes)

    @property
    def buses(self) -> List[int]:
        return [a.bus for a in self._axes]

    @property
    def labels(self) -> List[str]:
        return [a.label for a in self._axes]

    @property
    def pretty_labels(self) -> List[str]:
        return [a.pretty for a in self._axes]

    def label(self) -> str:
        """Joined label used in output filenames, e.g. ``b1p-b3p-b6p``."""
        return "-".join(self.labels)

    def index_of(self, bus: int, quantity: str = "p") -> int:
        """Position of a given ``(bus, quantity)`` axis, or ``-1`` if absent."""
        target = RegionAxis(bus, quantity)
        for k, axis in enumerate(self._axes):
            if axis == target:
                return k
        return -1

    def __repr__(self) -> str:
        return f"RegionAxes({self.labels})"
