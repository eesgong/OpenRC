"""Vectorised interval arithmetic with outward rounding, for rigorous certificates.

IEEE 754 returns the correctly rounded result of ``+``, ``-``, ``*``, ``/``
and ``sqrt``: the exact value lies within half an ulp of what is computed.
Moving every computed bound one ulp outward (``numpy.nextafter``) therefore
gives an interval that provably contains the exact result.  Chaining such
operations encloses any expression built from them, rigorously enclosing the
real-valued result against IEEE 754 rounding error.

NumPy evaluates each ufunc as one IEEE operation per element (no fused
multiply-add, no reassociation).  Multi-term reductions are accumulated
explicitly in :class:`TreeSums` with outward rounding after every binary
addition rather than delegated to ``numpy.sum`` or sparse matrix multiplication.

Only what the radial sweep certificate needs is provided: real intervals,
complex rectangles, and the two tree sums of a radial network.

The constructor rejects inverted bounds and NaN.  Results of the operations
below are ordered by construction and are not checked again (a certificate is
dominated by these small operations); a NaN produced inside an expression, from
an overflow such as ``inf - inf``, propagates, so a caller drawing a conclusion
must reject NaN, as :class:`~OpenRC.powerflow.sweep.RadialSweep` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np

Number = Union[float, np.ndarray]


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _extremes(a, b, c, d):
    """Outward-rounded ``[min, max]`` of four candidate endpoints."""
    return (_down(np.minimum(np.minimum(a, b), np.minimum(c, d))),
            _up(np.maximum(np.maximum(a, b), np.maximum(c, d))))


@dataclass(frozen=True)
class Interval:
    """Element-wise intervals ``[lo, hi]`` containing the exact values."""

    lo: np.ndarray
    hi: np.ndarray
    #: Disable NumPy ufunc dispatch so ``ndarray <op> Interval`` delegates to ``__r*__``.
    __array_ufunc__ = None

    @classmethod
    def point(cls, x: Number) -> "Interval":
        """Exact floating-point values (the interval of width zero)."""
        raw = np.asarray(x)
        if raw.dtype == bool:
            raise ValueError("interval point values must be numeric, not boolean")
        a = np.asarray(raw, dtype=float)
        return cls(a.copy(), a.copy())

    def __post_init__(self) -> None:
        raw_lo = np.asarray(self.lo)
        raw_hi = np.asarray(self.hi)
        if raw_lo.dtype == bool or raw_hi.dtype == bool:
            raise ValueError("interval bounds must be numeric, not boolean")
        lo = np.asarray(raw_lo, dtype=float)
        hi = np.asarray(raw_hi, dtype=float)
        if lo.shape != hi.shape:
            raise ValueError(f"interval bounds must have the same shape; got {lo.shape} and {hi.shape}")
         # NaN or inverted bounds both fail <=
        if not np.all(lo <= hi):
            raise ValueError("an interval needs lo <= hi (a NaN or an inverted bound)")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def _unchecked(cls, lo: np.ndarray, hi: np.ndarray) -> "Interval":
        """An operation's result, whose bounds are ordered by construction."""
        result = object.__new__(cls)
        object.__setattr__(result, "lo", lo)
        object.__setattr__(result, "hi", hi)
        return result

    @staticmethod
    def _of(x: Union["Interval", Number]) -> "Interval":
        return x if isinstance(x, Interval) else Interval.point(x)

    def __getitem__(self, index) -> "Interval":
        return Interval._unchecked(self.lo[index], self.hi[index])

    def __add__(self, other) -> "Interval":
        o = self._of(other)
        return Interval._unchecked(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval._unchecked(-self.hi, -self.lo)

    def __sub__(self, other) -> "Interval":
        o = self._of(other)
        return Interval._unchecked(_down(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, other) -> "Interval":
        return self._of(other) - self

    def __mul__(self, other) -> "Interval":
        o = self._of(other)
        return Interval._unchecked(*_extremes(self.lo * o.lo, self.lo * o.hi,
                                              self.hi * o.lo, self.hi * o.hi))

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Interval":
        o = self._of(other)
        if np.any((o.lo <= 0.0) & (o.hi >= 0.0)):
            raise ZeroDivisionError("interval division by an interval containing 0")
        return Interval._unchecked(*_extremes(self.lo / o.lo, self.lo / o.hi,
                                              self.hi / o.lo, self.hi / o.hi))

    def __rtruediv__(self, other) -> "Interval":
        return self._of(other) / self

    def square(self) -> "Interval":
        """``x^2``, tight at 0 (``[-1, 2]^2 = [0, 4]``)."""
        a, b = self.lo * self.lo, self.hi * self.hi
        straddles = (self.lo <= 0.0) & (self.hi >= 0.0)
        low = np.where(straddles, 0.0, _down(np.minimum(a, b)))
        return Interval._unchecked(np.maximum(low, 0.0), _up(np.maximum(a, b)))

    def sqrt(self) -> "Interval":
        if np.any(self.lo < 0.0):
            raise ValueError("sqrt of an interval reaching below 0")
        return Interval._unchecked(np.maximum(_down(np.sqrt(self.lo)), 0.0),
                                   _up(np.sqrt(self.hi)))

    def magnitude_hi(self) -> np.ndarray:
        """Upper bound of ``|x|``."""
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def magnitude_lo(self) -> np.ndarray:
        """Lower bound of ``|x|`` (0 when the interval contains 0)."""
        straddles = (self.lo <= 0.0) & (self.hi >= 0.0)
        return np.where(straddles, 0.0, np.minimum(np.abs(self.lo), np.abs(self.hi)))


@dataclass(frozen=True)
class ComplexInterval:
    """Rectangles ``re + i im`` of the complex plane."""

    re: Interval
    im: Interval
    __array_ufunc__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.re, Interval) or not isinstance(self.im, Interval):
            raise TypeError("ComplexInterval components re and im must be Interval instances")
        if self.re.lo.shape != self.im.lo.shape:
            raise ValueError(
                f"ComplexInterval real and imaginary parts must have the same shape; "
                f"got {self.re.lo.shape} and {self.im.lo.shape}"
            )

    @classmethod
    def point(cls, z: np.ndarray) -> "ComplexInterval":
        raw = np.asarray(z)
        if raw.dtype == bool:
            raise ValueError("ComplexInterval point values must be numeric, not boolean")
        z = np.asarray(raw, dtype=complex)
        return cls(Interval.point(z.real), Interval.point(z.imag))

    def __getitem__(self, index) -> "ComplexInterval":
        return ComplexInterval(self.re[index], self.im[index])

    def __add__(self, other: "ComplexInterval") -> "ComplexInterval":
        if not isinstance(other, ComplexInterval):
            raise TypeError("ComplexInterval operations require ComplexInterval operands")
        return ComplexInterval(self.re + other.re, self.im + other.im)

    def __sub__(self, other: "ComplexInterval") -> "ComplexInterval":
        if not isinstance(other, ComplexInterval):
            raise TypeError("ComplexInterval operations require ComplexInterval operands")
        return ComplexInterval(self.re - other.re, self.im - other.im)

    def conj(self) -> "ComplexInterval":
        return ComplexInterval(self.re, -self.im)

    def __mul__(self, other: "ComplexInterval") -> "ComplexInterval":
        if not isinstance(other, ComplexInterval):
            raise TypeError("ComplexInterval operations require ComplexInterval operands")
        return ComplexInterval(self.re * other.re - self.im * other.im,
                               self.re * other.im + self.im * other.re)

    def __truediv__(self, other: "ComplexInterval") -> "ComplexInterval":
        """``(a + ib) / (c + id) = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)``."""
        if not isinstance(other, ComplexInterval):
            raise TypeError("ComplexInterval operations require ComplexInterval operands")
        den = other.re.square() + other.im.square()
        return ComplexInterval((self.re * other.re + self.im * other.im) / den,
                               (self.im * other.re - self.re * other.im) / den)

    @staticmethod
    def _modulus(re: np.ndarray, im: np.ndarray) -> Interval:
        """``sqrt(re^2 + im^2)``; the sum of squares is exactly >= 0, so its
        lower bound may be clamped there (rounding 0 + 0 down gives -5e-324)."""
        total = Interval.point(re).square() + Interval.point(im).square()
        return Interval._unchecked(np.maximum(total.lo, 0.0), total.hi).sqrt()

    def abs_hi(self) -> np.ndarray:
        """Upper bound of the modulus over the rectangle."""
        return self._modulus(self.re.magnitude_hi(), self.im.magnitude_hi()).hi

    def abs_lo(self) -> np.ndarray:
        """Lower bound of the modulus over the rectangle."""
        return self._modulus(self.re.magnitude_lo(), self.im.magnitude_lo()).lo


class TreeSums:
    """The two sums of a radial network, one outward-rounded addition at a time.

    ``subtree(x)`` gives, per branch, the sum of ``x`` over the buses below it
    (``T x``); ``path(y)`` gives, per bus, the sum of ``y`` over the branches on
    its path to the slack (``T^T y``).  Trailing axes are independent columns
    summed in the same pass (``x`` of shape ``(n_bus, k)`` gives
    ``(n_branch, k)``), which saves the per-level overhead when several sums
    are needed.  Branches are processed level by level from the leaves
    (``subtree``) or from the slack (``path``).  Within a level, siblings are
    split by rank, preventing concurrent in-place updates to the same parent
    and ensuring outward rounding after each binary addition.

    Setup groups only the depth/rank pairs that exist: storage is linear in the
    number of branches and sorting costs O(n_branch log n_branch), even on a
    deep feeder with a wide terminal fan.
    """

    def __init__(self, topology) -> None:
        if not (hasattr(topology, "branches") and hasattr(topology, "n_bus")):
            raise TypeError(f"topology must be a RadialTopology; got {type(topology).__name__}")
        branches = list(topology.branches)
        self.n_branch = len(branches)
        self.n_bus = topology.n_bus
        self.child = np.array([br.child for br in branches], dtype=int)
        self.parent = np.array([br.parent for br in branches], dtype=int)
        row_of_child = {br.child: k for k, br in enumerate(branches)}
        parent_row = np.array([row_of_child.get(br.parent, -1) for br in branches], dtype=int)
        # Verify tree size, index bounds, and parent-before-child ordering.
        if (self.n_branch != self.n_bus - 1 or len(row_of_child) != self.n_branch
                or np.any((self.child < 0) | (self.child >= self.n_bus))
                or np.any((self.parent < 0) | (self.parent >= self.n_bus))
                or np.any(parent_row >= np.arange(self.n_branch))):
            raise ValueError("interval sums require a complete parent-before-child tree")
        depth = np.zeros(self.n_bus, dtype=int)
        seen: dict = {}
        up_groups: dict[Tuple[int, int], List[int]] = {}
        down_groups: dict[int, List[int]] = {}
        for k, br in enumerate(branches):
            # RadialTopology supplies a parent-before-child branch order.
            # Keep branch order within each group: interval additions must
            # preserve their original outward-rounded evaluation order.
            d = int(depth[br.parent]) + 1
            depth[br.child] = d
            rank = seen.get(br.parent, 0)
            seen[br.parent] = rank + 1
            down_groups.setdefault(d, []).append(k)
            if d > 1:
                up_groups.setdefault((d, rank), []).append(k)
        self._up: List[Tuple[np.ndarray, np.ndarray]] = []
        for key in sorted(up_groups, key=lambda pair: (-pair[0], pair[1])):
            rows = np.array(up_groups[key], dtype=int)
            self._up.append((rows, parent_row[rows]))
        self._down = [np.array(down_groups[d], dtype=int) for d in sorted(down_groups)]

    def subtree(self, at_bus: Interval) -> Interval:
        """Per branch, the enclosure of the sum of ``at_bus`` over its subtree."""
        if not isinstance(at_bus, Interval):
            raise TypeError("at_bus must be an Interval")
        if at_bus.lo.ndim < 1 or at_bus.lo.shape[0] != self.n_bus:
            raise ValueError(
                f"at_bus must have leading dimension {self.n_bus}; got shape {at_bus.lo.shape}"
            )
        lo = at_bus.lo[self.child]                     # fancy indexing copies
        hi = at_bus.hi[self.child]
        for rows, parents in self._up:
            lo[parents] = _down(lo[parents] + lo[rows])
            hi[parents] = _up(hi[parents] + hi[rows])
        return Interval._unchecked(lo, hi)

    def path(self, per_branch: Interval) -> Interval:
        """Per bus, the enclosure of the sum of ``per_branch`` along its path."""
        if not isinstance(per_branch, Interval):
            raise TypeError("per_branch must be an Interval")
        if per_branch.lo.ndim < 1 or per_branch.lo.shape[0] != self.n_branch:
            raise ValueError(
                f"per_branch must have leading dimension {self.n_branch}; "
                f"got shape {per_branch.lo.shape}"
            )
        shape = (self.n_bus,) + per_branch.lo.shape[1:]
        lo = np.zeros(shape)
        hi = np.zeros(shape)
        for rows in self._down:
            lo[self.child[rows]] = _down(lo[self.parent[rows]] + per_branch.lo[rows])
            hi[self.child[rows]] = _up(hi[self.parent[rows]] + per_branch.hi[rows])
        return Interval._unchecked(lo, hi)
