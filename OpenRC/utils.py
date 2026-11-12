"""Cross-cutting utilities for logging, wall-clock timing, and memory guards."""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterator, List

LOGGER = logging.getLogger("OpenRC")


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stream handler to the ``OpenRC`` logger (idempotent)."""
    if isinstance(level, bool) or type(level).__name__ == "bool_":
        raise ValueError(f"logging level must not be a boolean, got {level!r}")
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


# --------------------------------------------------------------------- timing
@dataclass
class Stopwatch:
    """Accumulates wall-clock time across several ``with`` blocks.

    Used to track cumulative runtime for distinct phases such as oracle queries
    and convex-hull updates.
    """

    label: str = ""
    elapsed: float = 0.0
    n_calls: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if (isinstance(self.elapsed, bool) or type(self.elapsed).__name__ == "bool_"
                or not isinstance(self.elapsed, (int, float, Real))
                or not math.isfinite(float(self.elapsed)) or float(self.elapsed) < 0.0):
            raise ValueError("elapsed must be a finite non-negative number")
        if (isinstance(self.n_calls, bool) or type(self.n_calls).__name__ == "bool_"
                or not isinstance(self.n_calls, Integral) or self.n_calls < 0):
            raise ValueError("n_calls must be a non-negative integer")
        self.elapsed = float(self.elapsed)
        self.n_calls = int(self.n_calls)

    @contextmanager
    def __call__(self) -> Iterator["Stopwatch"]:
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.elapsed += time.perf_counter() - start
            self.n_calls += 1

    def reset(self) -> None:
        self.elapsed = 0.0
        self.n_calls = 0

    def __str__(self) -> str:
        return f"{self.label or 'timer'}: {self.elapsed:.3f} s over {self.n_calls} calls"


@contextmanager
def timed(label: str, *, log: bool = True) -> Iterator[List[float]]:
    """``with timed("build") as t: ...`` leaves the duration in ``t[0]``."""
    if not isinstance(label, str):
        raise TypeError("label must be a string")
    if not isinstance(log, bool) and type(log).__name__ != "bool_":
        raise ValueError(f"log must be a boolean, got {log!r}")
    holder: List[float] = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
        if log:
            LOGGER.info("%s took %.3f s", label, holder[0])


# --------------------------------------------------------------------- memory
# Memory management lives in OpenRC.resources; re-exported here for convenience.
from OpenRC.resources import MemoryBudgetExceeded, MemoryGuard  # noqa: E402,F401
