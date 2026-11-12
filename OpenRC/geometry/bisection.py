"""Ray-boundary search via bracketed bisection and ITP margin interpolation.

Given an interior origin and a direction in region coordinates,
:func:`ray_boundary` locates the last feasible point along the ray:

* **Bracketing.** Doubles the step from ``initial_ratio`` up to ``max_ratio``
  until an infeasible exterior point is found (raising :class:`UnboundedRayError`
  if ``require_bracket=True`` and none is found).
* **Origin verification.** When ``check_origin=True``, verifies that ``origin``
  is feasible before searching and raises :class:`InfeasibleOriginError` otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RayResult:
    """Outcome of one boundary search."""

    boundary: np.ndarray        # last point certified feasible
    exterior: np.ndarray        # first point certified infeasible
    ratio: float                # distance along the unit ray at the boundary
    n_oracle_calls: int
    bracketed: bool             # False if the bracket hit max_ratio while feasible

    @property
    def width(self) -> float:
        """Remaining bracket width in region units - the achieved accuracy."""
        return math.hypot(*(self.exterior - self.boundary))


def ray_boundary(
    oracle,
    axes,
    direction: np.ndarray,
    *,
    origin: Optional[np.ndarray] = None,
    initial_ratio: float = 1.0,
    max_ratio: float = 1e6,
    tol: float = 1e-4,
    max_iter: int = 60,
    require_bracket: bool = True,
    hint: Optional[float] = None,
    hint_width: float = 0.15,
    check_origin: bool = True,
    search: str = "auto",
) -> RayResult:
    """Find where the unit ray ``origin + t * (direction / ||direction||)`` leaves the feasible set.

    Parameters
    ----------
    oracle:
        A :class:`~OpenRC.powerflow.base.FeasibilityOracle`.
    axes:
        Region axes, so the ray can be mapped back to nodal injections.
    direction:
        Non-zero direction vector in region coordinates (normalised internally).
    origin:
        Start of the ray; defaults to the origin of region coordinates, i.e. the
        case's base operating point.
    initial_ratio:
        First bracket attempt.  Doubled until infeasible or ``max_ratio``.
    max_ratio:
        Largest distance that may be probed while looking for an exterior point.
    tol:
        Stop once the bracket is narrower than this in region units (MW).
    max_iter:
        Hard cap on bisection steps.
    require_bracket:
        Raise :class:`UnboundedRayError` when no exterior point is found by
        ``max_ratio``.  If ``False``, returns the point at ``max_ratio`` with
        ``bracketed=False`` when no exterior point is found.
    hint:
        Optional boundary-distance estimate (e.g. from an adjacent vertex).
        Initialises the bracket around ``hint * (1 +/- hint_width)`` to skip
        outward doubling when the boundary lies inside that interval; falls
        back to doubling or ``[0, hint * (1 - hint_width)]`` otherwise.
    check_origin:
        Verify feasibility of ``origin`` before probing along the ray (one
        oracle call).  Set ``False`` when ``origin`` has already been verified
        feasible.
    search:
        Bracket-refinement method: ``"bisect"`` (plain bisection) or
        ``"margin"`` (ITP interpolation on the signed slack returned by
        :meth:`~OpenRC.powerflow.base.FeasibilityOracle.margin_at`, Oliveira &
        Takahashi 2020), achieving superlinear convergence on smooth margins
        while guaranteeing at most ``ceil(log2(width0 / tol)) + 1`` evaluations.
        ``"auto"`` (default) selects ``"margin"`` when
        ``oracle.margin_available()`` and ``oracle.prefers_margin_search`` are
        both ``True``, and ``"bisect"`` otherwise.

    Returns
    -------
    RayResult
    """
    if not (hasattr(oracle, "feasible_at") or hasattr(oracle, "margin_at")):
        raise TypeError("oracle must provide feasible_at or margin_at")
    raw_dir = np.asarray(direction)
    if raw_dir.dtype == bool:
        raise ValueError("direction must be a numeric vector, not boolean")
    direction = np.asarray(raw_dir, dtype=float)
    if direction.shape != (len(axes),):
        raise ValueError(f"direction has shape {direction.shape}, expected ({len(axes)},)")
    if not np.all(np.isfinite(direction)):
        raise ValueError("direction must contain only finite values")
    scale = float(np.max(np.abs(direction), initial=0.0))
    if scale == 0.0:
        raise ValueError("direction must be a non-zero vector")
    # Squaring a finite direction can overflow or underflow in np.linalg.norm.
    # Ordinary inputs take the plain path; only extreme ones are rescaled.
    info = np.finfo(float)
    if scale < math.sqrt(info.tiny) or scale > math.sqrt(info.max / len(direction)):
        scaled = direction / scale
        unit = scaled / np.linalg.norm(scaled)
    else:
        unit = direction / np.linalg.norm(direction)
    if origin is None:
        origin = np.zeros(len(axes), dtype=float)
    else:
        raw_orig = np.asarray(origin)
        if raw_orig.dtype == bool:
            raise ValueError("origin must be a numeric vector, not boolean")
        origin = np.asarray(raw_orig, dtype=float)
    if origin.shape != (len(axes),) or not np.all(np.isfinite(origin)):
        raise ValueError(f"origin must be a finite vector with shape ({len(axes)},)")
    if isinstance(initial_ratio, (bool, np.bool_)) or not np.isfinite(initial_ratio) or initial_ratio <= 0.0:
        raise ValueError("initial_ratio must be finite and strictly positive")
    if isinstance(max_ratio, (bool, np.bool_)) or not np.isfinite(max_ratio) or max_ratio < initial_ratio:
        raise ValueError("max_ratio must be finite and at least initial_ratio")
    if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and strictly positive")
    if (not isinstance(max_iter, (int, np.integer))
            or isinstance(max_iter, (bool, np.bool_)) or max_iter <= 0):
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(require_bracket, (bool, np.bool_)):
        raise ValueError(f"require_bracket must be a boolean, got {require_bracket!r}")
    if not isinstance(check_origin, (bool, np.bool_)):
        raise ValueError(f"check_origin must be a boolean, got {check_origin!r}")
    if hint is not None and (isinstance(hint, (bool, np.bool_)) or not (np.isfinite(hint) and hint > 0.0)):
        hint = None                      # an unusable guess is simply ignored
    if hint is not None and hint > max_ratio:
        hint = None                      # no hint may bypass the hard search cap
    if isinstance(hint_width, (bool, np.bool_)) or not (np.isfinite(hint_width) and 0.0 < hint_width < 1.0):
        raise ValueError("hint_width must lie in (0, 1)")
    if search not in ("auto", "bisect", "margin"):
        raise ValueError("search must be 'auto', 'bisect' or 'margin'")
    available = getattr(oracle, "margin_available", None)
    supported = (available() if callable(available)
                 else bool(getattr(oracle, "supports_margin", False)))
    use_margin = supported and (
        search == "margin"
        or (search == "auto" and bool(getattr(oracle, "prefers_margin_search", False))))
    if use_margin and not hasattr(oracle, "margin_at"):
        raise TypeError("oracle must provide a margin_at(axes, point) method")
    if not use_margin and not hasattr(oracle, "feasible_at"):
        raise TypeError("oracle must provide a feasible_at(axes, point) method")

    # A ray's result must not depend on the rays searched before it (serial and
    # parallel runs visit rays in different orders), so warm-start state from
    # earlier queries is dropped here.
    reset = getattr(oracle, "reset_warm_state", None)
    if callable(reset):
        reset()
    calls = 0
    margin_at = {}                        # t -> the oracle's margin there

    def probe(t: float) -> Tuple[bool, float]:
        nonlocal calls
        calls += 1
        x = origin + t * unit
        if use_margin:
            ok, margin = oracle.margin_at(axes, x)
            return ok, float(margin)
        return oracle.feasible_at(axes, x), math.nan

    def feasible(t: float) -> bool:
        ok, margin_at[t] = probe(t)
        return ok

    if check_origin and not feasible(0.0):
        raise InfeasibleOriginError(
            "the oracle does not accept the ray origin: it is infeasible there, or it "
            "could not decide (a power flow or OPF that did not converge is "
            "INCONCLUSIVE, which never counts as feasible); "
            "`oracle.verdict_at(axes, origin)` tells which. There is no region to "
            "explore from here. Check the case's base loading, the oracle's limits or "
            "solver options, or pass an `origin` that is known to be feasible."
        )

    # --- grow the bracket until the far end is infeasible ------------------
    lower, upper = 0.0, min(float(initial_ratio), float(max_ratio))
    bracketed = False
    if hint is not None:
        high = min(float(hint) * (1.0 + hint_width), float(max_ratio))
        if not feasible(high):
            low = float(hint) * (1.0 - hint_width)
            if feasible(low):
                lower, upper, bracketed = low, high, True
            else:
                lower, upper, bracketed = 0.0, low, True
        else:
            lower, upper = high, min(2.0 * high, float(max_ratio))
    while not bracketed:
        if upper <= lower:
            break
        if not feasible(upper):
            bracketed = True
            break
        lower = upper
        if upper >= max_ratio:
            break
        upper = min(2.0 * upper, float(max_ratio))
    if not bracketed:
        message = (
            f"no infeasible endpoint was found by max_ratio={max_ratio:g}; "
            "the feasible set may be unbounded in this direction or max_ratio "
            "may be too small"
        )
        if require_bracket:
            raise UnboundedRayError(message)
        point = origin + lower * unit
        return RayResult(point, point.copy(), lower, calls, bracketed=False)

    # --- narrow: bisection, or ITP on the margin ----------------------------
    m_lower = margin_at.get(lower, math.nan)
    m_upper = margin_at.get(upper, math.nan)
    width0 = upper - lower
    # Preserve the ordinary trajectory, without overflowing a valid quotient
    # when a caller asks for subnormal accuracy (max_iter remains the cap).
    n_max = 0
    if width0 >= tol:
        ratio = width0 / tol
        log_ratio = (math.log2(ratio) if math.isfinite(ratio)
                     else math.log2(width0) - math.log2(tol))
        n_max = math.ceil(log_ratio) + 1
    for step in range(max_iter):
        width = upper - lower
        if width < tol:
            break
        middle = (0.5 * (lower + upper) if math.isfinite(lower + upper)
                  else lower + 0.5 * width)
        if middle == lower or middle == upper:
            break                         # no representable point remains inside
        t = middle
        if (use_margin and math.isfinite(m_lower) and math.isfinite(m_upper)
                and m_lower >= 0.0 > m_upper):
            t = _itp_point(lower, upper, m_lower, m_upper, width0, tol, n_max, step)
        ok, margin = probe(t)
        if ok:
            lower, m_lower = t, margin
        else:
            upper, m_upper = t, margin

    if lower == 0.0 and not check_origin and not feasible(0.0):
        # When check_origin=False and lower remains 0.0, verify origin feasibility
        # before returning origin as a certified boundary point.
        raise InfeasibleOriginError(
            "every point probed on the ray was rejected and the ray origin itself is "
            "not accepted by the oracle; there is no certified boundary point")
    return RayResult(
        boundary=origin + lower * unit,
        exterior=origin + upper * unit,
        ratio=lower,
        n_oracle_calls=calls,
        bracketed=True,
    )


def _itp_point(lower: float, upper: float, m_lower: float, m_upper: float,
               width0: float, tol: float, n_max: int, step: int) -> float:
    """The next ITP point in ``(lower, upper)`` (Oliveira & Takahashi, 2020).

    Regula falsi on the margin, truncated towards the midpoint by
    ``0.2 (b - a)^2 / w0`` and projected into the ball around the midpoint that
    keeps the bracket on bisection's schedule - so at most ``n_max`` steps are
    ever needed.
    """
    width = upper - lower
    middle = (0.5 * (lower + upper) if math.isfinite(lower + upper)
              else lower + 0.5 * width)
    product, difference = width * m_lower, m_lower - m_upper
    if math.isfinite(product) and math.isfinite(difference):
        regula = lower + product / difference
    else:
        magnitude = max(m_lower, -m_upper)
        a, b = m_lower / magnitude, m_upper / magnitude
        regula = lower + width * (a / (a - b))
    sigma = 1.0 if middle >= regula else -1.0
    if math.sqrt(np.finfo(float).tiny) <= width <= math.sqrt(np.finfo(float).max):
        delta = 0.2 / width0 * width ** 2
    else:
        delta = 0.2 * (width / width0) * width
    truncated = regula + sigma * delta if delta <= abs(middle - regula) else middle
    if n_max - step <= 1023 and tol >= np.finfo(float).tiny:
        radius = 0.5 * tol * 2.0 ** (n_max - step) - 0.5 * width
    else:
        try:
            radius = math.ldexp(tol, n_max - step - 1) - 0.5 * width
        except OverflowError:
            radius = math.inf             # no projection needed at this step
    t = truncated if abs(truncated - middle) <= radius else middle - sigma * radius
    return t if lower < t < upper else middle


class InfeasibleOriginError(RuntimeError):
    """Raised when the point a region is grown from is already infeasible."""


class UnboundedRayError(RuntimeError):
    """Raised when a ray has no certified exterior endpoint by ``max_ratio``."""


def axis_box(oracle, axes, *, origin: Optional[np.ndarray] = None, **kwargs):
    """Probe the ``2 * dim`` axis directions and return their coordinate box.

    Returns ``(lower, upper, vertices, n_calls)``.  This is the warm-up step of
    the random-acceleration builder: the box it produces is the sampling domain
    for the initial random points, and the ``2 * dim`` boundary points it finds
    are themselves good initial vertices.  Coupled coordinates can permit
    points beyond an axis intercept, so this box need not enclose the full
    feasible region.
    """
    if not (hasattr(oracle, "feasible_at") or hasattr(oracle, "margin_at")):
        raise TypeError("oracle must provide feasible_at or margin_at")
    dim = len(axes)
    if dim < 1:
        raise ValueError("axes must contain at least one axis")
    lower = np.zeros(dim)
    upper = np.zeros(dim)
    vertices = []
    calls = 0
    for k in range(dim):
        for sign in (+1.0, -1.0):
            direction = np.zeros(dim)
            direction[k] = sign
            result = ray_boundary(oracle, axes, direction, origin=origin, **kwargs)
            calls += result.n_oracle_calls
            vertices.append(result.boundary)
            if sign > 0:
                upper[k] = result.boundary[k]
            else:
                lower[k] = result.boundary[k]
    return lower, upper, np.array(vertices), calls
