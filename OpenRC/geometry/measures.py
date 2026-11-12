"""Dimension-agnostic geometric measures used by the region algorithms.

Supports arbitrary ambient and simplex dimensions ``1 <= k - 1 <= dim``.
"""

from __future__ import annotations

import math

import numpy as np

_MIN_NORMAL_FLOAT = float(np.finfo(float).tiny)
#: The Gram fast path squares the edges' conditioning: at ``det(G) / prod(diag G)
#: = r`` its relative error is about ``eps / r``.  Below this ratio the unsquared
#: path is used, which keeps every volume to about 1e-12 relative error; thin
#: simplices are common among the facets of high-dimensional hulls.
_GRAM_SAFE_RATIO = 1e-4


def simplex_volume(points: np.ndarray) -> float:
    """``(k-1)``-volume of the simplex spanned by ``k`` points in ``R^n``.

    Valid for any ``k <= n + 1`` and does not need the points to be
    full-dimensional in ``R^n``.  In 3-D with three points this is the triangle
    area; in 8-D with eight points it is the 7-volume of the facet.

    Well-conditioned inputs retain the ordinary Gram-determinant path.  Thin
    or extreme-scale simplices use row scaling and unsquared edge factors
    instead: forming a Gram matrix squares the condition number (relative
    error about ``eps / r`` at ``r = det(G) / prod(diag G)``, so only
    ``r > 1e-4`` takes it) and can overflow even when the final volume is
    representable.  Numerically rank-deficient inputs return ``0.0``;
    non-finite coordinates are refused.
    """
    raw_pts = np.asarray(points)
    if raw_pts.dtype == bool:
        raise ValueError("simplex coordinates must be a numeric array, not boolean")
    pts = np.asarray(raw_pts, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (k, dim)")
    if pts.shape[0] < 2:
        if not np.all(np.isfinite(pts)):
            raise ValueError("simplex coordinates must contain only finite values")
        return 0.0
    k = pts.shape[0]
    rank = k - 1
    if rank > pts.shape[1]:
        if not np.all(np.isfinite(pts)):
            raise ValueError("simplex coordinates must contain only finite values")
        return 0.0
    # Fast path via Gram determinant; falls back to SVD in log-space on overflow,
    # underflow, or high conditioning.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        vectors = pts[1:] - pts[0]          # (k-1, n)
        gram = vectors @ vectors.T
        det = float(np.linalg.det(gram))
    hadamard = math.prod(float(d) for d in gram.diagonal())
    if (rank <= 170 and math.isfinite(det) and det >= _MIN_NORMAL_FLOAT
            and math.isfinite(hadamard) and hadamard > 0.0
            and det > _GRAM_SAFE_RATIO * hadamard):
        # Finite positive Hadamard product implies finite edge vectors; subnormal
        # determinants fall back to _unsquared_volumes to preserve relative precision.
        return math.sqrt(det) / math.factorial(rank)

    return float(_unsquared_volumes(pts[None, :, :])[0])


def _unsquared_volumes(pts: np.ndarray) -> np.ndarray:
    """Volumes of a stack ``(n, k, dim)`` of simplices, without a Gram matrix.

    Each edge is scaled to unit max-norm and measured through its singular
    values (numerically rank-deficient rows give ``0.0``).  The logarithm avoids
    an unrepresentable intermediate product; a volume beyond float range is
    ``inf``.  Non-finite coordinates or edge differences are refused.
    """
    n, k = pts.shape[0], pts.shape[1]
    with np.errstate(over="ignore", invalid="ignore"):
        vectors = pts[:, 1:] - pts[:, :1]                    # (n, rank, dim)
    if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(vectors)):
        raise ValueError("simplex coordinates and their differences must be finite")
    out = np.zeros(n)
    scales = np.max(np.abs(vectors), axis=2)                 # (n, rank)
    live = np.all(scales > 0.0, axis=1)
    if not live.any():
        return out
    # Underflow to zero is the right answer; overflow means an unrepresentable
    # volume (inf).
    with np.errstate(under="ignore"):
        edges = vectors[live] / scales[live][:, :, None]
        singular = np.linalg.svd(edges, compute_uv=False)
        flat = (singular[:, -1]
                <= np.finfo(float).eps * max(edges.shape[1:]) * singular[:, 0])
        with np.errstate(divide="ignore"):
            log_measure = np.log(singular).sum(axis=1)
        log_volume = log_measure + np.log(scales[live]).sum(axis=1) - math.lgamma(k)
        with np.errstate(over="ignore"):
            values = np.exp(log_volume)                      # inf: unrepresentable
    values[flat] = 0.0
    out[live] = values
    return out


def simplex_volumes(points: np.ndarray, *, max_working_mb: float = 16.0) -> np.ndarray:
    """Measure a stack of simplices, with bounded temporary arrays.

    ``points`` has shape ``(n, k, dim)`` and the result has shape ``(n,)``.
    Ordinary Gram calculations are vectorised; unsafe or poorly conditioned
    rows use :func:`simplex_volume` individually, including its strict finite
    checks and scaled fallback.  The soft budget excludes input/output storage;
    one simplex is the minimum working block and can exceed it.  Rows the
    Gram path would not keep to about 1e-12 are measured together, unsquared.
    """
    if (isinstance(max_working_mb, (bool, np.bool_))
            or not math.isfinite(max_working_mb) or max_working_mb <= 0.0):
        raise ValueError("max_working_mb must be finite and strictly positive")
    max_working_mb = float(max_working_mb)
    raw_pts = np.asarray(points)
    if raw_pts.dtype == bool:
        raise ValueError("simplex coordinates must be a numeric array, not boolean")
    pts = np.asarray(raw_pts, dtype=float)
    if pts.ndim != 3:
        raise ValueError("points must have shape (n, k, dim)")
    n, k, dim = pts.shape
    out = np.empty(n, dtype=float)
    rank = k - 1
    if n == 0:
        return out
    # Chunk rows to bound temporary edge and Gram arrays within max_working_mb.
    per_row = 8 * max(1, 3 * k * dim + 2 * max(rank, 0) ** 2 + 8 * k)
    rows = max(1, int(min(n, max_working_mb * 2 ** 20 / per_row)))
    for first in range(0, n, rows):
        chunk = pts[first:first + rows]
        if k < 2 or rank > dim:
            if not np.all(np.isfinite(chunk)):
                raise ValueError("simplex coordinates must contain only finite values")
            out[first:first + len(chunk)] = 0.0
            continue
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            edges = chunk[:, 1:] - chunk[:, :1]
            gram = edges @ edges.swapaxes(1, 2)
            det = np.linalg.det(gram)
            hadamard = np.prod(gram.diagonal(axis1=1, axis2=2), axis=1)
            safe = (np.isfinite(det) & (det >= _MIN_NORMAL_FLOAT)
                    & np.isfinite(hadamard) & (hadamard > 0.0)
                    & (det > _GRAM_SAFE_RATIO * hadamard) & (rank <= 170))
        values = out[first:first + len(chunk)]
        if np.any(safe):
            values[safe] = np.sqrt(det[safe]) / float(math.factorial(rank))
        if not np.all(safe):
            values[~safe] = _unsquared_volumes(chunk[~safe])
    return out


def cone_volume(facet_points: np.ndarray, apex: np.ndarray | None = None) -> float:
    """Volume of the cone spanned by a facet and an apex (the origin by default).

    Computes ``V(C(F, apex)) = |det([F - apex])| / d!``, combining
    ``(d-1)``-dimensional facet area with orthogonal distance from ``apex``.
    """
    raw_pts = np.asarray(facet_points)
    if raw_pts.dtype == bool:
        raise ValueError("facet_points must be a numeric array, not boolean")
    pts = np.asarray(raw_pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] == 0:
        raise ValueError(f"facet_points must be a 2D array with positive dimension, got shape {pts.shape}")
    if apex is None:
        apex_pt = np.zeros(pts.shape[1])
    else:
        raw_apex = np.asarray(apex)
        if raw_apex.dtype == bool:
            raise ValueError("apex must be a numeric vector, not boolean")
        apex_pt = np.asarray(raw_apex, dtype=float)
        if apex_pt.ndim != 1 or apex_pt.shape != (pts.shape[1],) or not np.all(np.isfinite(apex_pt)):
            raise ValueError(
                f"apex must be a finite 1D vector of length {pts.shape[1]}, got shape {apex_pt.shape}"
            )
    return simplex_volume(np.vstack([pts, apex_pt]))


def hull_volume(points: np.ndarray) -> float:
    """Hypervolume of the convex hull of ``points`` (``0.0`` if degenerate)."""
    from scipy.spatial import ConvexHull, QhullError

    from OpenRC.geometry.live_hull import _is_flat
    from OpenRC.geometry.polytope import _hull_geometry_consistent

    raw_pts = np.asarray(points)
    if raw_pts.dtype == bool:
        raise ValueError("points must be a numeric array, not boolean")
    pts = np.asarray(raw_pts, dtype=float).copy()
    if pts.ndim != 2 or pts.shape[0] <= pts.shape[1] or not np.all(np.isfinite(pts)):
        return 0.0
    if pts.shape[1] == 1:
        return float(np.ptp(pts[:, 0]))
    for qhull_options in (None, "Qx", "Qt", "Qx Q12", "Q14"):
        try:
            hull = ConvexHull(pts, qhull_options=qhull_options)
        except QhullError as exc:
            if _is_flat(exc):
                return 0.0
            continue
        except ValueError:
            continue
        if qhull_options is not None and not _hull_geometry_consistent(hull, pts):
            continue
        return float(hull.volume)
    # Coplanar / duplicate inputs raise; a flat body has zero volume.
    return 0.0


def volume_gain(points: np.ndarray, candidate: np.ndarray) -> float:
    """Hypervolume added by appending ``candidate`` to ``points``.

    This is the greedy ``Delta V(R)`` facet-selection measure and, with the sign
    flipped, the marginal loss used by every vertex-pruning method.
    """
    raw_pts = np.asarray(points)
    raw_cand = np.asarray(candidate)
    if raw_pts.dtype == bool or raw_cand.dtype == bool:
        raise ValueError("points and candidate must be numeric arrays, not boolean")
    pts = np.asarray(raw_pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] == 0 or not np.all(np.isfinite(pts)):
        raise ValueError("points must be a finite 2D array with positive dimension")
    cand = np.asarray(raw_cand, dtype=float)
    if cand.ndim != 1 or cand.shape != (pts.shape[1],) or not np.all(np.isfinite(cand)):
        raise ValueError(
            f"candidate must be a finite 1D vector of length {pts.shape[1]}, got shape {cand.shape}"
        )
    before = hull_volume(pts)
    after = hull_volume(np.vstack([pts, cand]))
    return max(0.0, after - before)


def bounding_box(points: np.ndarray, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around ``points``, grown by ``margin`` on every side.

    The validation procedure of the conservative-region paper draws its test
    points from exactly this box with a 2 MW margin, so that the sample covers
    both the derived region and a shell of space around it.
    """
    if isinstance(margin, (bool, np.bool_)) or not math.isfinite(float(margin)):
        raise ValueError(f"margin must be a finite number, got {margin!r}")
    raw_input = np.asarray(points)
    if raw_input.dtype == bool:
        raise ValueError("points must be a numeric array, not boolean")
    raw = np.asarray(raw_input, dtype=float)
    if raw.size == 0 or raw.ndim not in (1, 2) or not np.all(np.isfinite(raw)):
        raise ValueError("points must be a non-empty 1D or 2D array of finite coordinates")
    pts = np.atleast_2d(raw)
    lower = pts.min(axis=0) - float(margin)
    upper = pts.max(axis=0) + float(margin)
    return lower, upper
