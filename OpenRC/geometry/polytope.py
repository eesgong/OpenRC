"""The :class:`Polytope` - how a security region is represented and passed around.

A region is stored by its **vertices**.  Two derived representations are built
lazily and cached because the algorithms need them constantly:

* the convex hull (``scipy.spatial.ConvexHull``), giving facets and volume;
* the half-space form ``A x <= b``, which is how membership is tested and how
  the region enters an optimisation model.

Half-space membership
---------------------
:meth:`Polytope.contains` evaluates ``A x <= b`` directly from
``ConvexHull.equations`` rather than constructing a ``Delaunay`` triangulation
whose simplex count grows combinatorially with dimension.  Reading facet planes
from ``ConvexHull.equations`` also handles facets passing through the origin.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Dict, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from OpenRC.cases.axes import RegionAxes


#: Qhull option sets tried in order by :attr:`Polytope.hull`.  ``None`` is
#: SciPy's default; ``Qx`` merges exactly across coplanar facets (it also cures
#: the ``qh_check_maxout`` precision failure); ``Qt`` triangulates instead of
#: merging; ``Q12`` tolerates wide merges caused by nearly coincident points;
#: ``Q14`` merges such "pinched" points outright; ``QJ`` joggles the input.
#:
#: Excludes ``Qz`` (invalid for ``ConvexHull``) and ``Qbb`` (rescales the last
#: coordinate of the input array in place in SciPy <= 1.16).  Inputs are copied
#: before calling Qhull, and non-default option results are verified via
#: :func:`_hull_matches_hrep` or :func:`_hull_matches_points`.
_QHULL_OPTIONS = (None, "Qx", "Qt", "Qx Q12", "Q14", "QJ")

#: The ladder for ``HalfspaceIntersection``, which rejects ``Qbb``.  Every
#: result is validated by the caller.
_HALFSPACE_OPTIONS = (None, "Qx", "Qt", "Qx Q12", "Q14")

_GEOMETRY_CACHES = ("hull", "volume", "halfspaces", "facets", "n_facets",
                    "centroid", "hull_vertices")


def mc_volume(A: np.ndarray, b: np.ndarray, lower: np.ndarray, upper: np.ndarray, *,
              n_samples: int = 4000, seed: int = 0, batch: int = 500) -> tuple:
    """Monte Carlo volume of ``{x : A x <= b}`` inside the box ``[lower, upper]``.

    Uses a fixed RNG seed to cross-check volumes reported by fallback Qhull
    options.  Returns ``(estimate, standard_error)``.
    """
    if (isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer))
            or n_samples <= 0):
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
    if (isinstance(batch, (bool, np.bool_))
            or not isinstance(batch, (int, np.integer))
            or batch <= 0):
        raise ValueError(f"batch must be a positive integer, got {batch!r}")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError(f"seed must not be a boolean, got {seed!r}")
    raw_A = np.asarray(A)
    raw_b = np.asarray(b)
    raw_lo = np.asarray(lower)
    raw_hi = np.asarray(upper)
    if (raw_A.dtype == bool or raw_b.dtype == bool
            or raw_lo.dtype == bool or raw_hi.dtype == bool):
        raise ValueError("A, b, lower, and upper must be numeric arrays, not boolean")
    A = np.asarray(raw_A, dtype=float)
    b = np.asarray(raw_b, dtype=float)
    lower = np.asarray(raw_lo, dtype=float)
    upper = np.asarray(raw_hi, dtype=float)
    if (A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]
            or lower.ndim != 1 or upper.ndim != 1
            or lower.shape != (A.shape[1],) or upper.shape != (A.shape[1],)
            or A.shape[1] == 0):
        raise ValueError(
            f"Incompatible shapes for mc_volume: A={A.shape}, b={b.shape}, "
            f"lower={lower.shape}, upper={upper.shape}"
        )
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))
            and np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            and np.all(lower <= upper)):
        raise ValueError("A, b, lower, and upper must be finite with lower <= upper")
    rng = np.random.default_rng(seed)
    box = float(np.prod(upper - lower))
    hits = 0
    for start in range(0, n_samples, batch):
        m = min(batch, n_samples - start)
        x = lower + rng.random((m, len(lower))) * (upper - lower)
        hits += int(np.all(x @ A.T <= b[None, :], axis=1).sum())
    frac = hits / n_samples
    return box * frac, box * float(np.sqrt(frac * (1.0 - frac) / n_samples))


def _volume_agrees(A, b, lower, upper, volume: float, n_samples: int = 4000) -> tuple:
    """Whether ``volume`` matches the Monte Carlo volume of ``{A x <= b}`` in the box.

    Uses ``max(estimate, volume) / box`` for the binomial standard error so
    small-volume hulls with zero Monte Carlo hits are not spuriously rejected.
    """
    estimate, _ = mc_volume(A, b, lower, upper, n_samples=n_samples)
    box = float(np.prod(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))
    if not box > 0.0:
        return volume == 0.0, estimate
    share = min(max(estimate, volume) / box, 1.0)
    error = box * math.sqrt(share * (1.0 - share) / n_samples)
    return abs(estimate - volume) <= 5.0 * error + 0.02 * max(volume, estimate), estimate


def _sample_box(lower, upper, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return lower + rng.random((n, len(lower))) * (upper - lower)


_REFEREE_BLOCK_MB = 64.0


def _max_excess(planes: np.ndarray, points: np.ndarray) -> float:
    """``max(planes[:, :-1] @ points.T + planes[:, -1:])`` in bounded row blocks."""
    step = max(1, int(_REFEREE_BLOCK_MB * 2 ** 20 / (8 * max(len(points), 1))))
    worst = -math.inf
    for first in range(0, len(planes), step):
        block = planes[first:first + step]
        worst = max(worst, float((block[:, :-1] @ points.T + block[:, -1:]).max()))
    return worst


def _inside_planes(points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """``np.all(points @ planes[:, :-1].T + planes[:, -1] <= 0, axis=1)`` in plane blocks."""
    step = max(1, int(_REFEREE_BLOCK_MB * 2 ** 20 / (8 * max(len(points), 1))))
    inside = np.ones(len(points), dtype=bool)
    for first in range(0, len(planes), step):
        block = planes[first:first + step]
        inside &= np.all(points @ block[:, :-1].T + block[:, -1] <= 0.0, axis=1)
    return inside


class _Interval1DHull:
    """Exact 1-D convex hull ``[lo, hi]`` matching SciPy's ``ConvexHull`` interface."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.array(points, dtype=float, copy=True)
        self._rebuild()

    def _rebuild(self) -> None:
        from scipy.spatial import QhullError

        if len(self.points) < 2:
            raise QhullError("need at least 2 points for a 1-D hull")
        col = self.points[:, 0]
        i_lo = int(np.argmin(col))
        i_hi = int(np.argmax(col))
        lo = float(col[i_lo])
        hi = float(col[i_hi])
        if not hi > lo:
            raise QhullError("the input is flat: 1-D span is zero")
        self.vertices = np.array(sorted((i_lo, i_hi)), dtype=np.intp)
        self.simplices = np.array([[i_lo], [i_hi]], dtype=np.intp)
        self.equations = np.array([[-1.0, lo], [1.0, -hi]], dtype=float)
        self.neighbors = np.full((2, 1), -1, dtype=np.intp)
        self.volume = hi - lo
        self.area = 2.0

    @property
    def npoints(self) -> int:
        return int(len(self.points))

    def add_points(self, points: np.ndarray, restart: bool = False) -> None:
        del restart
        extra = np.asarray(points, dtype=float).reshape(-1, 1)
        self.points = np.vstack([self.points, extra])
        self._rebuild()

    def close(self) -> None:
        pass


def _point_excess_tol(points: np.ndarray, extra_coord: float = 0.0) -> float:
    """Scale- and translation-aware facet excess tolerance for ``points``."""
    span = float(np.ptp(points, axis=0).max())
    coord = max(float(np.abs(points).max()), float(extra_coord), span)
    return 1e-7 * span + 1e3 * np.finfo(float).eps * coord


def _hull_geometry_consistent(hull, points: np.ndarray) -> bool:
    """Fast LP-free referee for fallback Qhull options on bare point sets."""
    from OpenRC.geometry.live_hull import cone_volumes

    E = np.asarray(hull.equations, dtype=float)
    if E.ndim != 2 or len(E) == 0 or not np.all(np.isfinite(E)):
        return False
    pts = np.asarray(points, dtype=float)
    tol = _point_excess_tol(pts)
    if _max_excess(E, pts) > tol:
        return False
    hull_points = np.asarray(hull.points, dtype=float)
    verts = np.asarray(hull.vertices, dtype=np.intp)
    simplices = np.asarray(hull.simplices, dtype=np.intp)
    if len(verts) <= pts.shape[1] or len(simplices) <= pts.shape[1]:
        return False
    centre = hull_points[verts].mean(axis=0)
    coord = max(float(np.abs(pts).max()), float(np.ptp(pts, axis=0).max()))
    if _max_excess(E, centre[None, :]) > 1e3 * np.finfo(float).eps * coord:
        return False
    cones = cone_volumes(hull_points, simplices, centre)
    total = float(cones.sum())
    reported = float(hull.volume)
    if not (math.isfinite(total) and total > 0.0 and math.isfinite(reported)):
        return False
    return abs(total - reported) <= 1e-5 * total


def _hull_matches_hrep(hull, points: np.ndarray, A: np.ndarray, b: np.ndarray) -> tuple:
    """Referee a Qhull hull against the exact inequalities it should describe.

    Qhull-free: the facets must contain every vertex, and on a fixed sample of
    the bounding box, membership by the hull's facets must agree with
    membership by ``A x <= b`` (the exact definition of the body), up to a
    sliver of boundary noise.
    """
    E = np.asarray(hull.equations, dtype=float)
    if not np.all(np.isfinite(E)):
        return False, "non-finite facet equations"
    worst = _max_excess(E, points)
    if worst > _point_excess_tol(points):
        return False, f"a vertex lies {worst:.2e} outside a facet"
    # Every vertex inside the inequalities puts the hull inside the body, so
    # the volume check below only has to catch a hull that is too small -
    # which it does from the observed share, however thin the body.
    outside = _max_excess(np.hstack([A, -b[:, None]]), points)
    if outside > _point_excess_tol(points, float(np.abs(b).max())):
        return False, f"a vertex violates an inequality by {outside:.2e}"
    # Deduplicate exact rows only; rounding offsets would shift small boxes.
    rows = np.unique(E, axis=0)
    x = _sample_box(points.min(axis=0), points.max(axis=0), 3000)
    by_hull = _inside_planes(x, rows)
    by_hrep = _inside_planes(x, np.hstack([A, -b[:, None]]))
    wrong = int((by_hull != by_hrep).sum())
    if wrong > max(3, int(0.002 * len(x))):
        return False, f"facets disagree with the exact inequalities on {wrong}/{len(x)} samples"
    ok, estimate = _volume_agrees(A, b, points.min(axis=0), points.max(axis=0),
                                  float(hull.volume))
    if not ok:
        return False, (f"reported volume {hull.volume:.6g} but the body measures about "
                       f"{estimate:.6g}")
    return True, ""


def _hull_matches_points(hull, points: np.ndarray, n_samples: int = 200) -> tuple:
    """Validate a fallback Qhull hull of ``points`` via linear programs.

    ``x`` lies in ``conv(points)`` iff a convex combination reproduces it,
    tested via ``scipy.optimize.linprog``.  Used only for fallback hulls, since
    each sample solves one LP.  Three checks:

    1. every input point satisfies every facet, so ``conv(points)`` lies inside
       the reported body;
    2. points drawn uniformly *inside the reported body* (cones from the vertex
       mean over the reported simplices) are convex combinations of the input,
       so the body is not larger than ``conv(points)``, and the cone volumes
       add up to the reported volume;
    3. box samples get the same membership from the facets and from the LPs,
       and - where enough of them land inside - measure the reported volume.

    Simplex-cone sampling (step 2) remains effective in high dimensions
    (e.g. 16-D) where the polytope occupies a negligible fraction of its
    bounding box.
    """
    from scipy.optimize import linprog

    from OpenRC.geometry.live_hull import cone_volumes

    E = np.asarray(hull.equations, dtype=float)
    if not np.all(np.isfinite(E)):
        return False, "non-finite facet equations"
    points = np.asarray(points, dtype=float)
    dim = points.shape[1]
    worst = _max_excess(E, points)
    if worst > _point_excess_tol(points):
        return False, f"a vertex lies {worst:.2e} outside a facet"
    rows = np.unique(E, axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = np.where(upper > lower, upper - lower, 1.0)
    pts_unit = (points - lower) / extent
    a_eq = np.vstack([pts_unit.T, np.ones(len(points))])

    def in_hull(x: np.ndarray) -> np.ndarray:
        x_unit = (x - lower) / extent
        return np.array([
            linprog(np.zeros(len(points)), A_eq=a_eq, b_eq=np.append(xi, 1.0),
                    bounds=(0, None), method="highs").status == 0
            for xi in x_unit])

    # 2. samples inside the reported body
    hull_points = np.asarray(hull.points, dtype=float)
    simplices = np.asarray(hull.simplices)
    centre = hull_points[np.asarray(hull.vertices)].mean(axis=0)
    cones = cone_volumes(hull_points, simplices, centre)
    total = float(cones.sum())
    if not total > 0.0 or abs(total - float(hull.volume)) > 1e-6 * total:
        return False, (f"reported volume {hull.volume:.6g} but its simplices add up "
                       f"to {total:.6g}")
    rng = np.random.default_rng(0)
    picks = rng.choice(len(simplices), size=n_samples // 2, p=cones / total)
    weights = rng.dirichlet(np.ones(dim + 1), size=len(picks))
    corners = np.concatenate([hull_points[simplices[picks]],
                              np.broadcast_to(centre, (len(picks), 1, dim))], axis=1)
    inside = np.einsum("nk,nkd->nd", weights, corners)
    outside_true = int((~in_hull(inside)).sum())
    if outside_true > max(1, int(0.01 * len(inside))):
        return False, (f"{outside_true}/{len(inside)} points of the reported body lie "
                       "outside the convex hull of the input")

    # 3. box samples, where the box is not mostly empty
    x = _sample_box(lower, upper, n_samples // 2)
    by_lp = in_hull(x)
    by_hull = _inside_planes(x, rows)
    wrong = int((by_lp != by_hull).sum())
    if wrong > max(2, int(0.01 * len(x))):
        return False, f"facets disagree with LP membership on {wrong}/{len(x)} samples"
    hits = int(by_lp.sum())
    if hits >= 20:
        # Overlapping simplices would pass check 2 but inflate the volume.
        box = float(np.prod(upper - lower))
        frac = hits / len(x)
        error = box * float(np.sqrt(frac * (1 - frac) / len(x)))
        if abs(box * frac - hull.volume) > 5.0 * error + 0.02 * hull.volume:
            return False, (f"reported volume {hull.volume:.6g} but LP sampling measures "
                           f"about {box * frac:.6g}")
    return True, ""


def _normalise_rows(A: np.ndarray, b: np.ndarray) -> tuple:
    raw_A = np.asarray(A)
    raw_b = np.asarray(b)
    if raw_A.dtype == bool or raw_b.dtype == bool:
        raise ValueError("half-space rows and offsets must be numeric, not boolean")
    A = np.atleast_2d(np.asarray(raw_A, dtype=float))
    if raw_b.ndim > 1:
        raise ValueError("half-space rows need a non-empty (n, dim) matrix and n offsets")
    b = np.atleast_1d(np.asarray(raw_b, dtype=float))
    if A.ndim != 2 or not A.shape[0] or not A.shape[1] or b.shape != (A.shape[0],):
        raise ValueError("half-space rows need a non-empty (n, dim) matrix and n offsets")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError("half-space rows must be finite with non-zero normals")
    largest = np.max(np.abs(A), axis=1)
    if np.any(largest == 0.0):
        raise ValueError("half-space rows must be finite with non-zero normals")
    limits = np.finfo(float)
    extreme = ((largest < math.sqrt(limits.tiny))
               | (largest > math.sqrt(limits.max / A.shape[1])))
    # Preserve ordinary coefficients on the existing path; only rescale rows
    # whose squared norm could overflow or underflow.
    safe_A = A.copy() if np.any(extreme) else A
    safe_b = b.copy() if np.any(extreme) else b
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if np.any(extreme):
            safe_A[extreme] = A[extreme] / largest[extreme, None]
            safe_b[extreme] = b[extreme] / largest[extreme]
        norms = np.linalg.norm(safe_A, axis=1)
        normal_A, normal_b = safe_A / norms[:, None], safe_b / norms
    if not np.all(np.isfinite(normal_b)):
        raise ValueError("normalised half-space offsets exceed the floating-point range")
    return normal_A, normal_b


def _json_safe(value):
    """``value`` with non-finite floats as ``None`` and numpy types as Python ones."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)



def _staging_file(directory: Path, prefix: str, suffix: str, staged: list):
    """A new, uniquely named text file in ``directory``, appended to ``staged``.

    Created with mode 0666 so the operating system applies the umask, as for
    any ordinary file; ``tempfile`` would leave the published pair at 0600.
    """
    import secrets

    while True:
        candidate = Path(directory) / f"{prefix}{secrets.token_hex(8)}{suffix}"
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        staged.append(candidate)
        return os.fdopen(fd, "w", encoding="utf-8", newline="")

def _stream_sha256(stream: BinaryIO) -> str:
    """Fingerprint file bytes in bounded memory, leaving the stream at EOF."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(64 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def merge_close_points(points: np.ndarray, rel_tol: float = 1e-9) -> np.ndarray:
    """Drop near-duplicates in coordinates scaled by each axis's extent.

    Half-space enumeration reports a vertex once per combination of facets
    meeting there; in floating point those copies differ by ~1e-11 and make a
    subsequent hull ill-posed.  There is no absolute scale floor: a tiny body
    or a thin axis must not collapse merely because another coordinate is
    large.  Queries ``cKDTree`` neighbourhoods sequentially per retained point
    to avoid all-pairs memory allocation; removing duplicate vertices preserves
    inner approximations.
    """
    from scipy.spatial import cKDTree

    if isinstance(rel_tol, (bool, np.bool_)) or not np.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("rel_tol must be finite and non-negative")
    raw = np.asarray(points)
    if raw.dtype == bool:
        raise ValueError("points must be a numeric array, not boolean")
    points = np.atleast_2d(np.asarray(raw, dtype=float))
    if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite (n, dim) matrix with dim > 0")
    if len(points) < 2:
        return points
    with np.errstate(over="ignore"):
        extent = np.ptp(points, axis=0)
        offsets = points - points[0]
    extreme = ~np.isfinite(extent)
    if np.any(extreme):
        # Finite endpoints can still have an unrepresentable difference.
        # Rescale only those columns; ordinary close coordinates retain their
        # original subtraction, avoiding cancellation from a large translation.
        column_scale = np.max(np.abs(points[:, extreme]), axis=0)
        scaled = points[:, extreme] / column_scale
        offsets[:, extreme] = scaled - scaled[0]
        extent[extreme] = np.ptp(scaled, axis=0)
    normalised = offsets / np.where(extent > 0.0, extent, 1.0)
    keep = np.ones(len(points), dtype=bool)
    tree = cKDTree(normalised)
    for i in range(len(points)):
        if keep[i]:
            neighbours = tree.query_ball_point(normalised[i], rel_tol)
            keep[neighbours] = False
            keep[i] = True
    return points[keep]


@dataclass(eq=False)
class Polytope:
    """A bounded convex region in region-coordinate space (MW / Mvar).

    Polytopes compare and hash by identity: two vertex lists can describe the
    same body (interior points, order), so ``==`` makes no geometric claim.

    Parameters
    ----------
    vertices:
        ``(n_vertices, dim)`` array.  Points strictly inside the hull are kept
        as given; call :meth:`reduce_to_hull` to drop them.  Stored geometry is
        an independent, read-only snapshot; use :meth:`replace` to derive a
        different body.  Explicit reassignment invalidates geometric caches.
    axes:
        The :class:`~OpenRC.cases.axes.RegionAxes` these coordinates refer to.
        Optional, but carried through saves so a stored region stays meaningful.
    metadata:
        Free-form provenance (case, oracle, builder, timings, ...).
    """

    vertices: np.ndarray
    axes: Optional["RegionAxes"] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    #: Exact defining inequalities ``(A, b)``, when the body was *built* from
    #: them (by :func:`~OpenRC.geometry.ops.from_halfspaces`).  They are then
    #: authoritative: :attr:`halfspaces`, :attr:`facets` and :meth:`contains`
    #: use them directly, and the Qhull hull is only trusted after it has been
    #: checked against them.
    hrep: Optional[tuple] = field(default=None, repr=False, compare=False)
    #: Why :attr:`hull` could not be built (empty while it can).
    hull_error: str = field(default="", init=False, repr=False, compare=False)

    def __setattr__(self, name, value) -> None:
        if self.__dict__.get("_geometry_ready", False) and name in ("vertices", "hrep"):
            if name == "vertices":
                raw = np.asarray(value)
                if raw.dtype == bool:
                    raise ValueError("vertices must be a numeric array, not boolean")
                value = np.atleast_2d(np.array(raw, dtype=float, copy=True))
                if (value.ndim != 2 or 0 in value.shape
                        or not np.all(np.isfinite(value))):
                    raise ValueError("vertices must be a non-empty finite (n, dim) matrix")
                if self.axes is not None and len(self.axes) != value.shape[1]:
                    raise ValueError("vertices must match the axes dimension")
                value.setflags(write=False)
                # New vertices define a new body; old exact inequalities cannot
                # remain authoritative for it.
                object.__setattr__(self, "hrep", None)
            elif value is not None:
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise ValueError("hrep must be a 2-tuple (A, b)")
                A, b = _normalise_rows(*value)
                if A.shape[1] != self.dim:
                    raise ValueError("hrep must match the vertices dimension")
                A.setflags(write=False)
                b.setflags(write=False)
                value = (A, b)
            for key in _GEOMETRY_CACHES:
                self.__dict__.pop(key, None)
            object.__setattr__(self, "hull_error", "")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        raw = np.asarray(self.vertices)
        if raw.dtype == bool:
            raise ValueError("vertices must be a numeric array, not boolean")
        self.vertices = np.atleast_2d(np.array(raw, dtype=float, copy=True))
        if self.vertices.ndim != 2:
            raise ValueError("vertices must be a 2-D (n_vertices, dim) array")
        if self.vertices.shape[0] == 0 or self.vertices.shape[1] == 0:
            raise ValueError("vertices must contain at least one finite point and one coordinate")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("vertices must contain only finite values")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        if self.axes is not None:
            if not hasattr(self.axes, "labels"):
                raise TypeError("axes must be a RegionAxes instance or None")
            if len(self.axes) != self.dim:
                raise ValueError(
                    f"axes have dimension {len(self.axes)} but vertices have {self.dim}"
                )
        if self.hrep is not None:
            if not isinstance(self.hrep, (tuple, list)) or len(self.hrep) != 2:
                raise ValueError("hrep must be a 2-tuple (A, b)")
            A, b = _normalise_rows(*self.hrep)
            if A.shape[1] != self.dim:
                raise ValueError(f"hrep has dimension {A.shape[1]}, expected {self.dim}")
            A.setflags(write=False)
            b.setflags(write=False)
            self.hrep = (A, b)
        self.vertices.setflags(write=False)
        self._geometry_ready = True

    # ------------------------------------------------------------ basic shape
    @property
    def dim(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    # -------------------------------------------------------- cached geometry
    @cached_property
    def hull(self):
        """``scipy.spatial.ConvexHull`` of the vertices, or ``None`` if degenerate.

        Vertices produced by half-space enumeration can sit ~1e-11 apart where
        multiple facets meet, causing Qhull's default merging to abort with a
        topology error ("wide merge ... dupridge") even on full-dimensional
        inputs.  Fallback options in ``_QHULL_OPTIONS`` are validated before
        acceptance:

        * a body with exact inequalities (:attr:`hrep`) compares hull facets
          against ``hrep`` on a deterministic sample;
        * a bare point set accepts SciPy's default options directly and checks
          fallback options via LP point-in-hull tests.

        When no option yields an accepted hull, the reason is kept in
        :attr:`hull_error`.
        """
        from scipy.spatial import ConvexHull, QhullError

        if self.n_vertices <= self.dim:
            self.hull_error = (f"{self.n_vertices} points cannot span a "
                               f"{self.dim}-D body")
            return None
        if self.dim == 1:
            try:
                hull_1d = _Interval1DHull(self.vertices)
            except QhullError as exc:
                self.hull_error = str(exc)
                return None
            if self.hrep is not None:
                ok, reason = _hull_matches_hrep(hull_1d, self.vertices, *self.hrep)
                if not ok:
                    self.hull_error = reason
                    return None
            return hull_1d
        errors = []
        for options in _QHULL_OPTIONS:
            try:
                hull = ConvexHull(self.vertices.copy(), qhull_options=options)
            except QhullError as exc:
                errors.append(f"[{options or 'default'}] {str(exc).splitlines()[0]}")
                continue
            if self.hrep is not None:
                ok, reason = _hull_matches_hrep(hull, self.vertices, *self.hrep)
            elif options is None:
                return hull        # cheap path: the default is trusted on bare points
            else:
                ok, reason = _hull_matches_points(hull, self.vertices)
            if ok:
                return hull
            errors.append(f"[{options or 'default'}] rejected: {reason}")
        self.hull_error = "; ".join(errors)
        return None

    @cached_property
    def volume(self) -> float:
        """Hypervolume; ``0.0`` for a degenerate (lower-dimensional) body."""
        hull = self.hull
        return float(hull.volume) if hull is not None else 0.0

    @cached_property
    def halfspaces(self) -> tuple[np.ndarray, np.ndarray]:
        """``(A, b)`` with the region equal to ``{x : A x <= b}``.

        Taken straight from ``ConvexHull.equations``, which stores each facet as
        ``[normal | offset]`` with ``normal . x + offset <= 0`` inside.
        """
        if self.hrep is not None:
            return self.hrep
        hull = self.hull
        if hull is None:
            raise ValueError(
                f"polytope is degenerate ({self.n_vertices} vertices in {self.dim}-D) "
                "and has no half-space representation"
                + (f" - Qhull: {self.hull_error}" if self.hull_error else "")
            )
        equations = np.asarray(hull.equations, dtype=float)
        A, b = equations[:, : self.dim], -equations[:, self.dim]
        A.setflags(write=False)
        b.setflags(write=False)
        return A, b

    @cached_property
    def facets(self) -> tuple[np.ndarray, np.ndarray]:
        """:attr:`halfspaces` with one row per distinct facet, rows of unit norm.

        ``ConvexHull`` triangulates non-simplicial facets into multiple
        simplices sharing the same plane, and repeated intersections can produce
        planes differing by ~1e-12.  Near-duplicate planes cause dual-hull
        degeneracies in ``HalfspaceIntersection`` and add redundant constraints
        to downstream LPs.

        Merges rows with unit normals within ``1e-12`` inf-norm and relative
        offsets within ``1e-8``, keeping the tightest offset while retaining
        distinct nearly-parallel faces.
        """
        from scipy.spatial import cKDTree

        if self.hrep is not None:
            return self.hrep       # already one row per facet, unit norm
        A, b = self.halfspaces
        norms = np.linalg.norm(A, axis=1)
        A, b = A / norms[:, None], b / norms
        # Normals are dimensionless, whereas offsets carry region units.
        # Shift around the vertex mean and scale by the intrinsic span so that
        # tiny bodies and far-translated bodies deduplicate identically.
        c = self.vertices.mean(axis=0)
        b_rel = b - A @ c
        span = float(np.ptp(self.vertices, axis=0).max())
        offset_scale = span if span > 0.0 else 1.0
        rows = np.hstack([A, (b_rel / offset_scale)[:, None]])
        tol = 1e-8
        keep = np.ones(len(rows), dtype=bool)
        tree = cKDTree(rows)
        for i in np.argsort(b_rel):              # tightest parallel row survives
            if keep[i]:
                nearby = np.asarray(tree.query_ball_point(rows[i], tol), dtype=np.intp)
                parallel = nearby[np.max(np.abs(A[nearby] - A[i]), axis=1) <= 1e-12]
                keep[parallel[parallel != i]] = False
        distinct_A, distinct_b = A[keep], b[keep]
        distinct_A.setflags(write=False)
        distinct_b.setflags(write=False)
        return distinct_A, distinct_b

    @cached_property
    def n_facets(self) -> int:
        """Distinct facets (one per row of :attr:`facets`), however the hull is stored."""
        if self.hrep is not None:
            return int(len(self.hrep[1]))
        hull = self.hull
        return 0 if hull is None else int(len(self.facets[1]))

    @cached_property
    def centroid(self) -> np.ndarray:
        """Mean of the hull vertices (not the volumetric centroid)."""
        return self.hull_vertices.mean(axis=0)

    @cached_property
    def hull_vertices(self) -> np.ndarray:
        """Extreme points of ``vertices`` indexed by ``hull.vertices``."""
        hull = self.hull
        if hull is None:
            return self.vertices
        return self.vertices[hull.vertices]

    # --------------------------------------------------------------- queries
    def validate_axes(self, axes: "RegionAxes") -> None:
        """Validate axis count and ordering before a grid query.

        Raises ``ValueError`` if ``len(axes) != self.dim`` or if ``self.axes``
        is set and its labels or order differ from ``axes``.
        """
        if len(axes) != self.dim:
            raise ValueError(f"{len(axes)} axes for a {self.dim}-D region")
        if self.axes is not None and axes.labels != self.axes.labels:
            raise ValueError("query axes must match the region axes in the same order")

    def contains(self, points: np.ndarray, tol: float = 1e-9, *,
                 max_working_mb: float = 16.0) -> np.ndarray:
        """Boolean membership for one point or an array of points.

        ``tol`` is a relative slack on each half-space, so points exactly on a
        facet count as inside.  The point-by-facet matrix is evaluated in
        bounded row batches (``max_working_mb`` MiB per temporary matrix), so
        membership of a large sample cloud does not require gigabytes at once.
        One row is the minimum batch and can exceed this soft working budget
        when a hull has exceptionally many facets.
        """
        if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and non-negative")
        if (isinstance(max_working_mb, (bool, np.bool_))
                or not np.isfinite(max_working_mb) or max_working_mb <= 0.0):
            raise ValueError("max_working_mb must be finite and strictly positive")
        raw_input = np.asarray(points)
        if raw_input.dtype == bool:
            raise ValueError("query points must be numeric, not boolean")
        raw = np.asarray(raw_input, dtype=float)
        if raw.size == 0 and raw.ndim == 2 and raw.shape[1] == self.dim:
            _ = self.halfspaces
            return np.zeros(0, dtype=bool)
        pts = np.atleast_2d(raw)
        if pts.ndim != 2 or pts.shape[1] != self.dim:
            raise ValueError(f"points have dimension {pts.shape[1]}, expected {self.dim}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("query points must contain only finite values")
        A, b = self.halfspaces
        budget_rows = max_working_mb / (8 * max(len(b), 1)) * 2 ** 20
        rows = max(1, int(min(max(len(pts), 1), budget_rows)))
        inside = np.empty(len(pts), dtype=bool)
        if tol == 0.0:
            for start in range(0, len(pts), rows):
                stop = min(start + rows, len(pts))
                inside[start:stop] = np.all(pts[start:stop] @ A.T <= b[None, :], axis=1)
        else:
            c = self.vertices.mean(axis=0)
            b_rel = b - A @ c
            radius = float(np.linalg.norm(self.vertices - c, axis=1).max())
            eps_floor = 8.0 * np.finfo(float).eps * (np.abs(b) + float(np.linalg.norm(c)))
            threshold_rel = b_rel + tol * np.maximum(np.abs(b_rel), radius) + eps_floor
            for start in range(0, len(pts), rows):
                stop = min(start + rows, len(pts))
                inside[start:stop] = np.all(
                    (pts[start:stop] - c) @ A.T <= threshold_rel[None, :], axis=1
                )
        return inside if np.ndim(points) > 1 else inside[0]

    def bounding_box(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds, optionally grown by ``margin`` on every side."""
        from OpenRC.geometry.measures import bounding_box

        return bounding_box(self.vertices, margin)

    def volume_without(self, index: int) -> float:
        """Hypervolume after removing vertex ``index`` - the pruning primitive."""
        from OpenRC.geometry.measures import hull_volume

        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
                or not 0 <= int(index) < self.n_vertices):
            raise IndexError("vertex index out of range")
        return hull_volume(np.delete(self.vertices, int(index), axis=0))

    # ------------------------------------------------------------ derivations
    def reduce_to_hull(self) -> "Polytope":
        """Drop interior points, keeping only true vertices."""
        out = self.replace(vertices=self.hull_vertices)
        out.hrep = self.hrep           # same body, same inequalities
        return out

    @classmethod
    def with_known_volume(cls, vertices, volume: float, *, axes=None,
                          metadata: Optional[Dict[str, object]] = None) -> "Polytope":
        """A polytope whose volume is already known exactly (e.g. from a live hull).

        The volume is recorded instead of recomputed; every other hull-derived
        attribute is still built lazily if it is asked for.
        """
        if (isinstance(volume, (bool, np.bool_))
                or not math.isfinite(float(volume)) or float(volume) < 0.0):
            raise ValueError("volume must be a finite non-negative number")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        polytope = cls(vertices, axes=axes, metadata=dict(metadata or {}))
        polytope.__dict__["volume"] = float(volume)     # fills the cached_property
        return polytope

    def replace(self, *, vertices: Optional[np.ndarray] = None, **meta) -> "Polytope":
        """Copy with new vertices and/or extra metadata (caches are dropped).

        Exact inequalities survive only when the vertices do: a different
        vertex set is a different body.
        """
        merged = dict(self.metadata)
        merged.update(meta)
        return Polytope(
            vertices=self.vertices if vertices is None else vertices,
            axes=self.axes,
            metadata=merged,
            hrep=self.hrep if vertices is None else None,
        )

    def translate(self, offset: np.ndarray) -> "Polytope":
        """Rigid shift - the ``(+)`` operator of the conservative construction."""
        raw_offset = np.asarray(offset)
        if raw_offset.dtype == bool:
            raise ValueError("offset must be a numeric vector, not boolean")
        offset = np.asarray(raw_offset, dtype=float)
        if offset.shape != (self.dim,) or not np.all(np.isfinite(offset)):
            raise ValueError(f"offset must be a finite vector with shape ({self.dim},)")
        out = self.replace(vertices=self.vertices + offset[None, :])
        if self.hrep is not None:
            A, b = self.hrep
            out.hrep = (A, b + A @ offset)
        return out

    def subspace(self, dims: Sequence[int]) -> "Polytope":
        """Orthogonal projection onto coordinate indices ``dims``.

        Projects ``vertices[:, dims]`` (marginal feasible set, as opposed to a
        cross-section at fixed values of the omitted coordinates).
        """
        if isinstance(dims, (str, bytes, bytearray)):
            raise TypeError("dims must be a sequence of coordinate indices, not a string")
        dims = list(dims)
        # Reject booleans, negative indices, out-of-range indices, and duplicates.
        if (not dims or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, Integral)
                            for d in dims)
                or len(set(dims)) != len(dims) or min(dims) < 0 or max(dims) >= self.dim):
            raise ValueError(
                f"dims must be distinct coordinate indices in 0..{self.dim - 1}, got {dims}")
        axes = None
        if self.axes is not None:
            from OpenRC.cases.axes import RegionAxes

            axes = RegionAxes([self.axes[d] for d in dims])
        return Polytope(self.vertices[:, dims], axes=axes, metadata=dict(self.metadata))

    # --------------------------------------------------------------- storage
    def save(self, path: Path) -> Path:
        """Write ``<path>.csv`` (vertices) and ``<path>.json`` (metadata).

        The CSV keeps one column per axis, named after the axis label when
        known, so a stored region is readable without the library.  Files are
        staged before publication; the SHA-256 digest in the JSON sidecar
        detects accidental file truncation or mismatched CSV/JSON pairs on load.
        """
        path = Path(path)
        if path.suffix.lower() in (".csv", ".json"):   # keep dots inside a stem
            path = path.with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd

        columns = (list(self.axes.labels) if self.axes is not None
                   else [f"x{i}" for i in range(self.dim)])
        sidecar = dict(self.metadata)
        sidecar.update(
            dim=self.dim,
            n_vertices=self.n_vertices,
            volume=self.volume,
            axes=list(self.axes.labels) if self.axes is not None else None,
            units="MW for p axes, Mvar for q axes",
        )
        if self.hrep is not None:
            sidecar["hrep"] = {"A": self.hrep[0].tolist(), "b": self.hrep[1].tolist()}
        staged: list = []
        try:
            with _staging_file(path.parent, f".{path.name}-", ".csv", staged) as csv_file:
                csv_stage = staged[-1]
                pd.DataFrame(self.vertices, columns=columns).to_csv(csv_file, index=False)
            with csv_stage.open("rb") as stream:
                checksum = _stream_sha256(stream)
            sidecar["_openrc_storage"] = {"schema_version": 1, "csv_sha256": checksum}
            # Strict JSON: non-finite values become null.  Serialise before
            # replacing either destination so preparation errors preserve both.
            payload = json.dumps(_json_safe(sidecar), indent=2, allow_nan=False) + "\n"
            with _staging_file(path.parent, f".{path.name}-", ".json", staged) as json_file:
                json_stage = staged[-1]
                json_file.write(payload)
            # A filesystem cannot atomically replace two paths.  Either order
            # leaves a brief mixed-pair window; load refuses its checksum.
            json_stage.replace(path.with_name(path.name + ".json"))
            csv_stage.replace(path.with_name(path.name + ".csv"))
        finally:
            for temporary in staged:
                temporary.unlink(missing_ok=True)
        return path.with_name(path.name + ".csv")

    @classmethod
    def load(cls, path: Path, *, axes=None) -> "Polytope":
        """Read back a region written by :meth:`save`.

        Also accepts a bare CSV (``col1,col2,col3`` and no sidecar).
        Explicit axes must agree with saved axes; relabelling a stored region
        would change the injections being validated without changing its points.
        A sidecar with a checksum rejects a changed CSV or a pair from different
        saves; sidecars without one are still accepted.
        """
        import pandas as pd

        path = Path(path)
        suffix = path.suffix.lower()                 # as in save()
        if suffix == ".csv" and path.exists():
            csv_path = path
        else:
            stem = path.with_suffix("") if suffix in (".csv", ".json") else path
            csv_path = stem.with_name(stem.name + ".csv")
        metadata: Dict[str, object] = {}
        json_path = csv_path.with_suffix(".json")
        if json_path.exists():
            metadata = json.loads(json_path.read_text())
            if not isinstance(metadata, dict):
                raise ValueError("region JSON sidecar must contain an object")
        storage = metadata.get("_openrc_storage")
        if storage is not None and (not isinstance(storage, dict)
                                    or storage.get("schema_version") != 1
                                    or not isinstance(storage.get("csv_sha256"), str)):
            raise ValueError("invalid or unsupported region snapshot metadata")
        with csv_path.open("rb") as stream:
            # The default fast parser may round a shortest decimal to a
            # neighbouring float. Preserve the exact stored float64 value.
            frame = pd.read_csv(stream, float_precision="round_trip")
            # Keep the same open file across parsing/hash checking, even if a
            # concurrent save replaces the CSV path in the meantime.
            stream.seek(0)
            checksum_ok = storage is None or _stream_sha256(stream) == storage["csv_sha256"]
        vertices = frame.to_numpy(dtype=float)

        stored_axes = metadata.get("axes")
        if isinstance(stored_axes, list) and stored_axes:
            from OpenRC.cases.axes import RegionAxes

            saved_axes = RegionAxes(stored_axes)
            if list(frame.columns) != saved_axes.labels:
                raise ValueError(f"CSV column labels/order {list(frame.columns)} do not match "
                                 f"the stored axes {saved_axes.labels}")
            if axes is not None and axes.labels != saved_axes.labels:
                raise ValueError(f"explicit axes {axes.labels} do not match the stored axes "
                                 f"{saved_axes.labels} of {csv_path.name}")
            axes = saved_axes if axes is None else axes
        if not checksum_ok:
            raise ValueError(f"CSV checksum does not match the saved snapshot of {csv_path.name}; "
                             "keep the CSV and JSON from the same completed save")
        stored = metadata.pop("hrep", None)
        if storage is not None:
            for envelope_key in ("_openrc_storage", "dim", "n_vertices",
                                 "volume", "axes", "units"):
                metadata.pop(envelope_key, None)
        hrep = (None if not isinstance(stored, dict)
                else (np.array(stored["A"]), np.array(stored["b"])))
        return cls(vertices, axes=axes, metadata=metadata, hrep=hrep)

    # ------------------------------------------------------------------ repr
    def summary(self) -> str:
        lower, upper = self.bounding_box()
        span = ", ".join(f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in zip(lower, upper))
        labels = ", ".join(self.axes.labels) if self.axes is not None else "-"
        return (
            f"Polytope dim={self.dim} vertices={self.n_vertices} facets={self.n_facets}\n"
            f"    volume = {self.volume:.6g}\n"
            f"    axes   = {labels}\n"
            f"    box    = {span}"
        )

    def __repr__(self) -> str:
        return f"<Polytope dim={self.dim} n_vertices={self.n_vertices} volume={self.volume:.4g}>"
