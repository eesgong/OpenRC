"""Set operations on polytopes: intersection, shifting and box clipping.

These are the building blocks of the region-space conservative construction
(*Method 1*), which derives an inner approximation of the exact feasible set by
intersecting two convex models' regions and then intersecting the result with
translated copies of itself.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike

from OpenRC.geometry.polytope import Polytope, merge_close_points


def intersect(a: Polytope, b: Polytope, *, method: str = "halfspace",
              interior_point: Optional[np.ndarray] = None) -> Polytope:
    """Intersection of two polytopes.

    Parameters
    ----------
    method:
        ``"halfspace"`` (default)
            Stack both half-space systems and re-enumerate the vertices with
            ``scipy.spatial.HalfspaceIntersection``.  This is the exact
            intersection.
        ``"vertex_filter"``
            Keep the vertices of each body that lie inside the other.  Forms an
            inner approximation by omitting vertices where facets of ``a`` and
            ``b`` intersect (retained for reproducing published baselines).
    interior_point:
        A point strictly inside both bodies, needed to seed the half-space
        enumeration.  Found automatically by Chebyshev centring when omitted.

    Raises
    ------
    EmptyIntersectionError
        If the two bodies do not overlap in a full-dimensional way.
    """
    if a.dim != b.dim:
        raise ValueError(f"dimension mismatch: {a.dim} vs {b.dim}")
    if a.axes is not None and b.axes is not None and a.axes.labels != b.axes.labels:
        raise ValueError("intersection requires identical axes in the same order")

    if method == "vertex_filter":
        kept = [v for v in a.vertices if b.contains(v[None, :])[0]]
        kept += [v for v in b.vertices if a.contains(v[None, :])[0]]
        if not kept:
            raise EmptyIntersectionError(
                "vertex filtering kept too few points to form a body; the two regions "
                "barely overlap (try method='halfspace')"
            )
        unique_kept = merge_close_points(np.array(kept))
        if len(unique_kept) <= a.dim:
            raise EmptyIntersectionError(
                "vertex filtering kept too few points to form a body; the two regions "
                "barely overlap (try method='halfspace')"
            )
        out = Polytope(unique_kept, axes=a.axes or b.axes,
                       metadata={"op": "intersect", "method": method}).reduce_to_hull()
        if out.hull is None:
            raise EmptyIntersectionError(
                "vertex filtering kept only degenerate points; the two regions "
                "do not have a full-dimensional vertex-filtered overlap (try method='halfspace')"
            )
        return out

    if method != "halfspace":
        raise ValueError(f"unknown intersection method {method!r}")

    # Exact in half-space form: the intersection is the stacked system.
    A = np.vstack([a.facets[0], b.facets[0]])
    rhs = np.concatenate([a.facets[1], b.facets[1]])
    try:
        out = from_halfspaces(A, rhs, axes=a.axes or b.axes, interior_point=interior_point,
                              what="half-space intersection")
    except EmptyIntersectionError:
        raise EmptyIntersectionError("the two regions have empty interior intersection")
    out.metadata.update(op="intersect", method=method)
    return out


def intersect_all(bodies: Iterable[Polytope], **kwargs) -> Polytope:
    """Left fold of :func:`intersect` over an iterable of polytopes."""
    if isinstance(bodies, (str, bytes, bytearray)):
        raise TypeError("bodies must be an iterable of Polytopes, not a string")
    bodies = list(bodies)
    if not bodies:
        raise ValueError("nothing to intersect")
    if not all(isinstance(b, Polytope) for b in bodies):
        raise TypeError("every element of bodies must be a Polytope")
    out = bodies[0]
    for other in bodies[1:]:
        out = intersect(out, other, **kwargs)
    return out


def chebyshev_center(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Centre of the largest ball inscribed in ``{x : A x <= b}``.

    Returns ``None`` when the system has empty interior.  Solved via
    ``scipy.optimize.linprog`` (HiGHS): ``max r`` subject to
    ``a_i . x + r ||a_i||_2 <= b_i``.

    Raises ``ValueError`` if the radius is unbounded, and
    :class:`GeometryNumericsError` if HiGHS terminates without proving
    feasibility or infeasibility.
    """
    from scipy.optimize import linprog

    from OpenRC.geometry.polytope import _normalise_rows

    A, b = _normalise_rows(A, b)
    n = A.shape[1]
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    # variables: [x (n), r (1)];  objective: maximise r
    ub_A = np.hstack([A, norms])
    cost = np.zeros(n + 1)
    cost[-1] = -1.0
    bounds = [(None, None)] * n + [(0.0, None)]
    result = linprog(cost, A_ub=ub_A, b_ub=b, bounds=bounds, method="highs")
    if result.status == 2:  # The LP established infeasibility.
        return None
    if result.status == 3:
        raise ValueError("Chebyshev centre has an unbounded radius; a bounded polytope is required")
    if not result.success or result.x is None:
        raise GeometryNumericsError(
            f"Chebyshev centre LP did not establish an interior: {result.message}"
        )
    x = np.asarray(result.x, dtype=float)
    if x.shape != (n + 1,) or not np.all(np.isfinite(x)):
        raise GeometryNumericsError("Chebyshev centre LP returned an invalid solution")
    with np.errstate(over="ignore", invalid="ignore"):
        slack = b - A @ x[:n]
    if not np.all(np.isfinite(slack)):
        raise GeometryNumericsError("Chebyshev centre LP residuals are not finite")
    # Use a scale-relative slack threshold rather than a fixed radius floor so
    # small valid polytopes are accepted while zero-slack numerical noise is rejected.
    noise = 64.0 * np.finfo(float).eps * (np.abs(b) + float(np.linalg.norm(x[:n])))
    if np.all(slack > noise):
        return x[:n]
    solver_tol = max(float(np.max(noise)), 1e-7 * max(1.0, float(np.abs(b).max())))
    if x[-1] > solver_tol:
        raise GeometryNumericsError("Chebyshev centre LP returned no strictly interior point")
    return None


def clip_to_box(body: Polytope, lower: ArrayLike, upper: ArrayLike) -> Polytope:
    """Intersect with an axis-aligned box.

    Used as the last step of the conservative construction, where the POC's own
    rating limits are re-applied to the region.
    """
    raw_lo = np.asarray(lower)
    raw_hi = np.asarray(upper)
    if raw_lo.dtype == bool or raw_hi.dtype == bool:
        raise ValueError("box bounds must be numeric vectors, not boolean")
    lo = np.asarray(raw_lo, dtype=float)
    hi = np.asarray(raw_hi, dtype=float)
    dim = body.dim
    if lo.shape != (dim,) or hi.shape != (dim,):
        raise ValueError(f"box bounds must both have shape ({dim},)")

    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("box bounds must contain only finite values")
    if np.any(lo >= hi):
        raise ValueError("every lower box bound must be strictly below its upper bound")

    # Append the 2*dim box inequalities I x <= hi, -I x <= -lo directly to body.facets.
    body_A, body_b = body.facets
    eye = np.eye(dim)
    A = np.vstack([body_A, eye, -eye])
    rhs = np.concatenate([body_b, hi, -lo])
    try:
        out = from_halfspaces(A, rhs, axes=body.axes, what="box clipping")
    except EmptyIntersectionError:
        raise EmptyIntersectionError("the polytope and box have empty interior intersection")
    out.metadata.update(op="clip_to_box", method="halfspace")
    return out


def erode_by_box(body: Polytope, extent: ArrayLike) -> Polytope:
    """Points whose whole box ``y + [0, extent]`` lies inside ``body``.

    This is the Pontryagin difference of ``body`` and the box spanned by the
    origin and ``extent`` (components may have either sign), tightening each
    facet ``a . y <= b`` to ``a . y <= b - sum_k max(a_k * extent_k, 0)``.  The
    result equals the intersection of all ``2**dim`` translated copies, at the
    cost of one vertex enumeration.

    Raises
    ------
    EmptyIntersectionError
        If no point of the body can absorb the whole box.
    """
    raw_box = np.asarray(extent)
    if raw_box.dtype == bool:
        raise ValueError("extent must be a numeric vector, not boolean")
    box = np.asarray(raw_box, dtype=float)
    if box.shape != (body.dim,):
        raise ValueError(f"extent has shape {box.shape}, expected ({body.dim},)")
    if not np.all(np.isfinite(box)):
        raise ValueError("extent must contain only finite values")
    A, b = body.facets
    tightened = b - np.maximum(A * box[None, :], 0.0).sum(axis=1)
    try:
        out = from_halfspaces(A, tightened, axes=body.axes, what="box erosion")
    except EmptyIntersectionError:
        raise EmptyIntersectionError(
            "no point of the body can absorb the whole shift box; the eroded set is empty")
    out.metadata.update(op="erode_by_box", extent=box.tolist())
    return out


def from_halfspaces(A: np.ndarray, b: np.ndarray, *, axes=None,
                    interior_point: Optional[np.ndarray] = None,
                    what: str = "half-space system") -> Polytope:
    """Construct the polytope ``{x : A x <= b}`` from half-space inequalities.

    Stores the irredundant normalised inequalities in
    :attr:`~OpenRC.geometry.polytope.Polytope.hrep` for direct use by
    :attr:`~Polytope.facets` and :meth:`~Polytope.contains`.

    Steps: normalise rows; merge identical unit normals within offset tolerance
    (retaining the tightest offset); locate a strictly interior point; enumerate
    and validate vertices; drop non-facet-defining rows.

    Raises
    ------
    EmptyIntersectionError
        If the system has no full-dimensional interior.
    GeometryNumericsError
        If numerical centring or vertex enumeration cannot establish a result.
    ValueError
        If invalid coefficients or an unbounded centre problem are detected.
    """
    from OpenRC.geometry.polytope import _normalise_rows

    A, b = _normalise_rows(A, b)
    A, b = _merge_parallel_rows(A, b)
    dim = A.shape[1]
    seed: Optional[np.ndarray]
    if interior_point is not None:
        # Validate that interior_point satisfies A @ seed < b before calling Qhull.
        raw_seed = np.asarray(interior_point)
        if raw_seed.dtype == bool:
            raise ValueError("interior_point must be a numeric vector, not boolean")
        seed = np.asarray(raw_seed, dtype=float)
        if seed.shape != (dim,) or not np.all(np.isfinite(seed)):
            raise ValueError(f"interior_point must be a finite vector of length {dim}")
        if not np.all(A @ seed < b):
            raise ValueError(f"{what}: interior_point is not strictly inside every inequality")
    elif dim == 1:
        seed = None
    else:
        seed = chebyshev_center(A, b)
    if dim == 1:
        pos = A[:, 0] > 0.0
        neg = A[:, 0] < 0.0
        hi = float(np.min(b[pos])) if np.any(pos) else float("inf")
        lo = float(np.max(-b[neg])) if np.any(neg) else float("-inf")
        if not lo < hi:
            raise EmptyIntersectionError(f"{what}: the system has no interior")
        if not ( np.isfinite(lo) and np.isfinite(hi) ):
            raise ValueError(f"{what}: polyhedron is unbounded")
        return Polytope(
            np.array([[lo], [hi]], dtype=float),
            axes=axes,
            hrep=(np.array([[-1.0], [1.0]], dtype=float), np.array([-lo, hi], dtype=float)),
        )
    if seed is None:
        raise EmptyIntersectionError(f"{what}: the system has no interior")
    if len(b) <= dim or np.linalg.matrix_rank(A) < dim:
        raise ValueError(f"{what}: polyhedron is unbounded")
    points = _enumerate_vertices(np.hstack([A, -b[:, None]]), seed, what=what)
    # A row supports a facet iff the vertices lying on its hyperplane span a
    # (dim - 1)-dimensional affine subspace; lower-dimensional faces (e.g. a
    # 4-vertex square face of a 4-D hypercube) are redundant and dropped.
    pts_rel = points - seed[None, :]
    b_rel = b - A @ seed
    radius = float(np.linalg.norm(pts_rel, axis=1).max())
    eps_row = 64.0 * np.finfo(float).eps * (np.abs(b) + float(np.linalg.norm(seed)))
    tol_row = 1e-7 * radius + eps_row
    on_count = np.zeros(len(b), dtype=np.intp)
    step = max(1, int(2 ** 23 // max(len(b), 1)))       # 64 MiB per block
    for first in range(0, len(pts_rel), step):
        on_count += ((pts_rel[first:first + step] @ A.T) >= (b_rel[None, :] - tol_row[None, :])).sum(axis=0)
    keep = np.zeros(len(b), dtype=bool)
    rank_tol = 1e-6 * (radius if radius > 0.0 else 1.0)
    for row_idx in np.flatnonzero(on_count >= dim):
        on_mask = (pts_rel @ A[row_idx]) >= (b_rel[row_idx] - tol_row[row_idx])
        pts_on = pts_rel[on_mask]
        if len(pts_on) >= dim and np.linalg.matrix_rank(pts_on[1:] - pts_on[0], tol=rank_tol) >= dim - 1:
            keep[row_idx] = True
    return Polytope(points, axes=axes, hrep=(A[keep], b[keep])).reduce_to_hull()


def _merge_parallel_rows(A: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> tuple:
    """Merge identical unit normals, keeping the tightest offset.

    Only normals within ``1e-12`` inf-norm with offsets within ``tol * scale``
    are merged, preserving distinct nearly-parallel facets.
    """
    from scipy.spatial import cKDTree

    # Bitwise-identical unit normals point in the exact same direction, so any
    # looser offset is unconditionally redundant regardless of how far out it sits.
    order = np.argsort(b, kind="stable")
    A_sorted, b_sorted = A[order], b[order]
    _, first_idx = np.unique(A_sorted, axis=0, return_index=True)
    exact_keep = np.sort(first_idx)
    A, b = A_sorted[exact_keep], b_sorted[exact_keep]

    offset_scale = max(1.0, float(np.abs(b).max()))
    rows = np.hstack([A, (b / offset_scale)[:, None]])
    radius = tol
    keep = np.ones(len(rows), dtype=bool)
    tree = cKDTree(rows)
    for i in range(len(rows)):               # already sorted tightest first
        if keep[i]:
            nearby = np.asarray(tree.query_ball_point(rows[i], radius), dtype=np.intp)
            parallel = nearby[np.max(np.abs(A[nearby] - A[i]), axis=1) <= 1e-12]
            keep[parallel[parallel != i]] = False
    return A[keep], b[keep]


def _is_bounded_cone(A: np.ndarray) -> bool:
    """Whether ``{d : A d <= 0} == {0}``, via Stiemke's theorem (``A^T z = 0, z >= 1``)."""
    from scipy.optimize import linprog

    if len(A) <= A.shape[1] or np.linalg.matrix_rank(A) < A.shape[1]:
        return False
    res = linprog(
        np.zeros(len(A)),
        A_eq=A.T,
        b_eq=-A.sum(axis=0),
        bounds=(0.0, None),
        method="highs",
    )
    return bool(res.status == 0)


def _enumerate_vertices(halfspaces: np.ndarray, seed: np.ndarray, *, what: str) -> np.ndarray:
    """Vertices of ``{x : [A | -b] [x; 1] <= 0}`` from a strictly interior seed.

    ``HalfspaceIntersection`` computes a convex hull in the dual space, and on
    regions produced by repeated intersection that hull can hit the same Qhull
    topology error as the primal one (nearly coincident dual points).  The
    tolerant option ladder ``_HALFSPACE_OPTIONS`` is tried in turn.  Duplicate
    copies of a vertex, one per combination of facets meeting there, are merged.
    """
    from scipy.spatial import HalfspaceIntersection, QhullError

    from OpenRC.geometry.polytope import _HALFSPACE_OPTIONS

    A, b = halfspaces[:, :-1], -halfspaces[:, -1]
    errors = []
    for options in _HALFSPACE_OPTIONS:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = HalfspaceIntersection(halfspaces.copy(),
                                               np.array(seed, dtype=float, copy=True),
                                               qhull_options=options)
        except QhullError as exc:
            errors.append(f"[{options or 'default'}] {str(exc).splitlines()[0]}")
            continue
        if not np.all(np.isfinite(result.intersections)):
            if not _is_bounded_cone(A):
                raise ValueError(f"{what}: polyhedron is unbounded")
            errors.append(f"[{options or 'default'}] non-finite vertex intersections")
            continue
        points = merge_close_points(result.intersections)
        ok, reason = _enumeration_is_consistent(A, b, points)
        if ok:
            return points
        errors.append(f"[{options or 'default'}] rejected: {reason}")
    if not _is_bounded_cone(A):
        raise ValueError(f"{what}: polyhedron is unbounded")
    raise GeometryNumericsError(
        f"{what} failed numerically although the body has a non-empty interior "
        f"(Chebyshev centre {np.round(seed, 6).tolist()}): " + "; ".join(errors)
    )


def _enumeration_is_consistent(A: np.ndarray, b: np.ndarray, points: np.ndarray) -> tuple:
    """Validate enumerated vertices of ``{A x <= b}``: returns ``(ok, reason)``.

    Checks that every vertex satisfies ``A x <= b`` within tolerance (guarding
    against wide-merge artifacts under ``Qx Q12``) and that hull volume agrees
    with Monte Carlo volume estimation over the bounding box.
    """
    from OpenRC.geometry.polytope import _max_excess, _point_excess_tol, _volume_agrees

    if len(points) <= A.shape[1]:
        return False, f"only {len(points)} vertices"
    worst = _max_excess(np.hstack([A, -b[:, None]]), points)
    if worst > _point_excess_tol(points, float(np.abs(b).max())):
        return False, f"a vertex violates an inequality by {worst:.2e}"
    body = Polytope(points, hrep=(A, b))
    if body.hull is None:
        return False, f"the vertices have no hull ({body.hull_error})"
    lower, upper = points.min(axis=0), points.max(axis=0)
    margin = 0.05 * (upper - lower)
    ok, estimate = _volume_agrees(A, b, lower - margin, upper + margin, body.volume)
    if not ok:
        return False, (f"hull volume {body.volume:.6g} but the inequalities enclose "
                       f"about {estimate:.6g}")
    return True, ""


class EmptyIntersectionError(RuntimeError):
    """Raised when an intersection leaves no full-dimensional body."""


class GeometryNumericsError(RuntimeError):
    """Numerical centring or vertex enumeration could not establish a result.

    Distinct from :class:`EmptyIntersectionError`: raised when LP centring or
    Qhull vertex enumeration fails numerically without proving emptiness.
    """
