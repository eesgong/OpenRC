r"""Incremental convex-hull wrapper (:class:`LiveHull`) for region growth and pruning.

Maintains a single ``scipy.spatial.ConvexHull(..., incremental=True)`` instance
across point insertions and vertex-removal queries to avoid full hull
reconstructions on every candidate evaluation:

``gain(p)``
    Exact hypervolume gain from inserting ``p``, computed directly from
    ``hull.equations`` and ``hull.simplices`` without mutating the hull.
    Inserting ``p`` deletes visible facets and cones every horizon ridge to
    ``p``, giving::

        gain(p) = sum_{F visible from p} |det(F - p)| / d!

``add(p)``
    Qhull's incremental insertion: only the visible facets are rebuilt.

``marginal_loss(i)``
    Volume lost by deleting vertex ``i``.  Deleting a vertex only reshapes the
    hull around it: the facets that fill the hole are the facets of
    ``conv(N(i))`` visible from ``v_i``, where ``N(i)`` are the vertices sharing
    a facet with ``i``.  Hence

    .. math:: \mathrm{loss}(i) = \mathrm{vol}(\mathrm{conv}(N(i) \cup \{v_i\}))
              - \mathrm{vol}(\mathrm{conv}(N(i)))

    which in 3-D is two hulls of ~6 points instead of one of the whole body.
    In high dimension nearly every pair of vertices is adjacent, so the local
    form saves nothing and the full rebuild is used instead (``method="auto"``
    picks per vertex).

``star_volumes()``
    For every vertex, the volume of the cones from an interior point ``c`` over
    the facets containing it.  If ``c`` stays inside ``conv(V \ {v})`` this is
    an **upper bound** on the marginal loss (the lost cap lies inside the
    star); it is one pass over the facets for all vertices at once, which makes
    it the screen that keeps compression affordable in high dimension.

``remove(i)``
    Rebuilds the hull from the remaining vertices when a deletion is committed.

Call :meth:`LiveHull.release` when finished to free native Qhull memory.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Dict, Iterable, Optional

import numpy as np

#: Fallback Qhull options for incremental hulls.  Excludes ``Qbb`` (mutates
#: coordinates in place) and ``QJ`` (perturbs vertex coordinates).
_LIVE_OPTIONS = (None, "Qx", "Qt", "Qx Q12", "Q14")

#: Rows per chunk when stacking ``(d x d)`` matrices for determinants, which
#: bounds the temporary array at ``_CHUNK * d * d * 8`` bytes (32 MiB in 16-D).
_CHUNK = 16384

#: Minimum dimension for multi-threaded batch hull evaluation (Qhull releases
#: the GIL; thread dispatch overhead exceeds hull runtime for dim <= 4).
PARALLEL_FROM_DIM = 5

_workers: Optional[int] = None
#: One pool per requested size, so concurrent callers asking for different
#: thread counts never shut down each other's pool.
_pools: Dict[int, Any] = {}
_pool_lock = threading.Lock()


def _worker_count(value: Any, *, allow_none: bool = True) -> Optional[int]:
    """Validate a thread count with a stable error for NaN and bad types."""
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("workers must be None or a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("workers must be None or a positive integer") from None
    if not math.isfinite(numeric) or numeric < 1.0 or int(numeric) != numeric:
        raise ValueError("workers must be None or a positive integer")
    return int(numeric)


def shutdown_geometry_workers(*, wait: bool = True) -> None:
    """Release the shared geometry thread pool.

    Long-lived services and notebooks can call this after a large compression
    or relocation job.  A later parallel operation creates a fresh pool.  The
    pool is detached under the lock and shut down afterwards so worker teardown
    never blocks another thread while it holds OpenRC's global pool lock.
    """
    if not isinstance(wait, (bool, np.bool_)):
        raise ValueError("wait must be a boolean")
    with _pool_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.shutdown(wait=bool(wait), cancel_futures=False)


def set_geometry_workers(workers: Optional[int]) -> None:
    """Threads for batches of independent exact hulls, such as marginal losses.

    The results are identical, in the same order, whatever the setting.
    ``None`` (default) uses ``min(8, cores)``, or the ``OPENRC_GEOMETRY_WORKERS``
    environment variable when it is set; ``1`` computes everything in the
    calling thread.
    """
    global _workers
    selected = _worker_count(workers)
    if selected != _workers:
        _workers = selected
        # In particular, switching to one worker must not leave the previous
        # executor and all of its threads resident forever.
        shutdown_geometry_workers()


def geometry_workers() -> int:
    """The number of threads :func:`set_geometry_workers` currently gives."""
    if _workers is not None:
        return _workers
    env = os.environ.get("OPENRC_GEOMETRY_WORKERS")
    if env:
        try:
            selected = _worker_count(env, allow_none=False)
        except ValueError:
            raise ValueError(
                "OPENRC_GEOMETRY_WORKERS must contain a positive integer"
            ) from None
        assert selected is not None
        return selected
    return max(1, min(8, os.cpu_count() or 1))


def _thread_pool(workers: int):
    """The shared pool of ``workers`` threads (created on first use)."""
    from concurrent.futures import ThreadPoolExecutor

    with _pool_lock:
        pool = _pools.get(workers)
        if pool is None:
            pool = _pools[workers] = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="openrc-geometry")
        return pool


def parallel_map(fn, items, workers: Optional[int] = None) -> list:
    """``[fn(x) for x in items]``, in the shared pool when it has several threads.

    ``fn`` must be free of side effects (a hull, a volume).  If another thread
    shuts the pool down between pool lookup and task submission - through
    :func:`shutdown_geometry_workers` or :func:`set_geometry_workers` - the rest
    is computed in the calling thread; the results are the same either way.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    selected = _worker_count(workers)
    items = list(items)
    count = geometry_workers() if selected is None else selected
    if count <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    futures: list = []
    try:
        pool = _thread_pool(count)
        for x in items:
            futures.append(pool.submit(fn, x))
    except RuntimeError:            # "cannot schedule new futures after shutdown"
        return ([f.result() for f in futures]
                + [fn(x) for x in items[len(futures):]])
    return [f.result() for f in futures]


def live_hull_bytes_per_facet(dim: int) -> float:
    """Estimated resident memory per triangulated facet of an incremental Qhull, in bytes.

    Combines SciPy array storage (``8*(d+1)`` equation bytes + ``8*d`` simplex
    and neighbour index bytes) with Qhull internal facet structures
    (``~24*d + 100`` bytes), giving ``40*d + 108`` bytes/facet.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)) or int(dim) < 1:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    return 40.0 * int(dim) + 108.0


def live_hull_memory_mb(n_points: int, dim: int, n_facets: int) -> float:
    """Modelled resident memory of a live hull, in MiB (see above)."""
    if (isinstance(n_points, (bool, np.bool_))
            or not isinstance(n_points, (int, np.integer))
            or int(n_points) < 0):
        raise ValueError(f"n_points must be a non-negative integer, got {n_points!r}")
    if (isinstance(n_facets, (bool, np.bool_))
            or not isinstance(n_facets, (int, np.integer))
            or int(n_facets) < 0):
        raise ValueError(f"n_facets must be a non-negative integer, got {n_facets!r}")
    return (8.0 * int(n_points) * int(dim) + live_hull_bytes_per_facet(dim) * int(n_facets)) / 2 ** 20


def cone_volumes(points: np.ndarray, simplices: np.ndarray, apex: np.ndarray) -> np.ndarray:
    """``|det(points[s] - apex)| / d!`` for every simplex row ``s``, chunked."""
    raw_pts = np.asarray(points)
    raw_simp = np.asarray(simplices)
    raw_apex = np.asarray(apex)
    if raw_pts.dtype == bool or raw_simp.dtype == bool or raw_apex.dtype == bool:
        raise ValueError("points, simplices and apex must not be boolean arrays")
    pts = np.asarray(raw_pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] == 0:
        raise ValueError("points must be a 2-D numeric array with d >= 1")
    dim = pts.shape[1]
    simp = np.asarray(raw_simp, dtype=np.intp)
    if simp.ndim != 2 or simp.shape[1] != dim:
        raise ValueError(f"simplices must be a 2-D integer array with {dim} columns")
    apex_pt = np.asarray(raw_apex, dtype=float)
    if apex_pt.shape != (dim,):
        raise ValueError(f"apex must be a 1-D vector of length {dim}")
    out = np.empty(len(simp))
    scale = 1.0 / math.factorial(dim)
    for start in range(0, len(simp), _CHUNK):
        block = pts[simp[start:start + _CHUNK]] - apex_pt[None, None, :]
        out[start:start + _CHUNK] = np.abs(np.linalg.det(block)) * scale
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)


def _volume(points: np.ndarray) -> Optional[float]:
    """Hull volume, ``0.0`` for a flat set, ``None`` when Qhull fails outright."""
    from scipy.spatial import ConvexHull, QhullError

    from OpenRC.geometry.polytope import _hull_geometry_consistent

    if len(points) <= points.shape[1]:
        return 0.0
    if points.shape[1] == 1:
        return float(np.ptp(points[:, 0]))
    for options in _LIVE_OPTIONS:
        try:
            hull = ConvexHull(points.copy(), qhull_options=options)
        except QhullError as exc:
            if _is_flat(exc):
                return 0.0
            continue
        if options is not None and not _hull_geometry_consistent(hull, points):
            continue
        return float(hull.volume)
    return None


def hull_volume_and_facets(points: np.ndarray) -> Optional[tuple]:
    """``(volume, n_facets)`` of a fresh hull; ``None`` when Qhull fails or it is flat."""
    from scipy.spatial import ConvexHull, QhullError

    from OpenRC.geometry.polytope import _hull_geometry_consistent

    if len(points) <= points.shape[1]:
        return None
    if points.shape[1] == 1:
        span = float(np.ptp(points[:, 0]))
        return (span, 2) if span > 0.0 else None
    for options in _LIVE_OPTIONS:
        try:
            hull = ConvexHull(points.copy(), qhull_options=options)
        except QhullError as exc:
            if _is_flat(exc):
                return None
            continue
        if options is not None and not _hull_geometry_consistent(hull, points):
            continue
        return float(hull.volume), int(len(hull.simplices))
    return None


def _merged_near(qh, visible: np.ndarray) -> bool:
    """Whether a facet an insertion touches shares its hyperplane with a neighbour.

    Qhull merges coplanar facets and reports the pieces of a merged facet with
    one equation.  Only the visible facets and their neighbours are examined,
    which keeps the test local to the insertion.
    """
    if not visible.any():
        return False
    neighbors = qh.neighbors
    touched = np.flatnonzero(visible)
    across = neighbors[touched]
    touched = np.union1d(touched, across[across >= 0])
    eq = qh.equations
    step = max(1, 2 ** 21 // (neighbors.shape[1] * eq.shape[1]))   # ~16 MiB per block
    for first in range(0, len(touched), step):
        rows = touched[first:first + step]
        nbr = neighbors[rows]
        valid = nbr >= 0
        same = np.all(eq[np.where(valid, nbr, 0)] == eq[rows][:, None, :], axis=2) & valid
        if same.any():
            return True
    return False


def _is_flat(exc: Exception) -> bool:
    """Whether a Qhull error means "no full-dimensional body" rather than numerics."""
    message = str(exc).lower()
    return "flat" in message or "initial simplex" in message


class LiveHull:
    """One Qhull object kept alive across insertions and deletions.

    Parameters
    ----------
    points:
        Initial point set (``n > dim``, full-dimensional).
    compact_ratio:
        Insertions leave points that fell inside the hull in Qhull's point
        list.  Once there are more than ``compact_ratio`` interior points per
        vertex the hull is rebuilt from its vertices, which returns their
        memory.

    Positions
    ---------
    Everything indexed by *position* refers to :attr:`vertices`, i.e. to the
    hull vertices in their current order.  Qhull's own point indices (which
    include interior points) never leave this class.
    """

    def __init__(self, points, *, compact_ratio: float = 1.0) -> None:
        raw = np.asarray(points)
        if raw.dtype == bool or raw.ndim != 2 or raw.shape[1] == 0:
            raise ValueError("points must be a 2-D (n, dim) array")
        pts = np.asarray(raw, dtype=float)
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must be finite")
        if (isinstance(compact_ratio, (bool, np.bool_))
                or compact_ratio <= 0 or not math.isfinite(compact_ratio)):
            raise ValueError("compact_ratio must be finite and positive")
        self.dim = int(pts.shape[1])
        self.compact_ratio = float(compact_ratio)
        self.stats = {"builds": 0, "inserts": 0, "insert_fallbacks": 0,
                      "guarded_inserts": 0, "volume_repairs": 0, "removed": 0,
                      "compactions": 0}
        self.options: Optional[str] = None
        self._qh: Any = None           # scipy ConvexHull in incremental mode
        self._build(pts)

    # ------------------------------------------------------------ building
    def _build(self, pts: np.ndarray) -> None:
        from scipy.spatial import ConvexHull, QhullError

        from OpenRC.geometry.polytope import _Interval1DHull, _hull_matches_points

        if len(pts) <= self.dim:
            raise ValueError(f"{len(pts)} points cannot span a {self.dim}-D body")
        if self.dim == 1:
            try:
                qh_1d = _Interval1DHull(pts)
            except QhullError as exc:
                raise ValueError(
                    f"the points are degenerate (zero 1-D volume): {exc}"
                ) from exc
            self._install(qh_1d, None)
            return
        errors = []
        for options in _LIVE_OPTIONS:
            try:
                qh = ConvexHull(pts.copy(), incremental=True, qhull_options=options)
            except QhullError as exc:
                message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                if _is_flat(exc):
                    raise ValueError(
                        f"the points are degenerate (zero {self.dim}-D volume): {message}"
                    ) from exc
                errors.append(f"[{options or 'default'}] {message}")
                continue
            if options is not None:
                ok, reason = _hull_matches_points(qh, pts)
                if not ok:
                    qh.close()
                    errors.append(f"[{options}] rejected: {reason}")
                    continue
            self._install(qh, options)
            return
        from OpenRC.geometry.ops import GeometryNumericsError

        raise GeometryNumericsError("Qhull could not build a live hull: " + "; ".join(errors))

    def _install(self, qh, options: Optional[str]) -> None:
        if self._qh is not None:
            self._qh.close()
        self._qh = qh
        self.options = options
        self.stats["builds"] += 1
        self._refresh()

    def _refresh(self) -> None:
        qh = self._qh
        self._vertex_ids = np.sort(np.asarray(qh.vertices, dtype=np.intp))
        self._positions: Optional[np.ndarray] = None
        self._volume = float(qh.volume)

    def reset(self, points) -> np.ndarray:
        """Replace the whole point set (one rebuild).

        Returns the indices of ``points`` that are hull vertices, in position
        order - so arrays kept alongside the vertices can follow them.
        """
        raw = np.asarray(points)
        if raw.dtype == bool or raw.ndim != 2 or raw.shape[1] != self.dim:
            raise ValueError(
                f"points must be a 2-D numeric array with dimension {self.dim}, got {raw.shape}"
            )
        pts = np.asarray(raw, dtype=float)
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must be finite")
        self._build(pts)
        return self._vertex_ids.copy()

    def release(self) -> None:
        """Free Qhull's memory.  The object is unusable afterwards."""
        if self._qh is not None:
            self._qh.close()
            self._qh = None

    def _require(self):
        if self._qh is None:
            raise RuntimeError("this LiveHull has been released")
        return self._qh

    def _check_position(self, position) -> int:
        self._require()
        if isinstance(position, (bool, np.bool_)) or not isinstance(position, (int, np.integer)):
            raise ValueError("vertex positions must be integers")
        pos = int(position)
        if pos < 0 or pos >= self.n_vertices:
            raise IndexError("vertex position out of range")
        return pos

    def _check_point(self, point) -> np.ndarray:
        raw = np.asarray(point)
        if raw.dtype == bool:
            raise ValueError(f"point must be a finite ({self.dim},) vector")
        p = np.asarray(raw, dtype=float)
        if p.shape != (self.dim,) or not np.all(np.isfinite(p)):
            raise ValueError(f"point must be a finite ({self.dim},) vector")
        return p

    # ----------------------------------------------------------- read-only
    @property
    def vertices(self) -> np.ndarray:
        """Coordinates of the hull vertices (a copy), in position order."""
        return self._require().points[self._vertex_ids]

    @property
    def n_vertices(self) -> int:
        return int(len(self._vertex_ids))

    @property
    def n_points(self) -> int:
        """Points Qhull holds, including ones that have fallen inside."""
        return int(self._require().npoints)

    @property
    def n_facets(self) -> int:
        """Number of triangulated hull simplices."""
        return int(len(self._require().simplices))

    @property
    def volume(self) -> float:
        return self._volume

    def memory_mb(self) -> float:
        """Modelled resident memory of this hull (:func:`live_hull_memory_mb`)."""
        return live_hull_memory_mb(self.n_points, self.dim, self.n_facets)

    @property
    def points(self) -> np.ndarray:
        """Every point Qhull holds (vertices and fallen-inside points); read-only."""
        view = self._require().points.view()
        view.flags.writeable = False            # the view only; Qhull's array is untouched
        return view

    @property
    def simplices(self) -> np.ndarray:
        """``(n_facets, dim)`` indices into :attr:`points` of every triangulated facet."""
        view = self._require().simplices.view()
        view.flags.writeable = False
        return view

    @property
    def equations(self) -> np.ndarray:
        """``(n_facets, dim + 1)`` facet planes ``[n | c]`` with ``n.x + c <= 0`` inside."""
        view = self._require().equations.view()
        view.flags.writeable = False
        return view

    def facet_positions(self) -> np.ndarray:
        """``(n_facets, dim)`` vertex positions of every triangulated facet."""
        qh = self._require()
        if self._positions is None:
            lookup = np.full(qh.npoints, -1, dtype=np.intp)
            lookup[self._vertex_ids] = np.arange(len(self._vertex_ids))
            self._positions = lookup[qh.simplices]
            self._positions.flags.writeable = False
        return self._positions

    def neighbours(self, position: int) -> np.ndarray:
        """Positions of the vertices sharing at least one facet with ``position``."""
        pos = self._check_position(position)
        if self.dim == 1:
            return np.array([1 - pos], dtype=np.intp)
        rows = self.facet_positions()
        touching = rows[np.any(rows == pos, axis=1)]
        out = np.unique(touching)
        return out[out != pos]

    # ---------------------------------------------------------- insertions
    def gain(self, point) -> float:
        """Return the exact hypervolume added by ``point`` without mutating the hull."""
        qh = self._require()
        p = self._check_point(point)
        eq = qh.equations
        visible = np.flatnonzero(eq[:, :-1] @ p + eq[:, -1] > 0.0)
        if len(visible) == 0:
            return 0.0
        return float(cone_volumes(qh.points, qh.simplices[visible], p).sum())

    def facets_after(self, point) -> int:
        """Predict the triangulated facet count after inserting ``point`` without mutating the hull.

        Insertion deletes the facets ``point`` can see and cones each *horizon
        ridge* (shared by a visible and an invisible facet) to the point, one
        new simplex per ridge::

            facets_after = facets - visible + horizon ridges

        Exact for points in general position; used by :meth:`memory_after_mb` to
        check memory budgets prior to insertion.
        """
        qh = self._require()
        p = self._check_point(point)
        if self.dim == 1:
            return 2
        eq = qh.equations
        coord = float(np.abs(qh.points[self._vertex_ids]).max())
        tol = 1e2 * np.finfo(float).eps * coord
        visible = eq[:, :-1] @ p + eq[:, -1] > tol
        n_visible = int(visible.sum())
        if n_visible == 0:
            return self.n_facets
        across = qh.neighbors[visible]
        horizon = int((~visible[across[across >= 0]]).sum())
        return self.n_facets - n_visible + horizon

    def memory_after_mb(self, point) -> float:
        """Modelled memory after inserting ``point`` (see :meth:`facets_after`)."""
        return live_hull_memory_mb(self.n_points + 1, self.dim, self.facets_after(point))

    def add(self, point) -> float:
        """Insert ``point`` incrementally; returns the new volume."""
        from scipy.spatial import QhullError

        qh = self._require()
        p = self._check_point(point).reshape(1, self.dim)
        # Qhull documents the object as unusable after an incremental-update
        # failure.  Snapshot the current vertices before calling into it so
        # the recovery path never reads a potentially corrupted object.
        vertices_before = self.vertices
        # When a new point is nearly coplanar with an existing facet while outside
        # another visible facet, Qhull incremental mode can under-report hull.volume;
        # verify against cone_volumes() and rebuild from scratch on mismatch.
        eq = qh.equations
        distance = eq[:, :-1] @ p[0] + eq[:, -1]
        expected = None
        scale = max(1.0, float(np.abs(vertices_before).max()))
        if distance.max() > 0.0:
            if np.any(np.abs(distance) <= 1e-9 * scale):
                gain = cone_volumes(qh.points, qh.simplices[distance > 0.0], p[0]).sum()
                expected = self._volume + float(gain)
        # SciPy re-triangulates the hull after an incremental insertion outside
        # Qhull's error guard: a Qhull topology error there - possible once
        # facets merge, i.e. on (near-)coplanar input - calls exit() and ends
        # the interpreter.  Such insertions take the guarded path instead (a
        # rebuild with the same options: the same hull, failures raised as
        # QhullError); ordinary ones stay incremental.
        # Qhull merges within a few DISTround (about (d^2 + 1) eps scale); 1e3 times
        # that still routes exact and near-exact coplanarity through the guard,
        # while bisection points that merely sit near a flat face stay incremental.
        near = 1e3 * (self.dim ** 2 + 1) * np.finfo(float).eps * scale
        guarded = bool(np.any(np.abs(distance) <= near)) or _merged_near(
            qh, distance > 0.0)
        try:
            qh.add_points(p.copy(), restart=guarded)
            if guarded:
                # A rebuild: caches keyed on incremental facet updates must not
                # assume the surviving facets kept their triangulation.
                self.stats["guarded_inserts"] += 1
                self.stats["builds"] += 1
            self.stats["inserts"] += 1
            self._refresh()
            if expected is not None and not _close(self._volume, expected):
                self.stats["volume_repairs"] += 1
                self._build(np.vstack([vertices_before, p]))
        except QhullError:
            # A precision failure mid-insertion leaves Qhull's state unusable;
            # rebuild from the vertices plus the new point with the ladder.
            from OpenRC.geometry.ops import GeometryNumericsError

            self.stats["insert_fallbacks"] += 1
            try:
                self._build(np.vstack([vertices_before, p]))
            except GeometryNumericsError:
                # No mode hulls the enlarged set.  Leave the hull exactly as it
                # was before the insertion, so the caller keeps a valid region.
                self._build(vertices_before)
                raise
        if self.n_points - self.n_vertices > self.compact_ratio * self.n_vertices:
            self.compact()
        return self._volume

    def compact(self) -> None:
        """Rebuild from the vertices only, dropping interior points."""
        if self.n_points > self.n_vertices:
            self.stats["compactions"] += 1
            self._build(self.vertices)

    # ------------------------------------------------------------ removals
    def remove(self, positions: Iterable[int]) -> np.ndarray:
        """Delete the given vertex positions (one rebuild).

        Returns the old vertex positions retained on the rebuilt hull, in new
        position order.
        """
        self._require()
        if isinstance(positions, (str, bytes, bytearray)):
            raise TypeError("positions must be an iterable of integer vertex positions, not str or bytes")
        raw_positions = list(positions)
        for pos in raw_positions:
            if isinstance(pos, (bool, np.bool_)) or not isinstance(pos, (int, np.integer)):
                raise ValueError("vertex positions must be integers")
        drop = np.unique(np.asarray(raw_positions, dtype=np.intp))
        if len(drop) == 0:
            return np.arange(self.n_vertices)
        if int(drop[0]) < 0 or int(drop[-1]) >= self.n_vertices:   # np.unique sorts
            raise IndexError("vertex position out of range")
        keep = np.ones(self.n_vertices, dtype=bool)
        keep[drop] = False
        survivors = np.flatnonzero(keep)
        self._build(self.vertices[keep])
        self.stats["removed"] += int(len(drop))
        return survivors[self._vertex_ids]

    def marginal_loss(self, position: int, *, method: str = "auto") -> float:
        """Exact volume lost by deleting vertex ``position``.

        ``method``: ``"local"`` (two hulls of the vertex and its neighbours),
        ``"full"`` (one hull of everything else) or ``"auto"`` (local while the
        neighbourhood is under half the vertex set).
        """
        if method not in ("auto", "local", "full"):
            raise ValueError("method must be 'auto', 'local' or 'full'")
        position = self._check_position(position)
        vertices = self.vertices
        if method != "full":
            nbrs = self.neighbours(position)
            if method == "local" or len(nbrs) + 1 < 0.5 * len(vertices):
                ring = vertices[nbrs]
                with_v = _volume(np.vstack([ring, vertices[position]]))
                without = _volume(ring)
                if with_v is not None and without is not None and with_v > 0.0:
                    return max(with_v - without, 0.0)
        rest = _volume(np.delete(vertices, position, axis=0))
        if rest is None:
            return self._volume            # unmeasurable: never the cheapest
        return max(self._volume - rest, 0.0)

    def marginal_losses(self, positions: Optional[Iterable[int]] = None, *,
                        method: str = "auto",
                        workers: Optional[int] = None) -> np.ndarray:
        """:meth:`marginal_loss` for several positions (all by default).

        Each loss is an independent hull, so from :data:`PARALLEL_FROM_DIM` up
        they run in ``workers`` threads (default :func:`geometry_workers`).
        The result is identical either way.
        """
        self._require()
        if method not in ("auto", "local", "full"):
            raise ValueError("method must be 'auto', 'local' or 'full'")
        if isinstance(positions, (str, bytes, bytearray)):
            raise TypeError("positions must be an iterable of integer vertex positions, not str or bytes")
        chosen = (list(range(self.n_vertices)) if positions is None
                  else [self._check_position(i) for i in positions])
        selected = _worker_count(workers)
        n_workers = geometry_workers() if selected is None else selected
        if n_workers > 1 and len(chosen) > 1 and self.dim >= PARALLEL_FROM_DIM:
            self.facet_positions()          # fill the shared cache before the threads read it
            losses = parallel_map(lambda i: self.marginal_loss(i, method=method), chosen,
                                  n_workers)
        else:
            losses = [self.marginal_loss(i, method=method) for i in chosen]
        return np.array(losses, dtype=float)

    def star_volumes(self, centre: Optional[np.ndarray] = None) -> np.ndarray:
        """Per vertex, the volume of the cones from ``centre`` over its facets.

        ``centre`` defaults to the mean of the vertices.  One pass over the
        facets serves every vertex.
        """
        qh = self._require()
        if centre is None:
            c = self.vertices.mean(axis=0)
        else:
            raw_c = np.asarray(centre)
            if raw_c.dtype == bool:
                raise ValueError(f"centre must be a finite ({self.dim},) vector")
            c = np.asarray(raw_c, dtype=float)
            if c.shape != (self.dim,) or not np.all(np.isfinite(c)):
                raise ValueError(f"centre must be a finite ({self.dim},) vector")
        cones = cone_volumes(qh.points, qh.simplices, c)
        rows = self.facet_positions()
        return np.bincount(rows.ravel(), weights=np.repeat(cones, self.dim),
                           minlength=self.n_vertices)

    def __repr__(self) -> str:
        state = "released" if self._qh is None else (
            f"{self.n_vertices} vertices, {self.n_facets} facets, volume {self._volume:.4g}")
        return f"<LiveHull dim={self.dim} {state}>"
