r"""Banach fixed-point existence of power-flow solutions over a polytope.

The global certificate (:mod:`OpenRC.validation.certify`) bounds voltages and
currents for all branch-flow solutions within a voltage window, but does not by
itself establish that a power-flow solution exists at every injection in the
region.  The
surface sample checks existence at finitely many points and bridges the gaps
with a continuity/no-fold assumption.  This module removes that assumption
wherever a classical contraction condition holds.

The fixed-point form of power flow
----------------------------------
The certificate's model is a radial network of series impedances with the
slack held at 1 p.u. (per unit on 1 MVA, as in
:class:`~OpenRC.powerflow.exact.ExactDistFlowOracle`).  With consumption
``s_j = p_j + i q_j`` at the non-slack buses, the bus voltages solve

.. math::  V = \mathbf{1} - Z\,\overline{s} \oslash \overline{V},

where ``Z = Y_LL^{-1}``; for a radial network ``Z_ij`` is the sum of the
impedances of the branches that the paths from the slack to ``i`` and to ``j``
share.  Write ``T(V)`` for the right-hand side and

.. math::  \xi(s) = \max_i \sum_j |Z_{ij}|\,|s_j| .

On the ball ``B_rho = {V : max_i |V_i - 1| <= rho}`` with ``rho < 1`` one has
``|V_j| >= 1 - rho``, hence

* ``T`` maps ``B_rho`` into itself when ``xi / (1 - rho) <= rho``, i.e.
  ``xi <= rho (1 - rho)``, which has a solution iff ``xi <= 1/4``;
* ``T`` is a contraction on ``B_rho`` with constant ``xi / (1 - rho)^2``.

For ``xi < 1/4`` the smallest admissible radius
``rho* = (1 - sqrt(1 - 4 xi)) / 2`` satisfies both, so by Banach's fixed-point
theorem **a solution exists with every ``|V_i - 1| <= rho*``** (and it is the
only one in that ball).  This is the argument of Bolognani & Zampieri (2016)
and Wang et al. (2018) in its simplest, infinity-norm form.  With the slack at
``v0`` p.u. (``case.slack_vm_pu``) the map is ``V = v0 - Z s_bar / V_bar``: the
condition becomes ``xi < v0**2 / 4``, the radius
``rho* = 2 xi / (v0 + sqrt(v0**2 - 4 xi))`` and the band
``[v0 - rho*, v0 + rho*]``.

From one point to the whole region
----------------------------------
The injection is affine in the region coordinates, ``|s_j|`` is convex in
``(p_j, q_j)``, and a sum and a maximum of convex functions is convex - so
``xi`` is convex over the region and its maximum is attained at a **vertex**.
Checking the vertices therefore proves the condition, and the voltage band
``[1 - rho*, 1 + rho*]``, at every point of the polytope.

If that band lies inside the certificate's voltage window, every point of the
region has a solution inside the window, and the certificate's limit bounds
apply to it: the region is inner, with no continuity assumption.  When the
condition fails (``xi >= 1/4`` at some vertex, or a band wider than the
window) nothing is concluded and the surface sample remains the evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import List, Optional, Tuple

import numpy as np

from OpenRC.geometry.polytope import Polytope


@dataclass
class ExistenceProof:
    """Outcome of :func:`prove_existence`.

    ``proven`` means: at every point of the region a power-flow solution
    exists with every ``|V_i - v0| <= radius`` (``v0`` the slack voltage), and
    that band lies inside ``window`` (when a window was given).
    """

    proven: bool
    xi_max: float                         # max over the region of xi (at a vertex)
    radius: float                         # rho*: nan when xi_max >= v0**2/4
    band: Tuple[float, float]             # implied |V| range
    window: Optional[Tuple[float, float]]
    worst_vertex: np.ndarray
    n_vertices: int
    reason: str

    def describe(self) -> str:
        if self.proven:
            v0 = 0.5 * (self.band[0] + self.band[1])       # the band is centred on v0
            if math.isclose(v0, 1.0, rel_tol=0.0, abs_tol=1e-12):
                condition, centre = "1/4", "1"
            else:
                condition, centre = f"v0^2/4 = {v0 * v0 / 4:.4f}", f"{v0:g}"
            return (f"proven at every point by contraction (xi <= {self.xi_max:.4f} < "
                    f"{condition} over {self.n_vertices} vertices; |V - {centre}| <= "
                    f"{self.radius:.4f}, band {self.band[0]:.3f}-{self.band[1]:.3f} pu)")
        return f"not proven by contraction: {self.reason}"

    def as_dict(self) -> dict:
        return {
            "proven": self.proven, "xi_max": self.xi_max, "radius": self.radius,
            "band": list(self.band), "window": None if self.window is None else list(self.window),
            "worst_vertex": self.worst_vertex.tolist(), "n_vertices": self.n_vertices,
            "reason": self.reason,
        }


def impedance_matrix(case) -> Tuple[List[int], np.ndarray]:
    """``(buses, Z)``: the non-slack buses and ``Z = Y_LL^{-1}`` in per unit.

    For a radial network of series impedances, ``Z_ij`` is the impedance of the
    path the slack shares with both ``i`` and ``j``.  Per unit on 1 MVA
    three-phase, the base of the certificate and the exact oracle.  Rows are
    built in topological order: a child shares its parent's paths to all
    previously visited buses, and adds its own branch on the diagonal.  This
    takes O(n_bus**2) work/storage, without pairwise ancestor-set intersections.
    """
    from OpenRC.cases.base import validate_case_configuration

    validate_case_configuration(case)
    topo = case.topology
    v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
    z_base = v_base ** 2 / (1.0e6 / 3.0)
    buses = list(topo.buses)
    branches = list(topo.branches)
    row_of_child = {branch.child: k for k, branch in enumerate(branches)}
    if (len(branches) != len(buses) or len(row_of_child) != len(branches)
            or set(row_of_child) != set(buses) or topo.slack in row_of_child
            or any(branch.parent != topo.slack
                   and row_of_child.get(branch.parent, len(branches)) >= k
                   for k, branch in enumerate(branches))):
        raise ValueError("impedance construction requires a complete parent-before-child tree")
    position = {bus: k for k, bus in enumerate(buses)}
    Z = np.zeros((len(buses), len(buses)), dtype=complex)
    for branch in branches:
        child = position[branch.child]
        parent = position.get(branch.parent)
        if parent is not None:
            Z[child, :] = Z[parent, :]
        Z[:, child] = Z[child, :]
        upstream = 0.0j if parent is None else Z[parent, parent]
        Z[child, child] = upstream + complex(branch.r_ohm, branch.x_ohm) / z_base
    return buses, Z


def contraction_constant(case, axes, points: np.ndarray) -> np.ndarray:
    """``xi`` per point, as a vector of shape ``(n_points,)``.

    Accept one point of shape ``(len(axes),)`` or a matrix with that many
    columns.  Empty matrices return immediately, without preparing a network.
    """
    if axes is None or isinstance(axes, (str, bytes, bytearray)):
        raise TypeError("axes must be a sequence of Axis objects, not None or a string")
    raw_points = np.asarray(points)
    if raw_points.dtype == bool:
        raise ValueError("points must be a numeric array, not boolean")
    points = np.asarray(raw_points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != len(axes):
        raise ValueError(f"points must have shape ({len(axes)},) or (n, {len(axes)})")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite coordinates")
    if len(points) == 0:
        return np.empty(0, dtype=float)
    base_p, base_q = case.injection_vector(axes)
    base_q = case.branch_flow_q(base_q)
    buses, Z = impedance_matrix(case)
    weights = np.abs(Z)
    out = np.empty(len(points), dtype=float)
    for first in range(0, len(points), 512):
        chunk = points[first:first + 512]
        p = np.repeat(base_p[None, :], len(chunk), axis=0)
        q = np.repeat(base_q[None, :], len(chunk), axis=0)
        for k, axis in enumerate(axes):
            (p if axis.quantity == "p" else q)[:, axis.bus] += chunk[:, k]
        s = np.hypot(p[:, buses], q[:, buses])           # MW/Mvar = per unit on 1 MVA
        out[first:first + len(chunk)] = (s @ weights.T).max(axis=1, initial=0.0)
    return out


def prove_existence(region, case, axes=None, *,
                    window: Optional[Tuple[float, float]] = None,
                    safety: float = 1e-9) -> ExistenceProof:
    """Prove that every point of ``region`` has a power-flow solution.

    Parameters
    ----------
    region:
        A :class:`~OpenRC.geometry.polytope.Polytope`; only its vertices are
        evaluated (the condition is convex, see the module documentation).
    window:
        ``(v_low, v_high)`` the voltage band must fit into - the certificate's
        window.  Bounds must be finite, positive and strictly increasing.
        ``None`` checks the contraction condition only.
    safety:
        Relative margin kept below ``v0**2/4`` and inside the window, against
        rounding in the evaluation; finite and in ``[0, 1)``.  Zero explicitly
        disables this additional floating-point safety margin.
    """
    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    if isinstance(safety, (bool, np.bool_)) or not isinstance(safety, (int, float, Real)):
        raise ValueError("safety must be finite and in [0, 1)")
    try:
        safety = float(safety)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("safety must be finite and in [0, 1)") from exc
    if not np.isfinite(safety) or not 0.0 <= safety < 1.0:
        raise ValueError("safety must be finite and in [0, 1)")
    if window is not None:
        if isinstance(window, (str, bytes, bytearray)):
            raise ValueError("window must be a finite positive (low, high) pair")
        try:
            items = list(window)
            if any(isinstance(v, (bool, np.bool_)) for v in items):
                raise ValueError("window must be a finite positive (low, high) pair")
            bounds = np.asarray(items, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("window must be a finite positive (low, high) pair") from exc
        if (bounds.shape != (2,) or not np.all(np.isfinite(bounds))
                or not 0.0 < bounds[0] < bounds[1]):
            raise ValueError("window must be a finite positive (low, high) pair")
        window = (float(bounds[0]), float(bounds[1]))
    axes = axes if axes is not None else region.axes
    if axes is None:
        raise ValueError("the region carries no axes; pass them explicitly")
    region.validate_axes(axes)
    if not case.is_radial:
        return ExistenceProof(False, math.nan, math.nan, (math.nan, math.nan), window,
                              np.zeros(region.dim), region.n_vertices,
                              "the network is not radial")
    v0 = float(case.slack_vm_pu)        # the slack voltage; 1 p.u. by default
    vertices = region.vertices
    xi = contraction_constant(case, axes, vertices)
    worst = int(np.argmax(xi))
    xi_max = float(xi[worst])
    common = dict(xi_max=xi_max, window=window, worst_vertex=vertices[worst].copy(),
                  n_vertices=len(vertices))
    if not xi_max < 0.25 * v0 * v0 * (1.0 - safety):
        return ExistenceProof(False, radius=math.nan, band=(math.nan, math.nan),
                              reason=(f"xi reaches {xi_max:.4f} >= 1/4 at a vertex" if v0 == 1.0
                                      else f"xi reaches {xi_max:.4f} >= v0^2/4 = "
                                           f"{0.25 * v0 * v0:.4f} at a vertex"), **common)
    # Rationalise the small quadratic root: v0 - sqrt(v0**2 - 4*xi) cancels
    # completely at tiny nonzero injections and would claim a zero-radius ball.
    radius = 2.0 * xi_max / (v0 + math.sqrt(v0 * v0 - 4.0 * xi_max))
    band = (v0 - radius, v0 + radius)
    if window is not None and not (band[0] >= window[0] * (1 + safety)
                                   and band[1] <= window[1] * (1 - safety)):
        return ExistenceProof(False, radius=radius, band=band,
                              reason=(f"the guaranteed band {band[0]:.3f}-{band[1]:.3f} pu "
                                      f"is wider than the window {window[0]:.2f}-"
                                      f"{window[1]:.2f} pu"), **common)
    return ExistenceProof(True, radius=radius, band=band, reason="", **common)
