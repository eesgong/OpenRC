r"""Backward/forward sweep with a rigorous error bound: a fast feasibility proof.

The exact branch-flow model of :class:`~OpenRC.powerflow.exact.ExactDistFlowOracle`
- a radial network of series impedances, the slack at ``v0``
(``case.slack_vm_pu``, 1 p.u. by default), per unit on a 1 MVA three-phase
base - is, in bus-voltage form (shown for ``v0 = 1``),

.. math::  V = \mathbf 1 - Z\,\overline{s} \oslash \overline{V},
           \qquad Z = T^{\top} \operatorname{diag}(z)\, T ,

with ``T`` the branch-by-bus subtree incidence and ``z`` the branch
impedances.  Each **sweep** evaluates the right-hand side ``F(V)`` in closed
form: the bus currents ``I = conj(s / V)``, the branch currents ``J = T I``
(backward), the voltages ``V = 1 - T^T (z * J)`` (forward).  The first sweep
from a flat start, ``V_1 = 1 - Z conj(s)``, is itself a closed-form linear
estimate.

For two buses, ``|V_2|^2`` solves a quadratic.  For general radial trees, this
module iterates the sweep map and certifies a fixed point via a Banach
contraction bound.  Sweep until the iterate ``V^`` settles, take one more
sweep, ``delta = |F(V^) - V^|_inf``, and on the ball ``B(V^, r)`` use the
Lipschitz bound of the map (``|V_j| >= |V^_j| - r``)

.. math::  L(r) = \max_i \sum_j |Z_{ij}|\,|s_j| \,/\, (|\hat V_j| - r)^2
           \;\le\; \max_i \bigl[T^{\top}(|z| \circ T(|s| / (|\hat V| - r)^2))\bigr]_i .

Require ``0 < r < min_j |V^_j|``, ``L(r) < 1`` and ``delta <= r (1 - L(r))``.
The map then sends the closed infinity-norm ball into itself and contracts it,
so by Banach's theorem a unique solution ``V*`` exists in ``B(V^, r)`` (local
uniqueness), satisfying ``|V* - V^|_inf <= e = delta / (1 - L(r))``.  The local
bound can succeed where the global condition of :mod:`OpenRC.validation.existence`
(a ball around 1 p.u., fixed a priori) requires ``xi < 1/4``.

The certificate is checked in two stages.  A fast floating-point pass (with a
relative safety factor) screens out points that fail contraction or limits.
Every accepted point is then re-verified when ``certificate="interval"`` (the
default): ``delta``, ``L(r)``, ``e`` and the limit margins are recomputed in
interval arithmetic with outward rounding (:mod:`OpenRC.powerflow.interval`)
from the stored floating-point inputs (``r``, ``x``, ``s``, ``V^``).  When all
interval bounds hold, the Banach contraction inequalities are verified over the
reals (``SweepCertificate.rigorous = True``).  The interval pass takes roughly
0.15-0.25 ms per accepted point (9 to 33 buses); ``certificate="float"`` skips it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from OpenRC.powerflow.equations import STRUCTURED_SOLVERS, EquationResult


@dataclass(frozen=True)
class SweepCertificate:
    """Machine-readable a-posteriori certificate for one converged sweep.

    ``voltage_margin`` and ``current_margin`` already include the fixed-point
    error enclosure.  A non-negative value in both fields is the condition used
    for acceptance.  The dimensional quantities are expressed in per unit.
    """

    residual: float
    radius: float
    lipschitz: float
    error_bound: float
    voltage_margin: float
    current_margin: float
    #: ``True``: every number above was proven in interval arithmetic (outward
    #: rounding), so the conclusion holds for the real numbers.  ``False``:
    #: computed in floating point with a relative safety factor.
    rigorous: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.rigorous, (bool, np.bool_)):
            raise ValueError("rigorous must be a boolean")

    @property
    def limits_proven(self) -> bool:
        """Whether every voltage and rated-current limit is certified."""
        return self.voltage_margin >= 0.0 and self.current_margin >= 0.0

    def as_dict(self) -> dict:
        return {
            "residual": self.residual,
            "radius": self.radius,
            "lipschitz": self.lipschitz,
            "error_bound": self.error_bound,
            "voltage_margin": self.voltage_margin,
            "current_margin": self.current_margin,
            "limits_proven": self.limits_proven,
            "rigorous": self.rigorous,
        }


class RadialSweep:
    """Certified sweeps of the exact radial model of one case.

    Parameters
    ----------
    case:
        A radial :class:`~OpenRC.cases.base.GridCase`.
    max_sweeps:
        Sweeps before certifying.  Each shrinks the step by roughly the local
        contraction factor, so a handful to a few dozen suffice.
    s_base_va:
        Three-phase power base; 1 MVA, as in the exact oracle.
    certificate:
        ``"interval"`` (default) re-proves every accepted point in interval
        arithmetic with outward rounding (:mod:`OpenRC.powerflow.interval`),
        so the acceptance is a proof for the real numbers, including the unit
        conversions of the data.  ``"float"`` keeps the floating-point
        certificate with its relative safety factor: a little faster, and
        still conservative, but not a formal proof.
    """

    name = "sweep"

    def __init__(self, case, *, max_sweeps: int = 50, s_base_va: float = 1.0e6,
                 certificate: str = "interval") -> None:
        import scipy.sparse as sp
        from OpenRC.cases.base import validate_case_configuration

        if (not isinstance(max_sweeps, (int, np.integer))
                or isinstance(max_sweeps, (bool, np.bool_)) or max_sweeps < 1):
            raise ValueError("max_sweeps must be a positive integer")
        if (
            isinstance(s_base_va, (bool, np.bool_))
            or not isinstance(s_base_va, (int, float, np.number))
            or not np.isfinite(s_base_va)
            or s_base_va <= 0.0
        ):
            raise ValueError("s_base_va must be finite and strictly positive")
        if certificate not in ("interval", "float"):
            raise ValueError("certificate must be 'interval' or 'float'")
        validate_case_configuration(case)
        self.certificate = certificate
        topo = case.topology
        branches = list(topo.branches)
        row = {br.index: k for k, br in enumerate(branches)}
        entries = [(row[br.index], bus) for bus in topo.buses for br in topo.path_to_root(bus)]
        rows, cols = np.array(entries, dtype=int).reshape(-1, 2).T
        self.T: Any = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                    shape=(len(branches), case.n_bus))
        self.Tt = self.T.T.tocsr()
        v_base = case.v_nom_kv * 1e3 / math.sqrt(3.0)
        s_phase = s_base_va / 3.0
        z_base = v_base ** 2 / s_phase
        i_base = s_phase / v_base
        self.z = np.array([complex(br.r_ohm, br.x_ohm) / z_base for br in branches])
        self.abs_z = np.abs(self.z)
        self.i_max = np.array([br.i_max_a / i_base if br.rated else np.inf for br in branches])
        self.buses = np.array(topo.buses, dtype=int)
        self.vm_lo, self.vm_hi = case.vm_min_pu, case.vm_max_pu
        #: The slack voltage ``v0`` (per unit); the model's ``1`` becomes ``v0``.
        self.v0 = float(case.slack_vm_pu)
        self.scale = 1.0e6 / s_base_va                 # MW -> per unit
        self.max_sweeps = int(max_sweeps)
        self._branch_flow_q = case.branch_flow_q
        self.last_certificate: Optional[SweepCertificate] = None
        if certificate == "interval":
            self._interval_data(case, float(s_base_va))

    def _interval_data(self, case, s_base_va: float) -> None:
        """The model's parameters as intervals: the unit conversions are rounded too."""
        from OpenRC.powerflow.interval import ComplexInterval, Interval, TreeSums

        topo = case.topology
        branches = list(topo.branches)
        three = Interval.point(3.0)
        v_base = Interval.point(case.v_nom_kv) * 1e3 / three.sqrt()
        s_phase = Interval.point(s_base_va) / three
        z_base = v_base.square() / s_phase
        i_base = s_phase / v_base
        self._tree = TreeSums(topo)
        self._iz = ComplexInterval(
            Interval.point(np.array([br.r_ohm for br in branches])) / z_base,
            Interval.point(np.array([br.x_ohm for br in branches])) / z_base)
        self._iabs_z_hi = self._iz.abs_hi()
        rated = np.array([br.rated for br in branches], dtype=bool)
        limit = (Interval.point(np.array([br.i_max_a for br in branches])) / i_base).lo
        self._ii_max_lo = np.where(rated, limit, np.inf)
        self._iscale = Interval.point(1.0e6) / Interval.point(s_base_va)

    def _interval_certificate(self, p: np.ndarray, q: np.ndarray, v_hat: np.ndarray,
                              radius: float) -> Optional["SweepCertificate"]:
        """Re-prove the certificate at ``(v_hat, radius)`` in interval arithmetic.

        Every quantity of the Banach argument is enclosed with outward rounding
        (:mod:`OpenRC.powerflow.interval`): the residual ``delta`` from above,
        the Lipschitz bound ``L(r)`` from above, the condition
        ``delta <= r (1 - L)`` with ``r (1 - L)`` from below, the error bound
        from above, and every margin from below.  ``None`` when the proof does
        not go through.
        """
        from OpenRC.powerflow.interval import ComplexInterval, Interval

        tree, buses = self._tree, self.buses
        s = ComplexInterval(Interval.point(p) * self._iscale, Interval.point(q) * self._iscale)
        v = ComplexInterval.point(v_hat)
        v_lo, v_hi = v.abs_lo()[buses], v.abs_hi()[buses]
        floor = (Interval.point(v_lo) - radius).lo
        if not np.all(floor > 0.0):
            return None
        current = (s / v).conj()                                   # I_j = conj(s_j / V_j)
        # Four subtree sums in one pass: the branch currents (re, im), and the
        # weights of the Lipschitz bound, |s_j| / (|V^_j| - r)^2, and of the
        # current enclosure, |s_j| / ((|V^_j| - r) |V^_j|).  All are upper bounds.
        s_hi = Interval.point(s.abs_hi()[buses])
        lower = Interval.point(floor)
        weights = np.zeros((self.T.shape[1], 2))
        weights[buses, 0] = (s_hi / lower.square()).hi
        weights[buses, 1] = (s_hi / (lower * Interval.point(v_lo))).hi
        sums = tree.subtree(Interval(
            np.column_stack([current.re.lo, current.im.lo, weights]),
            np.column_stack([current.re.hi, current.im.hi, weights])))
        branch = ComplexInterval(sums[:, 0], sums[:, 1])
        drop = self._iz * branch
        # Three path sums in one pass: the voltage drops (re, im) and the
        # Lipschitz bound's |z| T(weight).
        summed = (sums[:, 2] * Interval.point(self._iabs_z_hi)).hi
        paths = tree.path(Interval(np.column_stack([drop.re.lo, drop.im.lo, summed]),
                                   np.column_stack([drop.re.hi, drop.im.hi, summed])))
        mapped = ComplexInterval(self.v0 - paths[:, 0], -paths[:, 1])  # F(v_hat)
        delta = float(np.max((mapped - v)[buses].abs_hi()))
        lipschitz = float(np.max(paths.hi[buses, 2]))
        room = (1.0 - Interval.point(lipschitz)).lo
        # A comparison with NaN (from an overflow inside the enclosure) is False,
        # so a NaN anywhere fails the proof.
        if not (room > 0.0 and delta <= (Interval.point(radius) * room).lo):
            return None
        error = float((Interval.point(delta) / room).hi)
        low = ((Interval.point(v_lo) - error) - self.vm_lo).lo
        high = ((self.vm_hi - Interval.point(v_hi)) - error).lo
        spread = (Interval.point(sums[:, 3].hi) * error).hi
        load = (Interval.point(branch.abs_hi()) + Interval.point(spread)).hi
        slack = (Interval.point(self._ii_max_lo) - Interval.point(load)).lo
        voltage_margin = float(np.min(np.concatenate([low, high])))   # NaN propagates
        current_margin = float(np.min(slack))
        if math.isnan(voltage_margin) or math.isnan(current_margin):
            return None
        return SweepCertificate(
            residual=delta, radius=radius, lipschitz=lipschitz, error_bound=error,
            voltage_margin=voltage_margin, current_margin=current_margin, rigorous=True)

    def _sweep(self, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """One backward/forward sweep: currents up the tree, voltage drops down."""
        return self.v0 - self.Tt @ (self.z * (self.T @ np.conj(s / v)))

    def _lipschitz(self, abs_s: np.ndarray, floor: np.ndarray) -> float:
        """Upper bound on the map's Lipschitz constant where every ``|V_j| >= floor_j``."""
        weight = np.zeros_like(abs_s)
        weight[self.buses] = abs_s[self.buses] / floor[self.buses] ** 2
        return float(np.max((self.Tt @ (self.abs_z * (self.T @ weight)))[self.buses]))

    def solve(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> EquationResult:
        """Certify the case's injections (MW / Mvar, as from ``case.injection_vector``).

        The case's line charging (``line_charging="constant"``) is applied here,
        exactly as the branch-flow oracles apply it before their models.
        """
        return self.solve_branch_flow(p_mw, self._branch_flow_q(q_mvar))

    def solve_branch_flow(self, p_mw: np.ndarray, q_mvar: np.ndarray) -> EquationResult:
        """Sweep to a fixed point, certify it, and check every limit against it.

        ``q_mvar`` is already the branch-flow reactive load
        (``case.branch_flow_q``); the oracles' candidate path calls this.
        """
        start = time.perf_counter()
        raw_p = np.asarray(p_mw)
        raw_q = np.asarray(q_mvar)
        if raw_p.dtype == bool or raw_q.dtype == bool:
            raise ValueError("p_mw and q_mvar must be numeric arrays, not boolean")
        p = np.asarray(raw_p, dtype=float)
        q = np.asarray(raw_q, dtype=float)
        expected = (self.T.shape[1],)
        if p.shape != expected or q.shape != expected:
            raise ValueError(
                f"p_mw and q_mvar must both have shape {expected}; "
                f"got {p.shape} and {q.shape}"
            )
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise ValueError("power injections must contain only finite values")
        self.last_certificate = None
        s = (p + 1j * q) * self.scale
        abs_s = np.abs(s)

        def result(accepted: bool, sweeps: int, message: str,
                   v: Optional[np.ndarray] = None) -> EquationResult:
            return EquationResult(self.name, v, accepted, accepted, 1, sweeps, None,
                                  time.perf_counter() - start, message)

        v = np.full(len(s), self.v0, dtype=complex)
        sweeps = 0
        for sweeps in range(1, self.max_sweeps + 1):
            new = self._sweep(s, v)
            step = float(np.max(np.abs(new - v)))
            v = new
            if not np.all(np.isfinite(v)):
                return result(False, sweeps, "the sweeps diverged")
            if step <= 1e-13:
                break
        # Certify around v: delta, then the smallest radius that works.
        delta = float(np.max(np.abs(self._sweep(s, v) - v))) * (1.0 + 1e-9) + 1e-15
        magnitude = np.abs(v)
        low = float(np.min(magnitude[self.buses]))
        radius, error, lip = None, None, None
        r = max(4.0 * delta, 1e-12)
        while r < 0.5 * low:
            lip = self._lipschitz(abs_s, magnitude - r) * (1.0 + 1e-9)
            if lip < 1.0 and delta <= r * (1.0 - lip):
                radius, error = r, delta / (1.0 - lip)
                break
            r *= 8.0
        if error is None:
            return result(False, sweeps, "no contraction around the sweep's fixed point")
        vm = magnitude[self.buses]
        current = np.abs(self.T @ np.conj(s / v))
        weight = np.zeros_like(abs_s)
        weight[self.buses] = abs_s[self.buses] / ((magnitude[self.buses] - radius)
                                                  * magnitude[self.buses])
        spread = error * (self.T @ weight)
        voltage_margin = float(min(np.min(vm - error - self.vm_lo),
                                   np.min(self.vm_hi - vm - error)))
        current_margin = float(np.min(self.i_max - current - spread))
        assert radius is not None and lip is not None
        cert = SweepCertificate(
            residual=delta, radius=radius, lipschitz=lip, error_bound=error,
            voltage_margin=voltage_margin, current_margin=current_margin,
        )
        if cert.limits_proven and self.certificate == "interval":
            proof = self._interval_certificate(p, q, v, radius)
            if proof is None or not proof.limits_proven:
                self.last_certificate = proof if proof is not None else cert
                return result(False, sweeps, "the interval-arithmetic proof did not "
                                             "confirm the floating-point certificate")
            self.last_certificate = proof
            return result(True, sweeps, f"proven feasible in interval arithmetic after "
                                        f"{sweeps} sweep(s), error bound {proof.error_bound:.1e}",
                          magnitude)
        self.last_certificate = cert
        if cert.limits_proven:
            return result(True, sweeps, f"proven feasible after {sweeps} sweep(s), "
                                        f"error bound {error:.1e}", magnitude)
        return result(False, sweeps, "the certified solution breaks a limit, or is too "
                                     "close to one")


STRUCTURED_SOLVERS.add(
    "sweep", RadialSweep,
    doc="closed-form backward/forward sweeps with a contraction error bound (exact model)")


__all__ = ["RadialSweep", "SweepCertificate"]
