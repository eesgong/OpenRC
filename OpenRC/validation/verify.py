"""Verifying that a region is inner: a certificate, or Monte Carlo.

:func:`verify_inner_region` checks whether every injection in a polytope admits
a feasible operating point, offering two modes with different guarantees:

``mode="surface"``
    The certificate (:func:`~OpenRC.validation.certify.certify_inner_region`):
    one spatial branch-and-bound per operating limit over the whole region,
    which *proves* the limits hold everywhere; existence of a solution proven
    at every point by the contraction condition where it holds; and points
    spread uniformly over the region's surface, checked point by point.  The
    vertices are not re-sampled - the construction already evaluated them.
    Needs Gurobi.  Only where the contraction condition fails does existence
    between the sampled points rest on the continuity/no-fold assumption
    (scope ``"conditional_inner"``).

``mode="monte-carlo"``
    Many uniform random points in a large cube around the region.  Points that
    fall inside the region are checked; points outside are only counted.  No
    global solver is needed for the primary sweep; the result provides a
    statistical error-rate estimate and confidence interval rather than a
    global certificate.

In both modes every checked point goes through the same escalation
(:class:`~OpenRC.validation.pointcheck.PointVerifier`): AC power flow, then a
global exact solve (Gurobi), then the sub-models of the region's conservative
method (MOSEK) and the SOC relaxation, and finally - if nothing settled it - the
point is recorded as *computationally infeasible* together with its most severe
limit overrun.
"""

from __future__ import annotations

import time
from contextlib import closing
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from OpenRC.geometry.polytope import Polytope
from OpenRC.validation.pointcheck import PointOutcome, PointVerification

if TYPE_CHECKING:
    from OpenRC.validation.certify import InnerCertificate

#: The two verification modes.
VERIFY_MODES = ("surface", "monte-carlo")


@dataclass
class VerificationReport:
    """Result of :func:`verify_inner_region`."""

    mode: str
    case: str
    axes: List[str]
    records: List[PointVerification]      # one per checked (in-region) point
    n_drawn: int
    cube: Optional[Tuple[np.ndarray, np.ndarray]] = None
    certificate: Optional["InnerCertificate"] = None  # surface mode only
    runtime: float = 0.0
    notes: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- counts
    def _with(self, outcome: PointOutcome) -> List[PointVerification]:
        return [r for r in self.records if r.outcome is outcome]

    @property
    def n_checked(self) -> int:
        return len(self.records)

    @property
    def infeasible(self) -> List[PointVerification]:
        """Points proven infeasible - real errors of the region."""
        return self._with(PointOutcome.INFEASIBLE)

    @property
    def computationally_infeasible(self) -> List[PointVerification]:
        """Points no stage could settle, recorded with their worst overrun."""
        return self._with(PointOutcome.COMPUTATIONALLY_INFEASIBLE)

    @property
    def n_decided(self) -> int:
        """Checked points with a definite feasible/infeasible classification."""
        return self.n_checked - len(self.computationally_infeasible)

    @property
    def error_rate(self) -> float:
        """Share of *decided* points proven infeasible.

        Computational failures are reported separately.  Including them in the
        denominator would make the apparent error rate improve when a solver
        fails more often.
        """
        return len(self.infeasible) / self.n_decided if self.n_decided else float("nan")

    @property
    def error_rate_ci(self) -> tuple:
        from OpenRC.validation.quality import wilson_interval

        return wilson_interval(len(self.infeasible), self.n_decided)

    @property
    def n_model_disagreements(self) -> int:
        """Feasible points whose first-stage verdict was a decided *infeasible*."""
        return sum(r.model_disagreement for r in self.records)

    @property
    def n_singular_points(self) -> int:
        """Singular points: :attr:`n_model_disagreements` under its reported name."""
        return self.n_model_disagreements

    @property
    def singular_points(self) -> np.ndarray:
        """Checked points whose first stage decided infeasible but that are feasible."""
        rows = [r.point for r in self.records if r.model_disagreement]
        return np.array(rows) if rows else np.empty((0, len(self.axes)))

    @property
    def computational_rate(self) -> float:
        """Share of checked points recorded as computationally infeasible."""
        n = len(self.computationally_infeasible)
        return n / self.n_checked if self.n_checked else float("nan")

    @property
    def status(self) -> str:
        """Certificate status, or ``passed`` / ``refuted`` / ``undecided``.

        Only ``mode="surface"`` can return ``"certified"``; a Monte Carlo sweep
        with zero violations and zero unresolved points returns ``"passed"``.
        A surface certificate may return ``limit-violated`` when an unsafe
        solution was found without proving its injection infeasible.
        """
        if self.certificate is not None:
            return self.certificate.status
        if self.infeasible:
            return "refuted"
        if self.computationally_infeasible or not self.records:
            return "undecided"
        return "passed"

    # ---------------------------------------------------------------- output
    def summary(self) -> str:
        if self.certificate is not None:
            head = self.certificate.summary()
        else:
            assert self.cube is not None, "a Monte Carlo report always records its cube"
            lo, hi = self.cube
            head = (f"Monte Carlo verification for {self.case} over "
                    f"{', '.join(self.axes)}: {self.status.upper()}\n"
                    f"  cube               [{np.round(lo, 3).tolist()}, "
                    f"{np.round(hi, 3).tolist()}]\n"
                    f"  points drawn       {self.n_drawn}, inside the region "
                    f"{self.n_checked}")
            if self.n_checked:
                lo_ci, hi_ci = self.error_rate_ci
                head += (f"\n  error rate         {100 * self.error_rate:.3f} % "
                         f"(95% CI {100 * lo_ci:.3f}-{100 * hi_ci:.3f} %)")
        lines = [head]
        counts = {o: len(self._with(o)) for o in PointOutcome}
        lines.append("  decided by         " + ", ".join(
            f"{sum(r.stage == s for r in self.records)} {s}"
            for s in dict.fromkeys(r.stage for r in self.records)))
        lines.append("  outcomes           " + ", ".join(
            f"{counts[o]} {o.value}" for o in PointOutcome if counts[o]))
        disagreements = self.n_model_disagreements
        if disagreements:
            lines.append(f"  SINGULAR           {disagreements} point(s) whose first solve violated "
                         "a limit have a safe alternative in a compatible model: counted "
                         "feasible (listed in singular_points)")
        if self.certificate is None:
            for rec in self.infeasible[:5]:
                lines.append(f"  INFEASIBLE {np.round(rec.point, 4).tolist()}"
                             + (f": {rec.worst.describe()}" if rec.worst else ""))
            for rec in self.computationally_infeasible[:10]:
                lines.append(f"  computationally infeasible {np.round(rec.point, 4).tolist()}"
                             + (f": {rec.worst.describe()}" if rec.worst else ""))
        lines.extend(f"  note: {n}" for n in self.notes)
        lines.append(f"  total runtime      {self.runtime:.1f} s")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "case": self.case, "axes": self.axes,
            "status": self.status, "n_drawn": self.n_drawn, "n_checked": self.n_checked,
            "error_rate": self.error_rate, "computational_rate": self.computational_rate,
            "n_model_disagreements": self.n_model_disagreements,
            "n_singular_points": self.n_singular_points,
            "singular_points": self.singular_points.tolist(),
            "cube": None if self.cube is None else [c.tolist() for c in self.cube],
            "records": [r.as_dict() for r in self.records],
            "certificate": None if self.certificate is None else self.certificate.as_dict(),
            "runtime_s": self.runtime, "notes": list(self.notes),
        }


def monte_carlo_cube(region, scale: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """A hypercube centred on the region's bounding box, ``scale`` times its widest side."""
    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    if (isinstance(scale, (bool, np.bool_))
            or not isinstance(scale, (int, float, Real))
            or not np.isfinite(float(scale)) or float(scale) <= 0.0):
        raise ValueError("scale must be finite and strictly positive")
    lower, upper = region.bounding_box()
    centre = 0.5 * (lower + upper)
    half = 0.5 * float(scale) * float(np.max(upper - lower))
    return centre - half, centre + half


def verify_inner_region(
    region,
    case,
    axes=None,
    *,
    mode: str = "surface",
    n_points: Optional[int] = None,
    cube: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    cube_scale: float = 1.5,
    min_inside: Optional[int] = None,
    max_draws: Optional[int] = None,
    primary="acpf",
    method: Optional[str] = "auto",
    exact_time_limit: float = 60.0,
    n_workers=1,
    backend: str = "process",
    rng=0,
    **certify_kwargs,
) -> VerificationReport:
    """Check that ``region`` is inner, by certificate or Monte Carlo.

    Parameters
    ----------
    mode:
        ``"surface"`` (default) - global limit bounds plus surface points;
        ``"monte-carlo"`` - uniform points in a large cube.
    n_points:
        Surface points (default 32) or Monte Carlo draws (default 5000).  In
        Monte Carlo mode only the draws that land inside the region are checked.
    cube, cube_scale:
        Monte Carlo only.  An explicit ``(lower, upper)`` box, or a cube
        ``cube_scale`` times the region's widest extent, centred on it.
    min_inside, max_draws:
        Monte Carlo only.  A large cube around a slender region catches few
        points - in 4-D often well under 1 %.  With ``min_inside`` set, drawing
        continues in batches of ``n_points`` until at least that many points
        have landed inside, or ``max_draws`` (default ``100 * n_points``) is
        reached.  Every draw stays uniform in the cube.
    primary:
        First-stage AC oracle, ``"acpf"`` or ``"acopf"``.
    method:
        Conservative method whose sub-models form the theory stage.  ``"auto"``
        reads it from the region's metadata (Method 2 if unknown); ``None``
        skips the sub-models.
    exact_time_limit:
        Seconds per global exact solve (and per limit in the certificate).
    n_workers, backend:
        Monte Carlo only: parallelism of the first-stage AC sweep.
    certify_kwargs:
        Surface mode only: passed to
        :func:`~OpenRC.validation.certify.certify_inner_region`.
    """
    from OpenRC.validation.certify import certify_inner_region, default_point_verifier

    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    if mode not in VERIFY_MODES:
        raise ValueError(f"mode must be one of {VERIFY_MODES}, got {mode!r}")
    if mode == "monte-carlo" and certify_kwargs:
        raise TypeError(f"options {sorted(certify_kwargs)} apply to mode='surface' only")
    start = time.perf_counter()
    axes = axes if axes is not None else region.axes
    if axes is None:
        raise ValueError("the region carries no axes; pass them explicitly")
    region.validate_axes(axes)

    verifier_kwargs = dict(primary=primary, exact_time_limit=exact_time_limit,
                           target_model="branch-flow" if mode == "surface" else "primary")
    if method != "auto":
        verifier_kwargs["method"] = method
    with closing(default_point_verifier(region, case, **verifier_kwargs)) as verifier:
        labels = [a.label for a in axes]

        if isinstance(rng, (bool, np.bool_)):
            raise TypeError("rng must be an integer seed or numpy Generator, not bool")

        if mode == "surface":
            if n_points is not None and (isinstance(n_points, (bool, np.bool_))
                                         or not isinstance(n_points, (int, float, Real))
                                         or not np.isfinite(float(n_points))
                                         or int(float(n_points)) != float(n_points)
                                         or float(n_points) < 0):
                raise ValueError("n_points must be a non-negative integer in surface mode")
            if "point_verifier" in certify_kwargs:
                raise TypeError("verify_inner_region builds its own point verifier; call "
                                "certify_inner_region to supply one")
            # The certificate's own names for the two options this function
            # also sets are accepted too, as the docstring forwards them.
            surface = certify_kwargs.pop("surface_points",
                                         32 if n_points is None else int(n_points))
            time_limit = certify_kwargs.pop("time_limit", exact_time_limit)
            cert = certify_inner_region(
                region, case, axes, surface_points=surface,
                time_limit=time_limit, point_verifier=verifier, rng=rng, **certify_kwargs)
            return VerificationReport(
                mode=mode, case=case.name, axes=labels, records=list(cert.surface_checks),
                n_drawn=len(cert.surface_checks), certificate=cert,
                runtime=time.perf_counter() - start)

        # ---- Monte Carlo ------------------------------------------------------
        from OpenRC.validation.quality import _reference_sweep
        from OpenRC.validation.samplers import BoxSampler

        if backend not in ("process", "thread"):
            raise ValueError(f"backend must be 'process' or 'thread', got {backend!r}")
        if n_points is not None and (isinstance(n_points, (bool, np.bool_))
                                     or not isinstance(n_points, (int, float, Real))
                                     or not np.isfinite(float(n_points))
                                     or int(float(n_points)) != float(n_points)
                                     or float(n_points) <= 0):
            raise ValueError("n_points must be a positive integer in Monte Carlo mode")
        if min_inside is not None and (isinstance(min_inside, (bool, np.bool_))
                                       or not isinstance(min_inside, (int, float, Real))
                                       or not np.isfinite(float(min_inside))
                                       or int(float(min_inside)) != float(min_inside)
                                       or float(min_inside) < 0):
            raise ValueError("min_inside must be a non-negative integer")
        if max_draws is not None and (isinstance(max_draws, (bool, np.bool_))
                                      or not isinstance(max_draws, (int, float, Real))
                                      or not np.isfinite(float(max_draws))
                                      or int(float(max_draws)) != float(max_draws)
                                      or float(max_draws) <= 0):
            raise ValueError("max_draws must be a positive integer")
        if (isinstance(cube_scale, (bool, np.bool_))
                or not isinstance(cube_scale, (int, float, Real))
                or not np.isfinite(float(cube_scale)) or float(cube_scale) <= 0.0):
            raise ValueError("cube_scale must be finite and strictly positive")
        if cube is not None:
            if isinstance(cube, (str, bytes, bytearray)):
                raise ValueError("cube must be a (lower, upper) pair")
            try:
                lo_raw, hi_raw = cube
                lo_arr, hi_arr = np.asarray(lo_raw), np.asarray(hi_raw)
                if lo_arr.dtype == bool or hi_arr.dtype == bool:
                    raise ValueError
                lower, upper = lo_arr.astype(float, copy=False), hi_arr.astype(float, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("cube must be a (lower, upper) pair") from exc
            if lower.shape != (region.dim,) or upper.shape != (region.dim,):
                raise ValueError(
                    f"cube bounds must have shape ({region.dim},), "
                    f"got {lower.shape} and {upper.shape}"
                )
        else:
            lower, upper = monte_carlo_cube(region, cube_scale)
        batch = 5000 if n_points is None else int(n_points)
        cap = int(max_draws) if max_draws is not None else 100 * batch
        sampler = BoxSampler(lower=lower, upper=upper)
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        kept, n_draw = [], 0
        while True:
            draw_now = min(batch, cap - n_draw)
            if draw_now <= 0:
                break
            points = sampler.draw(draw_now, rng=generator)
            n_draw += draw_now
            kept.append(points[region.contains(points)])
            n_inside = sum(len(k) for k in kept)
            if min_inside is None or n_inside >= min_inside or n_draw >= cap:
                break
        inside = np.vstack(kept) if kept else np.empty((0, region.dim), dtype=float)
        report = VerificationReport(mode=mode, case=case.name, axes=labels, records=[],
                                    n_drawn=n_draw, cube=(lower, upper))
        if min_inside is not None and len(inside) < min_inside:
            report.notes.append(f"only {len(inside)} of {n_draw} draws landed inside the "
                                f"region (min_inside={min_inside}, max_draws={cap})")
        elif min_inside is None and len(inside) < 100:
            report.notes.append(f"only {len(inside)} of {n_draw} draws landed inside the "
                                "region, so the error-rate interval is wide; set min_inside "
                                "or reduce cube_scale")
        if len(inside):
            first = _reference_sweep(verifier.primary, axes, inside, n_workers=n_workers,
                                     backend=backend, progress_every=0)
            report.records = verifier.verify_many(axes, inside, primary_verdicts=first)
        else:
            report.notes.append("no draw landed inside the region; increase n_points or "
                                "shrink the cube")
        report.notes.append("Monte Carlo provides a statistical estimate; use "
                            "mode='surface' for global limit bounds and existence certification")
        report.runtime = time.perf_counter() - start
        return report
