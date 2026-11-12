"""Measuring a region: the feasible space it loses, the infeasible points it admits.

Two complementary metrics quantify region accuracy against a reference oracle:

**Error rate** (``1 - P_RD``)
    Fraction of decided in-region samples proven infeasible.  Zero for a
    conservative (inner) approximation.

**Loss rate** (``1 - P_R``)
    Fraction of proven-feasible samples outside the region, measuring the
    feasible space omitted by conservatism.

Unresolved points
-----------------
A point on which the AC power flow failed to converge has not been shown
infeasible, so it is escalated to a global solve of the exact equations (see
:mod:`OpenRC.validation.adjudicate`).  Points that survive even that are
reported as **unresolved** and left out of the error-rate denominator - counting
them either way would be an unsupported claim.  They are listed so they can be
checked by hand.

For independent, identically distributed draws, both rates come with Wilson
score intervals.  Scrambled Sobol points and scenarios sampled without
replacement are dependent, so binomial intervals are not reported for them.
Sample size directly governs interval width even when zero violations are
observed (e.g. 0/200 vs. 0/20,000).
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import List, Optional

import numpy as np

from OpenRC.geometry.polytope import Polytope
from OpenRC.powerflow.base import Verdict, _reference_evidence
from OpenRC.utils import LOGGER


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation: it stays inside ``[0, 1]`` and
    remains meaningful at zero successes, which is the case that matters when
    checking that a conservative region admits nothing infeasible.
    """
    if (isinstance(z, (bool, np.bool_))
            or not isinstance(z, (int, float, Real))
            or not math.isfinite(float(z)) or float(z) <= 0.0):
        raise ValueError("z must be a positive finite number")
    if (isinstance(trials, (bool, np.bool_)) or isinstance(successes, (bool, np.bool_))
            or not isinstance(trials, (int, float, Real))
            or not isinstance(successes, (int, float, Real))
            or not math.isfinite(float(trials)) or not math.isfinite(float(successes))
            or int(float(trials)) != float(trials)
            or int(float(successes)) != float(successes)):
        raise ValueError("successes and trials must be integers")
    z = float(z)
    trials = int(float(trials))
    successes = int(float(successes))
    if trials <= 0:
        if successes != 0:
            raise ValueError("successes must be 0 when trials <= 0")
        return (0.0, 1.0)
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between 0 and trials")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (0.0 if successes == 0 else max(0.0, centre - spread),
            1.0 if successes == trials else min(1.0, centre + spread))


@dataclass
class RegionQuality:
    """Monte Carlo assessment of one region against a reference oracle."""

    n_samples: int
    n_in_region: int              # N_R
    n_feasible: int               # N_D (proven feasible, anywhere in the sample)
    n_both: int                   # N_RD (proven feasible and in the region)
    n_unresolved_in_region: int = 0
    n_unresolved_total: int = 0
    violating_points: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    unresolved_points: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    missed_points: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    adjudications: List = field(default_factory=list)
    seconds: float = 0.0
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------ denominators
    @property
    def n_decided_in_region(self) -> int:
        """In-region samples with a definite verdict - the error-rate denominator."""
        return self.n_in_region - self.n_unresolved_in_region

    @property
    def n_confirmed_violations(self) -> int:
        """In-region samples proven infeasible."""
        return self.n_decided_in_region - self.n_both

    # ------------------------------------------------------------------ rates
    @property
    def p_rd(self) -> float:
        """Share of *decided* in-region samples that are feasible."""
        return self.n_both / self.n_decided_in_region if self.n_decided_in_region else float("nan")

    @property
    def p_r(self) -> float:
        """Share of proven-feasible samples captured by the region.

        The numerator must be ``n_both``.  Using all in-region samples here can
        exceed one when a candidate region contains infeasible points.
        """
        return self.n_both / self.n_feasible if self.n_feasible else float("nan")

    @property
    def error_rate(self) -> float:
        """``1 - P_RD``: proven-infeasible points inside the region.  Zero is the goal."""
        return 1.0 - self.p_rd if self.n_decided_in_region else float("nan")

    @property
    def loss_measured(self) -> bool:
        """Whether the sampler drew points outside the region (see ``Sampler.measures_loss``).

        A sampler that draws only points of the region proves every feasible
        sample captured, so its loss rate would be zero by construction.
        """
        return bool(self.info.get("loss_measured", True))

    @property
    def loss_rate(self) -> float:
        """``1 - P_R``: feasible space the region gave up (NaN if not measured)."""
        return 1.0 - self.p_r if self.n_feasible and self.loss_measured else float("nan")

    @property
    def error_rate_ci(self) -> tuple:
        if not self.info.get("iid_sampling", True):
            return (float("nan"), float("nan"))
        return wilson_interval(self.n_confirmed_violations, self.n_decided_in_region)

    @property
    def loss_rate_ci(self) -> tuple:
        if not self.loss_measured or not self.info.get("iid_sampling", True):
            return (float("nan"), float("nan"))
        low, high = wilson_interval(self.n_both, self.n_feasible)
        return (max(0.0, 1.0 - high), min(1.0, 1.0 - low))

    @property
    def is_conservative(self) -> bool:
        """True when no in-region sample was *proven* infeasible.

        This finite-sample compatibility flag is not an inner-region proof.
        Unresolved points weaken the evidence, so :meth:`summary` reports them.
        """
        return self.n_decided_in_region > 0 and self.n_confirmed_violations == 0

    # --------------------------------------------------------------- reporting
    def summary(self) -> str:
        err_lo, err_hi = self.error_rate_ci
        loss_lo, loss_hi = self.loss_rate_ci
        iid = self.info.get("iid_sampling", True)
        error_interval = (f"[{100 * err_lo:.3f}, {100 * err_hi:.3f}] 95% CI" if iid
                          else "(binomial CI unavailable: dependent or unspecified draws)")
        loss_interval = (f"[{100 * loss_lo:.3f}, {100 * loss_hi:.3f}] 95% CI" if iid
                         else "(binomial CI unavailable: dependent or unspecified draws)")
        if self.n_confirmed_violations > 0:
            verdict = "NOT conservative - proven-infeasible points lie inside the region"
        elif self.n_decided_in_region == 0:
            verdict = "undecided - no in-region sample has a definite verdict"
        elif self.n_unresolved_in_region > 0:
            verdict = (f"no violation among decided samples, but {self.n_unresolved_in_region} "
                       "in-region point(s) could not be decided")
        else:
            verdict = "no violation observed (sampling only; not an inner-region certificate)"

        lines = [
            f"region quality over {self.n_samples} samples ({self.seconds:.1f} s)",
            f"    in region          N_R  = {self.n_in_region}"
            f"  (decided {self.n_decided_in_region}, unresolved {self.n_unresolved_in_region})",
            f"    proven feasible    N_D  = {self.n_feasible}",
            f"    both               N_RD = {self.n_both}",
            f"    error rate (1-P_RD) = {100 * self.error_rate:.3f}%  "
            f"{error_interval}"
            f"   ({self.n_confirmed_violations} confirmed violation(s))",
            (f"    loss  rate (1-P_R)  = {100 * self.loss_rate:.3f}%  "
             f"{loss_interval}" if self.loss_measured else
             "    loss  rate (1-P_R)  = not measured: the sampler draws only points of "
             "the region"),
            f"    verdict: {verdict}",
        ]
        if (self.info.get("reference_outer_relaxation")
                and not self.info.get("adjudicated", False)):
            lines.append(
                "    NOTE: the reference is an outer relaxation; its feasible verdicts are "
                "not proofs, so the error rate is a lower bound."
            )
        if self.info.get("reference_model_only"):
            lines.append(
                "    NOTE: adjudication is disabled; these are model-relative rates, "
                "not AC/exact feasibility certificates."
            )
        if self.n_unresolved_total:
            lines.append(
                f"    NOTE: {self.n_unresolved_total} sample(s) could not be decided by the "
                "reference or by the exact global solve."
            )
            lines.append(
                "          They are excluded from the error rate and listed in "
                "`unresolved_points`; confirm them manually before relying on this region."
            )
        disagreements = self.n_model_disagreements
        overturned = sum(
            1 for a in self.adjudications
            if getattr(a, "final", None) is Verdict.FEASIBLE
            and not getattr(a, "model_disagreement", False)
            and getattr(a, "primary", None) is not Verdict.FEASIBLE
        )
        if overturned:
            lines.append(
                f"    NOTE: {overturned} disputed point(s) were overturned on escalation "
                "(the primary reference failed on points the exact model could solve)."
            )
        if disagreements:
            lines.append(
                f"    SINGULAR: {disagreements} point(s) whose reference solve violated a limit "
                "have a safe alternative in a compatible model: counted feasible, "
                "listed in `singular_points`."
            )
            lines.append(
                "          Inspect model data, alternative power-flow solutions and "
                "solver tolerances; a different decision does not establish its cause. "
                "adjudicate=False counts the "
                "reference's own decisions."
            )
        return "\n".join(lines)

    @property
    def n_model_disagreements(self) -> int:
        """Points the reference decided infeasible but an escalation proved feasible.

        Counted as feasible, so they are absent from the error rate; see
        :attr:`~OpenRC.validation.adjudicate.Adjudication.model_disagreement`.
        """
        return sum(1 for a in self.adjudications if getattr(a, "model_disagreement", False))

    @property
    def n_singular_points(self) -> int:
        """Singular points: :attr:`n_model_disagreements` under its reported name."""
        return self.n_model_disagreements

    @property
    def singular_points(self) -> np.ndarray:
        """Points AC decided infeasible that have an exact operating point within limits.

        They count as feasible; this lists them for inspection (one row each).
        """
        rows = [np.asarray(a.point, dtype=float) for a in self.adjudications
                if getattr(a, "model_disagreement", False)]
        if rows:
            return np.array(rows, dtype=float)
        vp = np.asarray(self.violating_points)
        dim = vp.shape[1] if vp.ndim == 2 and vp.shape[1] else (
            int(np.asarray(self.adjudications[0].point).size) if self.adjudications else 0
        )
        return np.empty((0, dim), dtype=float)

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_in_region": self.n_in_region,
            "n_decided_in_region": self.n_decided_in_region,
            "n_feasible": self.n_feasible,
            "n_both": self.n_both,
            "n_confirmed_violations": self.n_confirmed_violations,
            "n_unresolved_in_region": self.n_unresolved_in_region,
            "n_unresolved_total": self.n_unresolved_total,
            "p_rd": self.p_rd,
            "p_r": self.p_r,
            "error_rate": self.error_rate,
            "loss_rate": self.loss_rate,
            "error_rate_ci": list(self.error_rate_ci),
            "loss_rate_ci": list(self.loss_rate_ci),
            "is_conservative": self.is_conservative,
            "n_model_disagreements": self.n_model_disagreements,
            "n_singular_points": self.n_singular_points,
            "singular_points": self.singular_points.tolist(),
            "seconds": self.seconds,
            **self.info,
        }


def evaluate_region(
    region: Polytope,
    case,
    axes=None,
    *,
    reference="acpf",
    n_samples: int = 1000,
    sampler="uniform-box",
    sampler_kwargs: Optional[dict] = None,
    rng: int = 0,
    n_workers=1,
    backend: str = "process",
    adjudicate: bool = True,
    escalate_to=("exact",),
    keep_points: bool = True,
    progress_every: int = 0,
) -> RegionQuality:
    """Estimate a region's error and loss rate against a reference oracle.

    Parameters
    ----------
    region:
        The polytope to assess.
    case, axes:
        Needed to map region coordinates back to nodal injections.
    reference:
        Oracle name or instance treated as the first-line ground truth.
        ``"acpf"`` (full AC power flow) by default.
    n_samples:
        Number of test points; the cost is one reference power flow each.
    sampler:
        Sampler name or instance; see :mod:`OpenRC.validation.samplers`.
    n_workers:
        Workers for the reference sweep.  ``"auto"`` measures the per-call cost
        and only starts a pool when it will repay itself; an explicit integer is
        always honoured.  Escalation of disputed samples runs sequentially.
    adjudicate:
        Escalate every answer that does not prove a reference verdict before
        counting it, inside or outside the region. Outer-model feasibility,
        local reference rejection and unknown component relationships require
        more evidence. A weak reference can therefore escalate every sample.
        ``False`` measures the chosen model's own decisions instead.
    escalate_to:
        Escalation chain, strongest last.  ``("exact",)`` is a global solve of
        the exact branch-flow equations.

    Returns
    -------
    RegionQuality
    """
    from OpenRC.powerflow.base import get_oracle
    from OpenRC.validation.adjudicate import Adjudicator
    from OpenRC.validation.samplers import get_sampler

    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    axes = axes if axes is not None else region.axes
    if axes is None:
        raise ValueError("the region carries no axes; pass them explicitly")
    if (isinstance(progress_every, (bool, np.bool_))
            or not isinstance(progress_every, (int, float, Real))
            or not np.isfinite(float(progress_every))
            or int(float(progress_every)) != float(progress_every) or float(progress_every) < 0):
        raise ValueError("progress_every must be a non-negative integer")
    if not isinstance(adjudicate, (bool, np.bool_)):
        raise ValueError("adjudicate must be a boolean")
    if not isinstance(keep_points, (bool, np.bool_)):
        raise ValueError("keep_points must be a boolean")
    if backend not in ("process", "thread"):
        raise ValueError(f"backend must be 'process' or 'thread', got {backend!r}")
    if sampler_kwargs is not None and not isinstance(sampler_kwargs, dict):
        raise TypeError("sampler_kwargs must be a dict")
    if not isinstance(sampler, str):
        if not callable(getattr(sampler, "draw", None)):
            raise TypeError("sampler must be a sampler name or an object with a .draw() method")
        if sampler_kwargs:
            raise TypeError("sampler_kwargs cannot be combined with a pre-built sampler instance")
    if not isinstance(reference, str) and not callable(getattr(reference, "verdict_at", None)):
        raise TypeError("reference must be an oracle name or an oracle")
    if adjudicate and isinstance(escalate_to, (str, bytes, bytearray)):
        raise TypeError("escalate_to must be a sequence of oracle names or instances, not a string")
    progress_every = int(progress_every)
    region.validate_axes(axes)
    # Materialise the chain once: the adjudicator and the returned provenance
    # both read it, and a one-shot iterable could be consumed only once.
    escalate_to = tuple(escalate_to) if adjudicate else ()
    start = time.perf_counter()
    # Named references are owned by this call; external instances stay open.
    with contextlib.ExitStack() as resources:
        oracle = get_oracle(reference, case) if isinstance(reference, str) else reference
        from OpenRC.cases.base import validate_case_configuration

        validate_case_configuration(case, oracle)
        if isinstance(reference, str):
            resources.callback(oracle.close)
        outer_reference = getattr(oracle, "inner_approximation", None) is False
        model_only = (not adjudicate and not getattr(oracle, "supports_adjudication", True))
        if outer_reference and not adjudicate:
            LOGGER.warning("the reference %r is an outer relaxation: its 'feasible' does not "
                           "prove feasibility, so the error rate is a lower bound; use 'acpf' "
                           "or 'exact' to measure it", getattr(oracle, "name", reference))
        if model_only:
            LOGGER.warning("the reference %r has no reference-feasibility inclusion "
                           "guarantee; adjudicate=False reports model-relative rates only",
                           getattr(oracle, "name", reference))
        sampler_obj = (get_sampler(sampler, **(sampler_kwargs or {}))
                       if isinstance(sampler, str) else sampler)

        # Instantiate the adjudicator before running the reference sweep.
        adjudicator = (resources.enter_context(
            Adjudicator(case, primary=oracle, escalate_to=escalate_to))
            if adjudicate else None)
        points = sampler_obj.draw(n_samples, region=region, rng=rng)
        in_region = region.contains(points)

        # --- first pass: the cheap reference -----------------------------------
        # One reference solve per point, all independent, so this is the part worth
        # parallelising: it dominates the runtime of a validation sweep.
        verdicts = _reference_sweep(oracle, axes, points, n_workers=n_workers,
                                    backend=backend, progress_every=progress_every)

        # --- second pass: adjudicate the disputed ones -------------------------
        adjudications = []
        if adjudicator is not None:
            disputed = [
                i for i, v in enumerate(verdicts)
                # Filtering only in-region negatives biases the loss estimate;
                # raw weak-model acceptance can also hide an error in-region.
                if _reference_evidence(oracle, v) is Verdict.INCONCLUSIVE
            ]
            if disputed:
                LOGGER.info("validation: escalating %d disputed point(s)", len(disputed))
            for i in disputed:
                record = adjudicator.judge(axes, points[i], primary_verdict=verdicts[i])
                adjudications.append(record)
                verdicts[i] = record.final

        verdict_array = np.array([v.value for v in verdicts])
        feasible = verdict_array == Verdict.FEASIBLE.value
        infeasible = verdict_array == Verdict.INFEASIBLE.value
        unresolved = verdict_array == Verdict.INCONCLUSIVE.value

        n_in = int(in_region.sum())
        empty = np.empty((0, region.dim))
        result = RegionQuality(
            n_samples=len(points),
            n_in_region=n_in,
            n_feasible=int(feasible.sum()),
            n_both=int((in_region & feasible).sum()),
            n_unresolved_in_region=int((in_region & unresolved).sum()),
            n_unresolved_total=int(unresolved.sum()),
            violating_points=points[in_region & infeasible] if keep_points else empty,
            unresolved_points=points[unresolved] if keep_points else empty,
            missed_points=points[~in_region & feasible] if keep_points else empty,
            adjudications=adjudications,
            seconds=time.perf_counter() - start,
            info={
                "reference": getattr(oracle, "name", str(reference)),
                "sampler": sampler_obj.describe(),
                "loss_measured": bool(getattr(sampler_obj, "measures_loss", True)),
                "iid_sampling": bool(getattr(sampler_obj, "iid_sampling", False)),
                "adjudicated": bool(adjudicate),
                "escalate_to": list(escalate_to) if adjudicate else [],
                "region_volume": region.volume,
                "region_n_vertices": region.n_vertices,
                "reference_outer_relaxation": outer_reference,
                "reference_model_only": model_only,
            },
        )
    result.seconds = time.perf_counter() - start  # include owned-model cleanup
    return result


def diagnose_violations(case, axes, points: np.ndarray, *, max_points: int = 50) -> dict:
    """Summarise worst-case bus-voltage and branch-loading violations on ``points``.

    Runs AC power flow on up to ``max_points`` violating samples and reports
    peak branch loading, voltage extrema, and violation counts by limit type.
    """
    import pandapower as pp

    from OpenRC.powerflow.acpf import AcPowerFlowOracle
    from OpenRC.powerflow.base import get_oracle

    if axes is None or isinstance(axes, (str, bytes, bytearray)):
        raise TypeError("axes must be a sequence of Axis objects, not None or a string")
    if (isinstance(max_points, (bool, np.bool_))
            or not isinstance(max_points, (int, float, Real))
            or not np.isfinite(float(max_points))
            or int(float(max_points)) != float(max_points) or float(max_points) <= 0):
        raise ValueError("max_points must be a positive integer")
    max_points = int(max_points)
    raw_arr = np.asarray(points)
    if raw_arr.dtype == bool:
        raise ValueError("points must be a numeric array, not boolean")
    if raw_arr.ndim > 2:
        raise ValueError(f"points must be 1-D or 2-D, got {raw_arr.ndim}-D")
    raw = np.asarray(raw_arr, dtype=float)
    if raw.size == 0:
        if raw.ndim == 2 and raw.shape[1] not in (0, len(axes)):
            raise ValueError(f"points have dimension {raw.shape[1]}, expected {len(axes)}")
        return {"n_points": 0}
    pts = np.atleast_2d(raw)
    if pts.ndim != 2 or pts.shape[1] != len(axes):
        raise ValueError(f"points have dimension {pts.shape[1]}, expected {len(axes)}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite values")
    points = pts[:max_points]

    oracle = get_oracle("acpf", case)
    if not isinstance(oracle, AcPowerFlowOracle):  # narrows the private AC interface
        raise TypeError("the 'acpf' registry entry must build AcPowerFlowOracle")
    with contextlib.closing(oracle):
        oracle._prepare_net()
        max_loading = 0.0
        max_vm = -np.inf
        min_vm = np.inf
        n_diverged = 0
        binding_branch = None
        binding_branch_name = None
        for point in points:
            p, q = case.injection_vector(axes, point)
            oracle._apply(p, q)
            try:
                pp.runpp(oracle._net, init="flat", algorithm="nr",
                         max_iteration=oracle.max_iteration, numba=oracle.numba)
            except Exception:
                n_diverged += 1
                continue
            vm = oracle._net.res_bus.vm_pu.to_numpy(dtype=float)
            if "in_service" in oracle._net.bus.columns:
                vm = vm[oracle._net.bus.in_service.to_numpy(dtype=bool)]
            for table, result_table in (("line", "res_line"),
                                        ("trafo", "res_trafo"),
                                        ("trafo3w", "res_trafo3w")):
                elements = getattr(oracle._net, table, None)
                results = getattr(oracle._net, result_table, None)
                if elements is None or results is None or len(elements) == 0:
                    continue
                loading = results.loading_percent.to_numpy(dtype=float)
                if "in_service" in elements.columns:
                    active = elements.in_service.to_numpy(dtype=bool)
                    loading = np.where(active, loading, np.nan)
                if np.isfinite(loading).any() and float(np.nanmax(loading)) > max_loading:
                    max_loading = float(np.nanmax(loading))
                    position = int(np.nanargmax(loading))
                    binding_branch = (table, int(elements.index[position]))
                    binding_branch_name = str(elements.name.iloc[position])
            if np.isfinite(vm).any():
                max_vm = max(max_vm, float(np.nanmax(vm)))
                min_vm = min(min_vm, float(np.nanmin(vm)))
    return {
        "max_loading_percent": max_loading,
        "max_vm_pu": max_vm,
        "min_vm_pu": min_vm,
        "n_diverged": n_diverged,
        "binding_branch": binding_branch,
        "binding_branch_name": binding_branch_name,
        "n_points": int(len(points)),
    }


def _reference_sweep(oracle, axes, points, *, n_workers, backend, progress_every):
    """First-pass verdicts for every sample point, optionally across workers.

    Every query is charged to ``oracle``, whoever answers it.
    """
    from OpenRC.parallel import OracleSpec, FeasibilityPool, resolve_workers
    from OpenRC.powerflow.base import OracleStats

    if backend not in ("process", "thread"):
        raise ValueError(f"backend must be 'process' or 'thread', got {backend!r}")
    if len(points) == 0:
        return []
    sequential = n_workers != "auto" and resolve_workers(n_workers) == 1
    spec = None if sequential else OracleSpec.from_oracle(oracle, axes)
    sequential = sequential or spec is None

    if sequential:
        out = []
        for i, point in enumerate(points):
            out.append(oracle.verdict_at(axes, point))
            if progress_every and (i + 1) % progress_every == 0:
                LOGGER.info("validation: %d/%d points checked", i + 1, len(points))
        return out

    assert spec is not None           # ``sequential`` covers the no-spec case
    remote = OracleStats()

    def charged(verdicts):
        # Worker copies answered these; their queries belong to ``oracle``.
        with getattr(oracle, "_lock", None) or contextlib.nullcontext():
            oracle.stats.n_calls += remote.n_calls
            oracle.stats.n_feasible += remote.n_feasible
            oracle.stats.seconds += remote.seconds
        return [Verdict(v) for v in verdicts]

    workers = resolve_workers(n_workers)
    if n_workers == "auto":
        import time as _time

        from OpenRC.parallel import plan_workers

        # Warm the model up before timing, then calibrate on a short batch.
        head = [oracle.verdict_at(axes, points[0])]
        timed = points[1:min(12, len(points))]
        if len(timed):
            start = _time.perf_counter()
            head += [oracle.verdict_at(axes, p) for p in timed]
            per_item = (_time.perf_counter() - start) / len(timed)
        else:
            per_item = 0.0
        rest = points[len(head):]
        workers = plan_workers(len(rest), per_item, "auto", key=(spec.case, spec.oracle))
        if workers == 1:
            return head + [oracle.verdict_at(axes, p) for p in rest]
        with FeasibilityPool(spec, n_workers=workers, backend=backend) as pool:
            tail = pool.verdicts(rest, stats=remote)
        return head + charged(tail)

    with FeasibilityPool(spec, n_workers=workers, backend=backend) as pool:
        return charged(pool.verdicts(points, stats=remote))
