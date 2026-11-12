"""Inner (conservative) approximations of the exact feasible region.

Linearised power-flow models can overestimate the feasible injection set.  For
operational publication or downstream optimisation, an inner approximation is
required: every point in the region must satisfy the exact branch-flow model,
trading hypervolume for feasibility guarantees.

Two methods do that.  Both build two convex regions and intersect them; they
differ in the pair of models and in whether a shift is needed afterwards.

Method 1 - ``intersect-shift`` (bracket, then shift)
-----------------------------------------------------
Models: :mod:`lindistflow <OpenRC.powerflow.distflow>` (Model-I) and
``lindistflow-vdrop`` (Model-II).

1. :math:`\\bar{C} = C^{I} \\cap C^{II}`.  The exact voltage carries
   :math:`+(r^2+x^2) l` with :math:`l \\in [0, l^{max}]`; Model-I evaluates that
   at the low end and Model-II at the high end, so the two **bracket** the true
   voltage and their intersection is contained in the region of the model that
   keeps the exact voltage-drop equation.
2. Neither model charges the loss to the *power balance*.  The exact model
   adds :math:`r\\,l` to every branch's active flow **and** :math:`x\\,l` to
   its reactive flow, all at the same time, with :math:`l \\in [0, l^{max}]`.
   So for every POC :math:`j` the region is shifted along :math:`P_j` by
   :math:`r\\,l^{max}` and along :math:`Q_j` by :math:`x\\,l^{max}`, and a
   point is kept only if it remains inside :math:`\\bar{C}` across the full
   product box of joint loss shifts.  Along a direction that is an axis of the region, a shift is a
   translation, and the whole box of on-axis shifts is absorbed exactly by a
   Pontryagin difference (each facet tightened by its worst box corner).
   Along a direction that is not an axis - a region over active power only
   has no :math:`Q_j` axis - the shifted copy is the *slice* of the feasible
   set with the base injection at :math:`j` raised by the shift, computed by
   rebuilding both component regions at that base; every corner of the
   off-axis box contributes its slice.  The loss bound is accumulated over
   the complete path from each POC to the slack (``shift_scope="path"``);
   ``feeding_branch`` charges only the feeding branch and can miss a
   downstream POC's load on an upstream branch whose loss no retained
   coordinate carries.  Either scope is an exact argument only when every
   lossy branch feeds a region bus (:func:`method1_covers_losses`): the loss
   of a branch feeding any other bus is absorbed by no shift.
3. Optionally clip back to each POC's own rating.

Method 2 - ``tight-intersect`` (margin on each side, no shift)
---------------------------------------------------------------
Models: ``tight-voltage`` (S1) and ``tight-flow`` (S2).

The published construction combines two complementary conservative models:

* **S1** drops balance losses, evaluates the positive voltage-loss term at
  ``l_max``, and tightens each thermal limit by losses along the path to slack;
* **S2** charges maximum losses to the power balance, drops the voltage-loss
  term, and retains the nameplate thermal limits.

Their regions are intersected **directly**.  No shift is needed: the two
models bound the loss terms from both sides, at the zero-loss and the
maximum-loss corner.  With the default path scope, a mixed combination of
branch losses (one branch near its maximum, another lossless) is covered by
neither, so the result requires post-hoc certification rather than holding
unconditionally.  This avoids post-intersection shift slices, though it is
more conservative and S1's tightening can empty the region on feeders with
high-reactance, high-rating branches (see
:meth:`DistFlowOracle.binding_tightening`).

With ``oracle_kwargs={"tighten_scope": "subtree"}`` S1 tightens each branch by
the largest losses of its own subtree, ``D``, and checks the lossless flow at
the lowest permissible voltage, ``|S^LL| <= (I_max - |D| / V_min) V_min``.
That is an argument (:func:`method2_covers_losses`).  For every loss vector
``l`` in the box ``[0, l_max]`` the exact flows and voltages are affine in
``l``; with ``r, x >= 0`` the voltages lie between S2's (a lower bound) and
S1's (an upper bound), so within limits, and every flow within ``|D|`` of the
lossless one, so ``|S(l)|^2 / v(l) <= l_max`` (with the slack held at or
above ``v_min``).  The map
``l -> |S(l)|^2 / v(l)`` therefore sends the box into itself, and Brouwer's
theorem gives a fixed point: an exact branch-flow solution within every
limit, at every point of both (convex) model regions.

Which to use is an engineering trade-off;
:func:`OpenRC.validation.evaluate_region` measures what each costs.
"""

from __future__ import annotations

import math
import time
from contextlib import ExitStack
from typing import Optional

import numpy as np

from OpenRC.geometry.bisection import InfeasibleOriginError
from OpenRC.geometry.ops import clip_to_box, intersect
from OpenRC.geometry.polytope import Polytope
from OpenRC.powerflow.base import FeasibilityOracle, Verdict, get_oracle
from OpenRC.region.base import BUILDERS, RegionBuilder, RegionResult
from OpenRC.utils import LOGGER

#: Above this many off-axis shift directions Method 1's 2**k slice builds get
#: expensive; the builder warns.
_SLICE_WARNING = 6

#: ``method -> (oracle A, oracle B, needs a shift step)``.
CONSERVATIVE_METHODS = {
    "intersect-shift": ("lindistflow", "lindistflow-vdrop", True),
    "tight-intersect": ("tight-voltage", "tight-flow", False),
}

_METHOD_ALIASES = {
    "method1": "intersect-shift",
    "method2": "tight-intersect",
}


@BUILDERS.register(
    "conservative",
    doc="Inner approximation of the exact region: intersect-shift, or tight-intersect.",
)
class ConservativeBuilder(RegionBuilder):
    """Build a candidate inner region from one of the published constructions.

    ``tight-intersect`` intersects the paper's two conservative models
    directly.  ``intersect-shift`` brackets the voltage-loss term with two
    linear models and then absorbs the balance losses with the box of
    :math:`P` and :math:`Q` shifts described in the module docstring.  Both
    passed the inner-region certificate
    (:func:`~OpenRC.validation.certify.certify_inner_region`) on every 4-D
    9-bus configuration tried.  Pass ``certify=True`` to run the same global
    limit bounds and existence proof on your own network.

    Parameters
    ----------
    method:
        ``"intersect-shift"`` (Model-I with Model-II, then shift) or
        ``"tight-intersect"`` (S1 with S2, intersected directly).
    inner_builder:
        The builder used for each underlying region computation.  Defaults to a
        fresh :class:`~OpenRC.region.expansive.ExpansiveBuilder`.
    shift_scope:
        Only for ``intersect-shift``.  Which branches' losses the shift absorbs:
        ``"path"`` (default; complete upstream path) or ``"feeding_branch"``
        (direct-feeder only).  See :meth:`_axis_shifts`.
    intersect_method:
        ``"halfspace"`` for the exact intersection, ``"vertex_filter"`` for
        the inner approximation used to reproduce published numbers (see
        :func:`OpenRC.geometry.ops.intersect`).
    poc_limits:
        Optional ``(lower, upper)`` arrays applied as a final box clip - the
        step that re-imposes each POC's own rating.
    oracle_kwargs:
        Passed to both component models, for example
        ``{"tighten_scope": "subtree"}`` for the Method-2 S1 scope that makes
        the result ``theorem_backed`` (see the module docstring).
    certify:
        Run global limit bounds plus the existence proof (or, where the
        contraction condition fails, the surface checks) on the final region,
        contracting it towards the base point until it is certified
        (:func:`~OpenRC.validation.certify.shrink_to_certified`).  The
        certificate is stored in ``builder_info``.  Needs Gurobi.
    certify_kwargs:
        Options for the certificate (``time_limit``, ``surface_points``, ...).
    """

    name = "conservative"

    def __init__(
        self,
        *,
        method: str = "intersect-shift",
        inner_builder=None,
        shift_scope: str = "path",
        intersect_method: str = "halfspace",
        poc_limits: Optional[tuple] = None,
        oracle_kwargs: Optional[dict] = None,
        certify: bool = False,
        certify_kwargs: Optional[dict] = None,
    ) -> None:
        method = _METHOD_ALIASES.get(method, method)
        if method not in CONSERVATIVE_METHODS:
            raise ValueError(
                f"method must be one of {tuple(CONSERVATIVE_METHODS)}, got {method!r}"
            )
        if shift_scope not in ("feeding_branch", "path"):
            raise ValueError(
                f"shift_scope must be 'feeding_branch' or 'path', got {shift_scope!r}"
            )
        if intersect_method not in ("halfspace", "vertex_filter"):
            raise ValueError("intersect_method must be 'halfspace' or 'vertex_filter', "
                             f"got {intersect_method!r}")
        if not isinstance(certify, (bool, np.bool_)):
            raise ValueError("certify must be a boolean")
        # Validate optional bounds and keyword dictionaries up front.
        if poc_limits is not None:
            try:
                raw_lower, raw_upper = poc_limits
                arr_l, arr_u = np.asarray(raw_lower), np.asarray(raw_upper)
                if arr_l.dtype == bool or arr_u.dtype == bool or arr_l.ndim != 1 or arr_u.ndim != 1:
                    raise ValueError
                lower = arr_l.astype(float, copy=False)
                upper = arr_u.astype(float, copy=False)
            except (TypeError, ValueError):
                raise ValueError("poc_limits must be a (lower, upper) pair of 1D arrays") from None
            if (lower.shape != upper.shape or not np.all(np.isfinite(lower))
                    or not np.all(np.isfinite(upper)) or np.any(lower >= upper)):
                raise ValueError("poc_limits must be finite (lower, upper) arrays of one "
                                 "shape with every lower bound below its upper bound")
            poc_limits = (lower, upper)
        if certify_kwargs is not None and not isinstance(certify_kwargs, dict):
            raise TypeError("certify_kwargs must be a dict")
        if oracle_kwargs is not None and not isinstance(oracle_kwargs, dict):
            raise TypeError("oracle_kwargs must be a dict")
        certify_kwargs = dict(certify_kwargs or {})
        if certify_kwargs:
            import inspect

            from OpenRC.validation.certify import certify_inner_region, shrink_to_certified

            accepted = ((set(inspect.signature(shrink_to_certified).parameters)
                         | set(inspect.signature(certify_inner_region).parameters))
                        - {"region", "case", "axes", "certify_kwargs"})
            unknown = sorted(set(certify_kwargs) - accepted)
            if unknown:
                raise TypeError(f"unknown certify_kwargs {unknown}; accepted: "
                                f"{', '.join(sorted(accepted))}")
        self.method = method
        self.shift_scope = shift_scope
        self.intersect_method = intersect_method
        self.poc_limits = poc_limits
        self.oracle_kwargs = dict(oracle_kwargs or {})
        self.certify = bool(certify)
        self.certify_kwargs = certify_kwargs
        if inner_builder is None:
            from OpenRC.region.expansive import ExpansiveBuilder

            inner_builder = ExpansiveBuilder()
        elif not callable(getattr(inner_builder, "build", None)):
            raise TypeError("inner_builder must have a callable build(case, axes, ...) method")
        self.inner_builder = inner_builder

    # ------------------------------------------------------------------ build
    def build(self, case, axes, oracle=None, **kwargs) -> RegionResult:
        """Compute the conservative region.

        ``oracle`` is accepted for interface compatibility but ignored: both
        methods need a specific *pair* of models, which the method selects.
        Surplus keyword arguments go to the inner builder's ``build``.
        """
        from OpenRC.cases.base import validate_case_configuration

        validate_case_configuration(case)
        if oracle is not None:
            LOGGER.info(
                "conservative builder ignores the supplied oracle; method %r selects "
                "its own pair of models", self.method,
            )
        start = time.perf_counter()
        if self.poc_limits is not None and self.poc_limits[0].shape != (len(axes),):
            raise ValueError(f"poc_limits have {self.poc_limits[0].size} entries for "
                             f"{len(axes)} axes")
        name_a, name_b, needs_shift = CONSERVATIVE_METHODS[self.method]

        # ---- 1. the two component regions ---------------------------------
        with ExitStack() as resources:
            oracle_a = get_oracle(name_a, case, **self.oracle_kwargs)
            resources.callback(oracle_a.close)
            oracle_b = get_oracle(name_b, case, **self.oracle_kwargs)
            resources.callback(oracle_b.close)
            runs = {}
            for label, model_oracle in ((name_a, oracle_a), (name_b, oracle_b)):
                LOGGER.info("conservative[%s]: building the %s region", self.method, label)
                try:
                    runs[label] = self.inner_builder.build(case, axes, model_oracle, **kwargs)
                except InfeasibleOriginError as exc:
                    # Rejection of one seed origin is not an infeasibility proof
                    # for the whole convex component; another centre may work.
                    raise InfeasibleComponentOriginError(
                        f"the selected seed origin is rejected by {label!r}, so the "
                        f"{self.method!r} construction could not start that component. "
                        "This is not a proof that its feasible set is empty. Supply a "
                        "known feasible common centre through the inner builder's "
                        "ray_kwargs={'origin': centre}, or inspect the component limits."
                    ) from exc
                if runs[label].polytope.volume <= 0.0:
                    raise EmptyComponentRegionError(_empty_message(label, self.method))
            run_a, run_b = runs[name_a], runs[name_b]

            # ---- 2. intersect --------------------------------------------------
            combined = intersect(run_a.polytope, run_b.polytope, method=self.intersect_method)
            combined.axes = axes
            volume_after_intersection = combined.volume

            # ---- 3. shift, for Method 1 only -----------------------------------
            region = combined
            shifts = {}
            off_axis = []
            if needs_shift:
                shifts = self._axis_shifts(case, axes)
                off_axis = self._off_axis_shifts(case, axes)
                region = self._apply_shifts(combined, shifts, off_axis, case, axes,
                                            oracle_a, oracle_b, kwargs)
                region.axes = axes
            volume_after_shift = region.volume

            # ---- 4. re-impose the POC ratings ----------------------------------
            region = self._apply_poc_limits(region)
            # Record the construction method in polytope metadata so verification
            # can select matching sub-models automatically.
            region.metadata.update(conservative_method=self.method)
            if needs_shift:
                region.metadata.update(shift_scope=self.shift_scope)
            else:
                region.metadata.update(tighten_scope=getattr(oracle_a, "tighten_scope", None))
            volume_before_certificate = region.volume

            # ---- 5. optionally, prove it (contracting until provable) ---------
            certificate, contraction = None, None
            if self.certify:
                from OpenRC.validation.certify import shrink_to_certified

                LOGGER.info("conservative[%s]: running the inner-region certificate", self.method)
                region, certificate, contraction = shrink_to_certified(
                    region, case, axes, **self.certify_kwargs)
                region.axes = axes

            merged_stats = {
                "n_calls": oracle_a.stats.n_calls + oracle_b.stats.n_calls,
                "seconds": round(oracle_a.stats.seconds + oracle_b.stats.seconds, 4),
                name_a: oracle_a.stats.as_dict() | oracle_a.describe(),
                name_b: oracle_b.stats.as_dict() | oracle_b.describe(),
            }
        # Method 1's shift argument is a theorem only when every loss the
        # linear models drop can be represented as a shift of a region
        # coordinate: every lossy branch must feed a region bus.  A loss on a
        # branch feeding any other bus raises the exact flows like a load at
        # that bus, which no shift of a region coordinate reproduces.  Method
        # 2's path-scope models bound the all-zero and the all-maximum loss
        # corners, and a mixed combination of branch losses can escape both;
        # with the subtree scope S1 bounds every combination (module docstring).
        theorem_backed = (method1_covers_losses(case, axes) if needs_shift
                          else method2_covers_losses(case, oracle_a))
        if certificate is not None:
            guarantee = (("certified inner: global branch-flow limit bounds, existence "
                          "proven at every point" if certificate.scope == "inner" else
                          "conditionally certified inner with global branch-flow limit bounds")
                         + (f" after contraction by {contraction:.4f}"
                            if contraction is not None and contraction < 1.0 else ""))
        elif theorem_backed and needs_shift:
            guarantee = (
                "inner approximation: every lossy branch feeds a region bus, so the box "
                "of P and Q loss shifts covers every loss the linear models drop "
                "(subject to tolerances)"
            )
        elif theorem_backed:
            guarantee = (
                "inner approximation: S1's subtree tightening, checked at v_min, bounds "
                "every combination of branch losses up to l_max, so an exact branch-flow "
                "solution within every limit exists at every point (subject to tolerances)"
            )
        elif self.method == "tight-intersect":
            guarantee = (
                "paper-derived candidate inner approximation: a mixed combination of "
                "branch losses can escape both models - certify before relying on it"
            )
        elif self.shift_scope == "path":
            guarantee = (
                "paper-derived candidate inner approximation: the loss of a branch feeding "
                "a bus that is not a region axis is absorbed by no shift - certify before "
                "relying on it"
            )
        else:
            guarantee = (
                "box shift with direct-feeder scope: losses on upstream branches that "
                "no retained POC feeds can be missed - certify before relying on it"
            )
        info = {
            "builder": self.name,
            "conservative_method": self.method,
            "model_a": name_a,
            "model_b": name_b,
            "shift_applied": bool(needs_shift),
            "intersect_method": self.intersect_method,
            "inner_builder": self.inner_builder.name,
            "volume_model_a": run_a.polytope.volume,
            "volume_model_b": run_b.polytope.volume,
            "volume_after_intersection": volume_after_intersection,
            "volume_after_shift": volume_after_shift,
            "volume_final": region.volume,
            "volume_before_certificate": volume_before_certificate,
            "guarantee": guarantee,
            "theorem_backed": theorem_backed,
            "case": getattr(case, "name", "?"),
            "axes": list(axes.labels),
        }
        if certificate is not None:
            info.update(
                certified=certificate.certified,
                certified_contraction=contraction,
                certificate=certificate.as_dict(),
            )
        if not needs_shift:
            info.update(tighten_scope=getattr(oracle_a, "tighten_scope", None))
        if needs_shift:
            info.update(
                shift_scope=self.shift_scope,
                axis_shifts_mw={axes[k].label: shifts[k] for k in range(len(axes))},
                off_axis_shift_values={f"b{bus}{qty}": amount
                                       for bus, qty, amount in off_axis},
            )
        return RegionResult(
            polytope=region,
            trace=run_a.trace,
            oracle_stats=merged_stats,
            builder_info=info,
            seconds=time.perf_counter() - start,
            stopped_because=run_a.stopped_because,
        )

    # ---------------------------------------------------------------- helpers
    def _apply_shifts(self, combined: Polytope, shifts: dict, off_axis: list, case, axes,
                      oracle_a, oracle_b, build_kwargs) -> Polytope:
        """Absorb the loss the linear models drop: every combination of shifts.

        Everything here is exact on inequalities.  For each corner ``delta`` of
        the box of off-axis shifts, the slice :math:`\\bar{C}(\\delta)`
        (:meth:`_slice`; :math:`\\bar{C}` itself at the zero corner) is eroded
        by the box ``[0, s]`` of on-axis shifts: each facet ``a . y <= b``
        becomes ``a . y <= b - sum_k max(0, a_k s_k)``, its worst box corner.
        The rows of all corners are stacked and the vertices enumerated once.

        An empty intersection indicates that the worst-case loss shifts exceed
        the bracketed region's width: the bound assumes simultaneous full-rating
        losses on every branch in scope, which on a deep feeder can exceed the
        hosting range.
        """
        import itertools

        from OpenRC.geometry.ops import EmptyIntersectionError, from_halfspaces

        s = np.array([shifts[k] for k in range(len(axes))])
        if len(off_axis) > _SLICE_WARNING:
            LOGGER.warning("intersect-shift with %d off-axis shift directions builds %d "
                           "slices", len(off_axis), 2 ** len(off_axis))
        blocks = []
        for corner in itertools.product((False, True), repeat=len(off_axis)):
            dp, dq = np.zeros(case.n_bus), np.zeros(case.n_bus)
            for used, (bus, quantity, amount) in zip(corner, off_axis):
                if used:
                    (dp if quantity == "p" else dq)[bus] += amount
            body = (combined if not any(corner) else
                    self._slice(case, axes, oracle_a, oracle_b, dp, dq, build_kwargs,
                                combined=combined))
            blocks.append(_tighten(*body.facets, s))
        A = np.vstack([blk[0] for blk in blocks])
        b = np.concatenate([blk[1] for blk in blocks])
        try:
            return from_halfspaces(A, b, axes=axes, what="loss shift")
        except EmptyIntersectionError as exc:
            lower, upper = combined.bounding_box()
            extent = upper - lower
            too_wide = [f"axis {k}: shift {s[k]:.3g} vs extent {extent[k]:.3g}"
                        for k in range(len(s)) if s[k] >= extent[k]]
            detail = "; ".join(too_wide) or "the combined shifts leave no overlap"
            hint = (" The 'path' scope charges every branch between the POC and the "
                    "substation its full-rating loss; on a deep feeder that is very "
                    "conservative. shift_scope='feeding_branch' charges only the "
                    "feeding branch, without the upstream-loss guarantee. "
                    "(method='tight-intersect' charges the "
                    "same per-branch bound and is usually no better on such a feeder.)"
                    if self.shift_scope == "path" else "")
            raise EmptyComponentRegionError(
                f"the loss shifts of the intersect-shift construction are wider than "
                f"the region itself ({detail}), so nothing survives the "
                f"intersection.{hint}"
            ) from exc

    def _off_axis_shifts(self, case, axes) -> list:
        """Shifts along directions that are *not* axes of the region.

        For every POC bus ``j`` of the region Method 1 shifts along both
        :math:`P_j` (by ``r * l_max``) and :math:`Q_j` (by ``x * l_max``).  The
        axes cover one or both; this returns ``(bus, quantity, amount)`` for
        each direction they do not cover, with the same ``shift_scope`` and the
        same three-phase MW / Mvar scaling as :meth:`_axis_shifts`.
        """
        topo = case.topology
        present = {(axis.bus, axis.quantity) for axis in axes}
        out = []
        for bus in dict.fromkeys(axis.bus for axis in axes):
            branches = self._shift_branches(topo, bus)
            for quantity in ("p", "q"):
                if (bus, quantity) in present:
                    continue
                per_phase_w = sum((br.r_ohm if quantity == "p" else br.x_ohm) * br.l_max
                                  for br in branches)
                amount = per_phase_w * 3.0 / 1e6
                if amount > 0.0:
                    out.append((bus, quantity, amount))
        return out

    def _slice(self, case, axes, oracle_a, oracle_b, dp, dq, build_kwargs,
               combined: Optional[Polytope] = None) -> Polytope:
        """:math:`\\bar{C}` rebuilt with the base injection raised by ``(dp, dq)``.

        The inner builder grows a region around the base point.  A loss shift
        can make the base point itself infeasible - raising the reactive
        injection at the end of a long feeder pushes its voltage under the
        limit - while the slice still has feasible points elsewhere (for
        instance with some local generation).  The growth centre is then moved
        to a point of :math:`\\bar{C}` that both shifted models accept
        (:meth:`_slice_centre`), the region is grown around it, and translated
        back.  Only when no such point exists is the slice treated as empty.
        """
        moved = ", ".join(f"{q.upper()}{bus}+{d[bus]:.4g}"
                          for q, d in (("p", dp), ("q", dq)) for bus in np.flatnonzero(d))
        inner_origin = _builder_ray_origin(self.inner_builder)
        centre = (np.zeros(len(axes)) if inner_origin is None
                  else np.asarray(inner_origin, dtype=float))
        shifted = [OffsetOracle(model, dp_mw=dp, dq_mvar=dq) for model in (oracle_a, oracle_b)]
        if not all(o.feasible_at(axes, centre) for o in shifted):
            centre = self._slice_centre(shifted, axes, combined)
            if centre is None:
                raise SliceCentreNotFoundError(
                    f"with the loss shift {moved} (MW / Mvar) applied off-axis, the "
                    "deterministic centre search did not locate a point accepted by "
                    "both linear models. This is not a proof that the slice is empty; "
                    "supply a less conservative shift/model or inspect the slice "
                    "feasibility problem directly."
                )
            LOGGER.info("conservative[%s]: slice at %s grown around %s (the base point "
                        "is infeasible there)", self.method, moved,
                        np.round(centre, 4).tolist())
        p_c, q_c = case.injection_vector(axes, centre)
        p_0, q_0 = case.injection_vector(axes)
        slice_builder = _without_ray_origin(self.inner_builder)
        parts = []
        for model in (oracle_a, oracle_b):
            LOGGER.info("conservative[%s]: %s slice at base %s", self.method,
                        model.name, moved)
            around = OffsetOracle(model, dp_mw=dp + (p_c - p_0), dq_mvar=dq + (q_c - q_0))
            try:
                part = slice_builder.build(case, axes, around, **build_kwargs)
            except InfeasibleOriginError as exc:       # guarded above
                raise EmptyComponentRegionError(
                    f"the slice at {moved} could not be grown around "
                    f"{np.round(centre, 4).tolist()}") from exc
            parts.append(part.polytope.translate(centre))
        return intersect(parts[0], parts[1], method=self.intersect_method)

    @staticmethod
    def _slice_centre(shifted_oracles, axes, combined: Optional[Polytope]):
        """A point of the bracketed region accepted by every shifted model.

        Tries the Chebyshev centre of :math:`\\bar{C}`, every vertex, and points
        on the segments between them, returning the first point accepted by all
        shifted models.  Because this finite deterministic search does not prove
        infeasibility, ``None`` does not imply that the slice is empty.
        """
        if combined is None:
            return None
        from OpenRC.geometry.ops import chebyshev_center

        centre = chebyshev_center(*combined.facets)
        if centre is None:
            return None
        vertices = list(combined.hull_vertices)
        candidates = [centre, *vertices] + [centre + t * (v - centre)
                                            for t in (0.25, 0.5, 0.75)
                                            for v in vertices]
        for point in candidates:
            if all(o.feasible_at(axes, point) for o in shifted_oracles):
                return point
        return None

    def _shift_branches(self, topo, bus) -> list:
        """The branches whose loss the shift at ``bus`` absorbs (``shift_scope``)."""
        if self.shift_scope == "path":
            return topo.path_to_root(bus)
        branch = topo.branch_to_child.get(bus)
        return [branch] if branch is not None else []

    def _axis_shifts(self, case, axes) -> dict:
        """Loss-absorbing translation per axis, in MW / Mvar (three-phase).

        An active axis is shifted by ``r * l_max`` and a reactive one by
        ``x * l_max``.  Both are per-phase watts, so they are scaled by ``3e-6``
        to reach the region's three-phase MW units.

        ``shift_scope`` chooses which branches contribute:

        ``"path"`` (default)
            every branch from the point of connection up to the slack.  The loss
            a linearised model drops does not appear only on the last branch - it
            accumulates along the whole path to the substation, and in practice
            the binding thermal constraint is often an upstream branch rather
            than the feeding one.  This shifts at least as far, so its region
            lies inside the ``"feeding_branch"`` one.
        ``"feeding_branch"``
            only the branch that feeds the point of connection.  Exact when every
            bus is a POC; with only selected POC coordinates it can miss losses on
            an upstream branch.
        """
        topo = case.topology
        shifts = {}
        for k, axis in enumerate(axes):
            branches = self._shift_branches(topo, axis.bus)
            per_phase_w = sum(
                (br.r_ohm if axis.quantity == "p" else br.x_ohm) * br.l_max
                for br in branches
            )
            shifts[k] = per_phase_w * 3.0 / 1e6
        return shifts

    def _apply_poc_limits(self, region: Polytope) -> Polytope:
        if self.poc_limits is None:
            return region
        lower, upper = self.poc_limits
        return clip_to_box(region, lower, upper)


def _empty_message(model: str, method: str) -> str:
    """Explain an empty component region and point at the two usual causes."""
    return (
        f"the {model!r} region is empty, so the {method!r} construction has nothing to "
        "intersect. Two things make a conservative model infeasible even at the base "
        "operating point:\n"
        "  1. the thermal tightening consumed a branch's whole rating - check "
        "DistFlowOracle.binding_tightening(), which reports the tightened/nameplate "
        "ratio per branch;\n"
        "  2. the loss charged to the power balance is large enough to push a bus "
        "voltage outside its envelope. A branch with both a high reactance and a high "
        "current rating (a transformer modelled as a line, say) is the usual source, "
        "because the charge scales with the *square* of the rating.\n"
        "If the network's ratings were scaled up for another study, that scaling is "
        "very likely the cause."
    )


def _builder_ray_origin(builder) -> Optional[np.ndarray]:
    """Configured ray ``origin`` on ``builder`` (or its wrapped ``inner_builder``)."""
    cur = builder
    while cur is not None:
        ray_kw = getattr(cur, "ray_kwargs", None)
        if isinstance(ray_kw, dict) and ray_kw.get("origin") is not None:
            return np.asarray(ray_kw["origin"], dtype=float)
        cur = getattr(cur, "inner_builder", None)
    return None


def _without_ray_origin(builder):
    """Shallow copy of ``builder`` with ``ray_kwargs['origin']`` stripped.

    :meth:`ConservativeBuilder._slice` shifts the oracle itself so ``0`` is the
    slice centre; leaving ``origin`` on the inner builder would shift twice.
    """
    import copy

    out = builder
    inner = getattr(builder, "inner_builder", None)
    if inner is not None:
        stripped_inner = _without_ray_origin(inner)
        if stripped_inner is not inner:
            out = copy.copy(out)
            out.inner_builder = stripped_inner
    ray_kw = getattr(out, "ray_kwargs", None)
    if isinstance(ray_kw, dict) and "origin" in ray_kw:
        if out is builder:
            out = copy.copy(out)
        out.ray_kwargs = {k: v for k, v in ray_kw.items() if k != "origin"}
    refiner = getattr(out, "refiner", None)
    ref_kw = getattr(refiner, "ray_kwargs", None)
    if refiner is not None and isinstance(ref_kw, dict) and "origin" in ref_kw:
        if out is builder:
            out = copy.copy(out)
        ref_copy = copy.copy(refiner)
        ref_copy.ray_kwargs = {k: v for k, v in ref_kw.items() if k != "origin"}
        out.refiner = ref_copy
    return out


def method1_covers_losses(case, axes) -> bool:
    """Whether Method 1's loss shifts provably absorb every loss the models drop.

    The exact flows equal the lossless flows at the injection raised by each
    branch's loss ``(r l, x l)`` at the branch's child bus.  Only a raise at a
    region bus is a shift of the region, so the construction's argument holds
    when every branch with a non-zero impedance is rated and feeds a bus that
    is an axis of the region (both scopes charge at least the feeding branch's
    full-rating loss there).
    """
    branches = case.topology.branches
    if any(br.r_ohm < 0.0 or br.x_ohm < 0.0 for br in branches):
        return False        # a negative loss (series compensation) lies in no shift box
    if any(
        not br.rated or not math.isfinite(br.l_max)
        for br in branches
        if br.r_ohm > 0.0 or br.x_ohm > 0.0
    ):
        return False
    axis_buses = {axis.bus for axis in axes}
    return all(br.child in axis_buses for br in branches
               if br.r_ohm > 0.0 or br.x_ohm > 0.0)


def method2_covers_losses(case, s1_model) -> bool:
    """Whether Method 2 with this S1 model provably bounds every branch-loss mix.

    True when S1 tightens each branch by its subtree's largest losses and
    checks the lossless flow at ``v_min`` (``tighten_scope="subtree"``, see
    :attr:`DistFlowOracle.thermal_cap_at_v_min`), every branch is rated with
    non-negative resistance and reactance, and the slack is held at or above
    ``v_min``.  The voltage bracket between S1 and S2, on which the argument in
    the module docstring rests, needs ``r, x >= 0``; the current bound needs
    every parent voltage, the slack's included, at or above ``v_min``.
    """
    if not getattr(s1_model, "thermal_cap_at_v_min", False):
        return False
    if case.topology.unrated_branches:
        return False
    if float(case.slack_vm_pu) < float(case.vm_min_pu):
        return False
    return not any(br.r_ohm < 0.0 or br.x_ohm < 0.0 for br in case.topology.branches)


def _tighten(A: np.ndarray, b: np.ndarray, shift: np.ndarray) -> tuple:
    """``{A y <= b}`` eroded by the box ``[0, shift]`` (Pontryagin difference).

    A copy shifted by ``-t`` is ``{A y <= b - A t}``: the same rows with other
    offsets.  Requiring every corner ``t`` of the box tightens each row by its
    worst corner, ``sum_k max(0, a_k * shift_k)``.
    """
    return A, b - np.maximum(A * shift[None, :], 0.0).sum(axis=1)


class OffsetOracle(FeasibilityOracle):
    """An oracle evaluated at a base injection raised by a fixed offset.

    ``OffsetOracle(model, dq_mvar=d)`` answers, for every injection ``(p, q)``,
    what ``model`` answers for ``(p, q + d)``.  A region built with it is the
    slice of ``model``'s feasible set at the shifted base - the shift of
    Method 1 along a direction that is not an axis of the region.
    """

    # A shifted query concerns another injection.  Even an exact base oracle
    # cannot certify the original unshifted point through this wrapper.
    supports_adjudication = False

    def __init__(self, base: FeasibilityOracle, *, dp_mw=None, dq_mvar=None) -> None:
        if getattr(base, "case", None) is None or not hasattr(base, "name"):
            raise TypeError(f"base must be a FeasibilityOracle, got {type(base).__name__}")
        super().__init__(base.case, name=f"{base.name}+offset")
        n = base.case.n_bus
        self.base = base
        if dp_mw is not None and np.asarray(dp_mw).dtype == bool:
            raise ValueError("offsets must be numeric, not boolean")
        if dq_mvar is not None and np.asarray(dq_mvar).dtype == bool:
            raise ValueError("offsets must be numeric, not boolean")
        self.dp = np.zeros(n) if dp_mw is None else np.asarray(dp_mw, dtype=float)
        self.dq = np.zeros(n) if dq_mvar is None else np.asarray(dq_mvar, dtype=float)
        if self.dp.shape != (n,) or self.dq.shape != (n,):
            raise ValueError(f"offsets must have shape ({n},)")
        if not (np.all(np.isfinite(self.dp)) and np.all(np.isfinite(self.dq))):
            raise ValueError("offsets must be finite")
        self.inner_approximation = base.inner_approximation
        self.supports_margin = base.supports_margin
        self.prefers_margin_search = base.prefers_margin_search

    def margin_available(self) -> bool:
        return self.base.margin_available()

    def reset_warm_state(self) -> None:
        self.base.reset_warm_state()

    def _verdict(self, p_mw, q_mvar):
        return self.base.verdict(p_mw + self.dp, q_mvar + self.dq)

    def _verdict_margin(self, p_mw, q_mvar):
        return self.base.verdict_margin(p_mw + self.dp, q_mvar + self.dq)

    def _check(self, p_mw, q_mvar) -> bool:
        return self._verdict(p_mw, q_mvar) is Verdict.FEASIBLE


class EmptyComponentRegionError(RuntimeError):
    """A positive-volume component could not be built.

    Inspect subclasses and the exception cause: a failed start/search does not
    establish that the entire mathematical feasible set has empty interior.
    """


class InfeasibleComponentOriginError(EmptyComponentRegionError):
    """The selected component seed is rejected, without proving emptiness."""


class SliceCentreNotFoundError(EmptyComponentRegionError):
    """A shifted-slice centre search failed without proving the slice empty."""
