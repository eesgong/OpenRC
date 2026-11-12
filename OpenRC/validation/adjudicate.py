"""Multi-stage adjudication of inconclusive and disputed power-flow verdicts.

Validation counts how often a region admits a point the reference rejects.  That
count is only meaningful if "rejected" means *infeasible*.  In practice a
Newton-Raphson power flow fails on perfectly feasible points - a bad flat start,
an ill-conditioned Jacobian near a limit, a point close to the nose of the PV
curve.  Charging those to the region turns a solver artefact into a reported
modelling error.

The adjudicator escalates every disputed point instead of counting it:

1. the primary reference (usually AC power flow) answers;
2. if that answer does not prove a reference verdict, the point goes to a global
   solve of the exact branch-flow equations
   (:class:`~OpenRC.powerflow.exact.ExactDistFlowOracle`), which either
   produces an operating point satisfying every limit or proves that none
   exists;
3. whatever survives that - the solver timed out, or hit numerical trouble -
   is recorded as **unresolved**.

Unresolved points are reported separately and excluded from the error rate,
because neither feasibility nor infeasibility was established; callers can
inspect them via ``unresolved_points``.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from OpenRC.powerflow.base import (SolverUnavailableError, Verdict, _model_compatibility,
                                  _reference_evidence)
from OpenRC.powerflow.topology import AmbiguousSlackError, MeshedNetworkError


@dataclass
class Adjudication:
    """The full record of how one disputed point was decided."""

    point: np.ndarray
    primary: Verdict
    escalations: List[Tuple[str, Verdict]] = field(default_factory=list)
    final: Verdict = Verdict.INCONCLUSIVE
    note: str = ""
    #: Whether the primary's verdict is evidence about the network itself
    #: (its ``supports_adjudication``): an AC power flow's is, a linear
    #: approximation's or a DC power flow's is not.
    primary_decides: bool = True

    @property
    def overturned(self) -> bool:
        """Whether escalation changed the verdict, including to or from undecided.

        A changed verdict alone is not evidence of a recovered feasible point;
        reporting code must also inspect the final and primary verdicts.
        """
        return self.primary is not self.final

    @property
    def model_disagreement(self) -> bool:
        """The primary *decided* infeasible, yet an escalation proved feasibility.

        Unlike an overturned failure (no answer from the primary), this is two
        models answering differently: the AC power flow converged outside a
        limit where the branch-flow model has a solution within every limit.
        The physical data and limits have been checked for compatibility.
        Numerical tolerances or an alternative nonlinear power-flow solution
        can explain the disagreement.  The point counts as feasible, and the
        discrepancy deserves inspection; a different physical model cannot
        overturn the primary's reference verdict.
        """
        return (self.primary_decides and self.primary is Verdict.INFEASIBLE
                and self.final is Verdict.FEASIBLE)

    @property
    def singular(self) -> bool:
        """A *singular point*: the same condition as :attr:`model_disagreement`.

        AC power flow converged outside a limit, yet the exact model has an
        operating point within every limit.  The point counts as feasible and
        is reported separately (``singular_points``) for inspection.
        """
        return self.model_disagreement

    def as_dict(self) -> dict:
        return {
            "point": self.point.tolist(),
            "primary": self.primary.value,
            "escalations": [(name, v.value) for name, v in self.escalations],
            "final": self.final.value,
            "overturned": self.overturned,
            "model_disagreement": self.model_disagreement,
            "singular": self.singular,
            "note": self.note,
        }


class Adjudicator:
    """Escalates disputed points through progressively stronger oracles.

    Use as a context manager, or call :meth:`close`.  Oracles constructed from
    registry names are owned here; caller-supplied instances are never closed.

    Parameters
    ----------
    case:
        The network.
    primary:
        Oracle name or instance used for the first, cheap answer.
    escalate_to:
        Oracle names tried, in order, whenever the primary answer lacks
        reference evidence. The default is the global exact solve; ``("socp", "exact")``
        tries the cheap conic certificate first.  Approximation direction is
        respected: SOCP feasibility alone is not treated as exact feasibility,
        while SOCP infeasibility is a valid certificate.

    The reference is the primary's underlying physical family.  Linear
    construction and DC screening primaries do not decide it themselves;
    matched stronger models may do so.  Native AC, branch flow and OPF are
    compared only when their physical data and constraints are equivalent.
    Different models are recorded as unavailable evidence, not as verdicts
    about this reference.  To measure an approximation's own set, use region
    evaluation with ``adjudicate=False``.
    """

    def __init__(self, case, *, primary="acpf",
                 escalate_to: Sequence[str] = ("exact",)) -> None:
        from OpenRC.powerflow.base import get_oracle

        from OpenRC.powerflow.base import ORACLES

        from OpenRC.cases.base import GridCase, validate_case_configuration

        if isinstance(escalate_to, (str, bytes, bytearray)):
            raise TypeError("escalate_to must be a sequence of oracle names or instances, not a string")
        validate_case_configuration(case, primary if not isinstance(primary, str) else None)
        self.case = case.snapshot() if isinstance(case, GridCase) else case
        self.escalation_names = list(escalate_to)
        for name in self.escalation_names:
            if not isinstance(name, str):
                validate_case_configuration(self.case, name)
            # Validate registry names eagerly even though escalation oracles
            # are instantiated lazily on first dispute.
            if isinstance(name, str):
                ORACLES.get(name)
            elif not callable(getattr(name, "verdict_at", None)):
                raise TypeError("escalate_to entries must be oracle names or oracles")
        self._lock = threading.RLock()
        self._resources = ExitStack()
        self._closed = False
        if not isinstance(primary, str) and not callable(getattr(primary, "verdict_at", None)):
            raise TypeError("primary must be an oracle name or an oracle")
        self.primary = get_oracle(primary, self.case) if isinstance(primary, str) else primary
        if isinstance(primary, str):
            self._resources.callback(self.primary.close)
        # built lazily: often never needed
        self._escalations: Optional[List[Tuple[str, Any]]] = None
        self._unavailable: dict[int, str] = {}

    def close(self) -> None:
        """Release owned models once; a closed adjudicator cannot be reused."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._resources.close()

    def __enter__(self) -> "Adjudicator":
        if self._closed:
            raise RuntimeError("the adjudicator is closed; construct a new one")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _oracles(self):
        from OpenRC.powerflow.base import get_oracle

        if self._escalations is None:
            oracles = []
            # If a later factory fails, already-created models must still be
            # released.  Ownership transfers only after all factories succeed.
            with ExitStack() as pending:
                for entry in self.escalation_names:
                    if isinstance(entry, str):
                        oracle = get_oracle(entry, self.case)
                        pending.callback(oracle.close)
                        oracles.append((entry, oracle))
                    else:
                        oracles.append((getattr(entry, "name", "oracle"), entry))
                self._resources.callback(pending.pop_all().close)
            self._escalations = oracles
        return self._escalations

    def judge(self, axes, point: np.ndarray, *,
              primary_verdict: Optional[Verdict] = None) -> Adjudication:
        """Decide one point, escalating unless the primary proves a verdict.

        ``primary_verdict`` lets a caller reuse an answer it already has, so a
        sweep does not pay for the same power flow twice.
        """
        with self._lock:
            return self._judge(axes, point, primary_verdict=primary_verdict)

    def _judge(self, axes, point: np.ndarray, *,
               primary_verdict: Optional[Verdict] = None) -> Adjudication:
        """Serialised decision, including lazy stage creation and cleanup."""
        if self._closed:
            raise RuntimeError("the adjudicator is closed; construct a new one")
        if isinstance(axes, (str, bytes, bytearray)):
            raise TypeError("axes must be a sequence of Axis objects, not a string")
        raw_pt = np.asarray(point)
        if raw_pt.dtype == bool:
            raise ValueError("point must be a numeric 1-D vector, not boolean")
        point = np.asarray(raw_pt, dtype=float)
        dim = None if axes is None else len(axes)
        if point.ndim != 1 or (dim is not None and point.shape[0] != dim):
            raise ValueError(f"point must be a 1-D vector of length {dim}, got shape {point.shape}")
        if not np.all(np.isfinite(point)):
            raise ValueError("point must contain only finite values")
        if primary_verdict is not None and not isinstance(primary_verdict, Verdict):
            if isinstance(primary_verdict, (bool, np.bool_)):
                raise ValueError("primary_verdict must be a Verdict, not a boolean")
            primary_verdict = Verdict(primary_verdict)
        verdict = (primary_verdict if primary_verdict is not None
                   else self.primary.verdict_at(axes, point))
        record = Adjudication(point=point, primary=verdict, final=verdict,
                              primary_decides=bool(getattr(self.primary,
                                                           "supports_adjudication", True)))

        evidence = _reference_evidence(self.primary, verdict)
        if evidence is not Verdict.INCONCLUSIVE:
            record.final = evidence
            return record

        nonconclusive = []
        if verdict is not Verdict.INCONCLUSIVE:
            nonconclusive.append(f"{self.primary.name} returned {verdict.value}")
        for name, oracle in self._oracles():
            if not getattr(oracle, "supports_adjudication", True):
                # Skip stages lacking a directional inclusion guarantee for the
                # reference model, while recording the skipped stage in the trail.
                record.escalations.append((name, Verdict.INCONCLUSIVE))
                nonconclusive.append(f"{name} skipped (unknown relationship to the reference)")
                continue
            compatible, reason = _model_compatibility(oracle, self.primary)
            if not compatible:
                record.escalations.append((name, Verdict.INCONCLUSIVE))
                nonconclusive.append(f"{name} skipped (different physical model: {reason})")
                continue
            # Cache expected solver/topology unavailability per oracle instance
            # to avoid repeated setup attempts; let unexpected exceptions propagate.
            identity = id(oracle)
            if identity not in self._unavailable:
                try:
                    answer = oracle.verdict_at(axes, point)
                except (SolverUnavailableError, MeshedNetworkError, AmbiguousSlackError) as exc:
                    self._unavailable[identity] = str(exc)
                    answer = Verdict.INCONCLUSIVE
            else:
                answer = Verdict.INCONCLUSIVE
            record.escalations.append((name, answer))
            if identity in self._unavailable:
                nonconclusive.append(f"{name} unavailable: {self._unavailable[identity]}")
                continue

            # Evidence is directional for approximations.  Feasibility in an
            # outer relaxation (for example SOCP) does not prove exact
            # feasibility; infeasibility does.  The direction is reversed for
            # an inner approximation.  Component models with no known
            # inclusion cannot prove either direction.  Local reference
            # rejection is not a global infeasibility certificate either.
            approximation = getattr(oracle, "inner_approximation", None)
            evidence = _reference_evidence(oracle, answer, reference=self.primary,
                                           legacy_infeasibility=True)

            if evidence is Verdict.FEASIBLE:
                record.final = Verdict.FEASIBLE
                if verdict is Verdict.INFEASIBLE and record.primary_decides:
                    record.note = (
                        f"{self.primary.name} decided infeasible (a solution outside a "
                        f"limit), but {name} found an operating point satisfying every "
                        "limit: the two models disagree on this decision. Possible "
                        "causes include an alternative power-flow solution or numerical "
                        "tolerances. Counted "
                        "as feasible; inspect it"
                    )
                else:
                    record.note = (
                        f"{self.primary.name} said {verdict.value}, but {name} found an "
                        "operating point satisfying every limit; the point is feasible "
                        "and the primary reference failed on it"
                    )
                return record
            if evidence is Verdict.INFEASIBLE:
                record.final = Verdict.INFEASIBLE
                record.note = f"confirmed infeasible by {name}"
                return record
            if answer is not Verdict.INCONCLUSIVE:
                if not getattr(oracle, "supports_adjudication", True):
                    detail = "unknown relationship to the reference"
                elif approximation is not None:
                    direction = "inner" if approximation else "outer"
                    detail = f"{direction} approximation"
                else:
                    detail = "no global infeasibility certificate"
                nonconclusive.append(f"{name} returned {answer.value} ({detail})")

        record.final = Verdict.INCONCLUSIVE
        detail = f" Non-conclusive evidence: {'; '.join(nonconclusive)}." if nonconclusive else ""
        record.note = (
            "neither the primary reference nor any escalation could prove a verdict; "
            "the point is excluded from the error rate and needs manual confirmation."
            + detail
        )
        return record

    def judge_many(self, axes, points: np.ndarray,
                   primary_verdicts: Optional[Sequence[Verdict]] = None
                   ) -> List[Adjudication]:
        if self._closed:
            raise RuntimeError("the adjudicator is closed; construct a new one")
        if isinstance(axes, (str, bytes, bytearray)):
            raise TypeError("axes must be a sequence of Axis objects, not a string")
        raw_in = np.asarray(points)
        if raw_in.dtype == bool:
            raise ValueError("points must be a numeric array, not boolean")
        if primary_verdicts is not None and isinstance(primary_verdicts, (str, bytes, bytearray)):
            raise TypeError("primary_verdicts must be a sequence of Verdicts, not a string")
        raw = np.asarray(raw_in, dtype=float)
        dim = None if axes is None else len(axes)
        if raw.size == 0:
            if raw.ndim != 1 and not (
                    raw.ndim == 2 and raw.shape[0] == 0
                    and (dim is None or raw.shape[1] in (0, dim))):
                if raw.ndim not in (1, 2):
                    raise ValueError(f"points must be 1D or 2D, got shape {raw.shape}")
                raise ValueError(f"points have dimension {raw.shape[1]}, expected {dim}")
            if primary_verdicts is not None and len(list(primary_verdicts)) != 0:
                raise ValueError(
                    f"primary_verdicts has {len(list(primary_verdicts))} entries for 0 points")
            return []
        pts = np.atleast_2d(raw)
        if pts.ndim != 2 or (dim is not None and pts.shape[1] != dim):
            raise ValueError(f"points have dimension {pts.shape[-1]}, expected {dim}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must contain only finite values")
        verdicts: List[Optional[Verdict]] = (
            list(primary_verdicts) if primary_verdicts is not None else [None] * len(pts))
        if len(verdicts) != len(pts):
            raise ValueError(
                f"primary_verdicts has {len(verdicts)} entries for {len(pts)} points")
        return [self.judge(axes, p, primary_verdict=v) for p, v in zip(pts, verdicts)]
