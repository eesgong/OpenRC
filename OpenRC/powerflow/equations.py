"""Independently verified local root-finding pre-solvers for feasibility models.

Whereas convex/global optimisers decide feasibility directly, local Newton or
Gauss-Seidel methods search for a single root of the equality equations.
Because non-convergence of a local root-finder does not prove infeasibility,
candidates are accepted as ``FEASIBLE`` only if all equality, bound, and cone
residuals satisfy ``tolerance``; otherwise the result is ``INCONCLUSIVE`` and
the calling oracle falls back to its optimiser.

``EquationProblem`` also provides a SciPy-compatible interface: its ``fun``,
``jac``, ``x0``, and ``bounds`` attributes can be passed to
``scipy.optimize.root`` or ``scipy.optimize.least_squares``.  Custom algorithms
can be registered in :data:`EQUATION_SOLVERS` while retaining independent
residual verification.
"""

from __future__ import annotations

import time
import threading
import warnings
import math
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Protocol, Tuple, cast

import numpy as np

from OpenRC.registry import Registry

Array = np.ndarray
Jacobian = Any

# Warning filters are process-global on supported Python configurations.
# Only sparse factorisation needs a temporary MatrixRankWarning policy.
_SPARSE_WARNING_LOCK = threading.RLock()


@dataclass(frozen=True)
class ConstraintResiduals:
    """Largest scaled violation in each constraint family.

    A value of zero is exactly satisfied.  Scaling by the magnitude of each
    constraint keeps one tolerance meaningful for both SI and per-unit models.
    """

    equality: float
    bounds: float
    cones: float

    @property
    def maximum(self) -> float:
        """Largest violation across equality, bound, and cone families.

        Explicitly checks finiteness and non-negativity because Python's
        ``max(0.0, float('nan'))`` evaluates to ``0.0``; invalid or non-finite
        entries return ``inf``.
        """
        values = (self.equality, self.bounds, self.cones)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            return math.inf
        return max(values)

    def as_dict(self) -> dict:
        return {
            "equality": self.equality,
            "bounds": self.bounds,
            "cones": self.cones,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class EquationProblem:
    """A SciPy-compatible equation system plus an independent acceptor.

    Parameters
    ----------
    names:
        Variable names, in the same order as ``x``.
    fun, jac:
        Equality residual and its Jacobian.  They have SciPy's usual
        ``fun(x)`` / ``jac(x)`` signatures.
    x0:
        A finite starting point.  Callers may replace it or provide several
        starts to :func:`solve_equations`.
    bounds:
        Lower and upper variable bounds.
    violations:
        Re-evaluates *all* model constraints, including inequalities not
        represented by ``fun``.
    tolerance:
        Maximum scaled violation accepted as numerically feasible.
    """

    names: Tuple[str, ...]
    fun: Callable[[Array], Array]
    jac: Callable[[Array], Jacobian]
    x0: Array
    bounds: Tuple[Array, Array]
    violations: Callable[[Array], ConstraintResiduals]
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if isinstance(self.names, (str, bytes, bytearray)):
            raise TypeError("names must be a sequence of variable names, not a string")
        object.__setattr__(self, "names", tuple(self.names))
        if not (callable(self.fun) and callable(self.jac) and callable(self.violations)):
            raise TypeError("fun, jac, and violations must be callable")
        if (isinstance(self.bounds, (str, bytes))
                or not hasattr(self.bounds, "__len__")
                or len(self.bounds) != 2):
            raise ValueError("bounds must be a (lower, upper) pair of arrays")
        raw_x0 = np.asarray(self.x0)
        raw_lo = np.asarray(self.bounds[0])
        raw_hi = np.asarray(self.bounds[1])
        if raw_x0.dtype == bool or raw_lo.dtype == bool or raw_hi.dtype == bool:
            raise ValueError("x0 and bounds must be numeric arrays, not boolean")
        object.__setattr__(self, "x0", np.asarray(raw_x0, dtype=float))
        object.__setattr__(
            self, "bounds",
            (np.asarray(raw_lo, dtype=float), np.asarray(raw_hi, dtype=float)),
        )
        expected = (len(self.names),)
        lo, hi = self.bounds
        if self.x0.shape != expected or lo.shape != expected or hi.shape != expected:
            raise ValueError(
                f"x0 and both bounds must have shape {expected}; got "
                f"{self.x0.shape}, {lo.shape}, and {hi.shape}"
            )
        if not np.all(np.isfinite(self.x0)):
            raise ValueError("x0 must contain only finite values")
        if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
            raise ValueError("bounds must not contain NaN")
        if np.any(lo > hi):
            raise ValueError("every lower bound must be less than or equal to its upper bound")
        if (isinstance(self.tolerance, (bool, np.bool_))
                or self.tolerance <= 0.0 or not np.isfinite(self.tolerance)):
            raise ValueError("tolerance must be a positive finite number")

    def check(self, x: Array) -> ConstraintResiduals:
        """Validate shape/finite values, then evaluate every constraint."""
        try:
            raw = np.asarray(x)
            if raw.dtype == bool:
                return ConstraintResiduals(float("inf"), float("inf"), float("inf"))
            candidate = np.asarray(raw, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return ConstraintResiduals(float("inf"), float("inf"), float("inf"))
        if candidate.shape != self.x0.shape or not np.all(np.isfinite(candidate)):
            return ConstraintResiduals(float("inf"), float("inf"), float("inf"))
        return self.violations(candidate)

    def accepts(self, x: Array) -> bool:
        """Whether ``x`` is a numerically certified feasible point."""
        return self.check(x).maximum <= self.tolerance

    def dense_jac(self, x: Array) -> Array:
        """Dense ``ndarray`` Jacobian for SciPy routines such as ``root(method='hybr')``."""
        from scipy import sparse

        j = self.jac(x)
        return np.asarray(cast(Any, j).toarray() if sparse.issparse(j) else j, dtype=float)


@dataclass(frozen=True)
class EquationResult:
    """Result of a local equation search; it never claims infeasibility."""

    method: str
    x: Optional[Array]
    accepted: bool
    converged: bool
    starts_tried: int
    iterations: int
    residuals: Optional[ConstraintResiduals]
    seconds: float
    message: str = ""

    @property
    def needs_fallback(self) -> bool:
        """A rejected local search must be escalated to the optimiser."""
        return not self.accepted

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "accepted": self.accepted,
            "converged": self.converged,
            "starts_tried": self.starts_tried,
            "iterations": self.iterations,
            "residuals": None if self.residuals is None else self.residuals.as_dict(),
            "seconds": round(self.seconds, 6),
            "message": self.message,
        }


@dataclass
class EquationSolverStats:
    """Lightweight counters for an optional local pre-solver."""

    attempts: int = 0
    accepted: int = 0
    fallbacks: int = 0
    iterations: int = 0
    seconds: float = 0.0
    #: Set to ``True`` by :meth:`gate` when the acceptance rate falls below ``min_rate``.
    disabled: bool = False
    #: Attempt on which the circuit breaker opened; useful in saved provenance.
    disabled_after: Optional[int] = None

    def record(self, result: EquationResult) -> None:
        self.attempts += 1
        self.accepted += int(result.accepted)
        self.fallbacks += int(result.needs_fallback)
        self.iterations += int(result.iterations)
        self.seconds += result.seconds

    def gate(self, min_rate: float, patience: int) -> None:
        """Disable the pre-solver after ``patience`` attempts below ``min_rate``.

        A method that cannot solve a model (generic Newton on the
        under-determined SOCP form, for instance) would otherwise tax
        every query.  ``min_rate=0`` never disables.
        """
        if (
            isinstance(min_rate, (bool, np.bool_))
            or not isinstance(min_rate, (int, float, np.number))
            or not 0.0 <= min_rate <= 1.0
        ):
            raise ValueError("min_rate must lie in [0, 1]")
        patience = _positive_integer(patience, "patience")
        if min_rate > 0.0 and self.attempts >= patience \
                and self.accepted < min_rate * self.attempts:
            self.disabled = True
            if self.disabled_after is None:
                self.disabled_after = self.attempts

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "fallbacks": self.fallbacks,
            "iterations": self.iterations,
            "seconds": round(self.seconds, 6),
            "disabled": self.disabled,
            "disabled_after": self.disabled_after,
        }


class EquationAlgorithm(Protocol):
    """Signature for one-start equation algorithms stored in the registry."""

    def __call__(
        self, problem: EquationProblem, x0: Array, options: Mapping[str, Any]
    ) -> Tuple[Optional[Array], bool, int, str]:
        """Return ``(candidate, equations_converged, iterations, message)``."""


EQUATION_SOLVERS: Registry[EquationAlgorithm] = Registry("equation solver")

#: Candidate solvers that use the network's structure rather than a
#: :class:`~OpenRC.powerflow.modeling.ConicForm`: ``factory(case, **options)``
#: returning an object whose ``solve(p_mw, q_mvar)`` gives an
#: :class:`EquationResult`.  The SOCP and exact oracles accept them by name.
STRUCTURED_SOLVERS: Registry = Registry("structured candidate solver")


def check_candidate(name: str, *, structured: bool = False) -> None:
    """Raise ``KeyError`` for a name that is not a registered candidate solver."""
    if name in EQUATION_SOLVERS or (structured and name in STRUCTURED_SOLVERS):
        return
    if not structured:
        EQUATION_SOLVERS.get(name)                 # the registry's own message
    known = EQUATION_SOLVERS.names() + STRUCTURED_SOLVERS.names()
    raise KeyError(f"unknown equation solver or structured solver {name!r}; "
                   f"available: {', '.join(sorted(known))}")


def list_equation_solvers() -> List[Tuple[str, str]]:
    """Return the registered local equation algorithms and their descriptions."""
    return EQUATION_SOLVERS.describe()


def list_structured_solvers() -> List[Tuple[str, str]]:
    """Return candidates that exploit network structure, such as sweeps."""
    return STRUCTURED_SOLVERS.describe()


def _positive_integer(value: Any, name: str) -> int:
    """Validate and convert ``value`` to a positive integer, rejecting booleans and non-integral floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if not np.isfinite(numeric) or numeric < 1.0 or int(numeric) != numeric:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _non_negative_integer(value: Any, name: str) -> int:
    """Normalise a non-negative integer option without truncating floats or accepting bools."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be non-negative")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be non-negative") from None
    if not np.isfinite(numeric) or numeric < 0.0 or int(numeric) != numeric:
        raise ValueError(f"{name} must be non-negative")
    return int(numeric)


def _finite_option(value: Any, message: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _iteration_count(value: Any) -> int:
    """Validate an extension's counter before it can corrupt shared statistics."""
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_)) or value < 0):
        raise ValueError("iteration counts must be non-negative integers")
    return int(value)


def _project(x: Array, bounds: Tuple[Array, Array]) -> Array:
    """Project onto finite box bounds without modifying the caller's array."""
    lo, hi = bounds
    return np.minimum(np.maximum(np.asarray(x, dtype=float), lo), hi)


def _linear_step(jacobian: Jacobian, residual: Array) -> Optional[Array]:
    """Solve ``J step = -residual`` robustly for dense or sparse ``J``."""
    from scipy import sparse
    from scipy.sparse.linalg import MatrixRankWarning, lsqr, spsolve

    rows, cols = jacobian.shape
    try:
        if rows == cols:
            if sparse.issparse(jacobian):
                with _SPARSE_WARNING_LOCK, warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    step = spsolve(cast(Any, jacobian).tocsc(), -residual)
            else:
                step = np.linalg.solve(np.asarray(jacobian), -residual)
        else:
            matrix = (cast(Any, jacobian) if sparse.issparse(jacobian)
                      else sparse.csr_matrix(jacobian))
            step = lsqr(matrix, -residual, atol=1e-12, btol=1e-12)[0]
    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError, MatrixRankWarning):
        return None
    answer = np.asarray(step, dtype=float)
    return answer if np.all(np.isfinite(answer)) else None


@EQUATION_SOLVERS.register(
    "newton", doc="Damped Newton with sparse linear solves and bound projection"
)
def _damped_newton(
    problem: EquationProblem, x0: Array, options: Mapping[str, Any]
) -> Tuple[Optional[Array], bool, int, str]:
    """Damped Newton; affine full-rank systems converge in one step."""
    max_iterations = _non_negative_integer(options.get("max_iterations", 30), "max_iterations")
    residual_tolerance = _finite_option(
        options.get("residual_tolerance", 1e-11), "Newton tolerances must be positive"
    )
    step_tolerance = _finite_option(
        options.get("step_tolerance", 1e-12), "Newton tolerances must be positive"
    )
    min_damping = _finite_option(
        options.get("min_damping", 2.0 ** -16), "min_damping must be in (0, 1]"
    )
    if residual_tolerance <= 0.0 or step_tolerance <= 0.0:
        raise ValueError("Newton tolerances must be positive")
    if not (0.0 < min_damping <= 1.0):
        raise ValueError("min_damping must be in (0, 1]")
    x = _project(np.asarray(x0, dtype=float), problem.bounds)

    for iteration in range(max_iterations + 1):
        residual = np.asarray(problem.fun(x), dtype=float)
        if not np.all(np.isfinite(residual)):
            return x, False, iteration, "non-finite equation residual"
        residual_norm = float(np.linalg.norm(residual, ord=np.inf)) if residual.size else 0.0
        if residual_norm <= residual_tolerance:
            return x, True, iteration, "equations converged"
        if iteration == max_iterations:
            break
        step = _linear_step(problem.jac(x), residual)
        if step is None:
            return x, False, iteration, "singular or ill-conditioned Jacobian"
        if np.linalg.norm(step, ord=np.inf) <= step_tolerance * (
            1.0 + np.linalg.norm(x, ord=np.inf)
        ):
            return x, False, iteration, "Newton step stagnated"

        damping = 1.0
        accepted_step = False
        while damping >= min_damping:
            trial = _project(x + damping * step, problem.bounds)
            trial_residual = np.asarray(problem.fun(trial), dtype=float)
            if np.all(np.isfinite(trial_residual)):
                trial_norm = (
                    float(np.linalg.norm(trial_residual, ord=np.inf))
                    if trial_residual.size else 0.0
                )
                if trial_norm < residual_norm:
                    x = trial
                    accepted_step = True
                    break
            damping *= 0.5
        if not accepted_step:
            return x, False, iteration + 1, "line search could not reduce the residual"

    return x, False, max_iterations, "iteration limit reached"


def _matched_rows(jacobian: Jacobian) -> Optional[Array]:
    """Find a non-zero diagonal by a sparse bipartite row matching."""
    from scipy import sparse
    from scipy.sparse.csgraph import maximum_bipartite_matching

    if jacobian.shape[0] != jacobian.shape[1]:
        return None
    matrix = (cast(Any, jacobian).tocsr()
              if sparse.issparse(jacobian) else sparse.csr_matrix(jacobian))
    structure = matrix.copy()
    structure.data = np.asarray(structure.data != 0.0, dtype=np.int8)
    structure.eliminate_zeros()
    # perm_type="row" maps each column j to a row order[j] with a non-zero entry.
    order = maximum_bipartite_matching(structure, perm_type="row")
    if np.any(order < 0):
        return None
    return np.asarray(order, dtype=int)


@EQUATION_SOLVERS.register(
    "gauss-seidel", doc="Damped nonlinear Gauss-Seidel with structural row matching"
)
def _gauss_seidel(
    problem: EquationProblem, x0: Array, options: Mapping[str, Any]
) -> Tuple[Optional[Array], bool, int, str]:
    """Solve square systems by sequential variable updates.

    The equations and variables of a branch-flow model are not stored in
    diagonal order.  A sparse structural matching first permutes the
    rows so that every diagonal entry is non-zero.  This is computed once per
    call; :class:`PreparedEquationSolver` caches it across oracle queries.
    """
    max_iterations = _non_negative_integer(options.get("max_iterations", 500), "max_iterations")
    residual_tolerance = _finite_option(
        options.get("residual_tolerance", 1e-10), "residual_tolerance must be positive"
    )
    omega = _finite_option(
        options.get("omega", 1.0), "Gauss-Seidel omega must be between 0 and 2"
    )
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")
    if not (0.0 < omega < 2.0):
        raise ValueError("Gauss-Seidel omega must be between 0 and 2")
    x = _project(np.asarray(x0, dtype=float), problem.bounds)
    order = _matched_rows(problem.jac(x))
    if order is None:
        return None, False, 0, "Gauss-Seidel needs a structurally nonsingular square Jacobian"

    for iteration in range(max_iterations + 1):
        residual = np.asarray(problem.fun(x), dtype=float)
        if not np.all(np.isfinite(residual)):
            return x, False, iteration, "non-finite equation residual"
        if residual.size == 0 or np.linalg.norm(residual, ord=np.inf) <= residual_tolerance:
            return x, True, iteration, "equations converged"
        if iteration == max_iterations:
            break
        for column, row in enumerate(order):
            jacobian = problem.jac(x)
            diagonal = float(jacobian[row, column])
            if not np.isfinite(diagonal) or abs(diagonal) <= np.finfo(float).tiny:
                return x, False, iteration, "zero diagonal during Gauss-Seidel iteration"
            residual_row = float(problem.fun(x)[row])
            x[column] -= omega * residual_row / diagonal
    return x, False, max_iterations, "iteration limit reached"


def _starts(
    problem: EquationProblem,
    starts: Optional[Iterable[Array]],
    n_starts: int,
    seed: Optional[int],
    start_scale: float,
) -> Iterable[Array]:
    """Yield explicit starts, or a deterministic bounded multi-start sequence."""
    if starts is not None:
        for start in starts:
            # Conversion belongs inside the per-start guard in solve_equations.
            # A malformed start must not prevent trying the remaining ones.
            yield start
        return
    yield np.asarray(problem.x0, dtype=float).copy()
    if n_starts <= 1:
        return
    rng = np.random.default_rng(seed)
    lo, hi = problem.bounds
    for _ in range(n_starts - 1):
        finite_box = np.isfinite(lo) & np.isfinite(hi)
        start = problem.x0 + rng.normal(
            scale=start_scale * np.maximum(np.abs(problem.x0), 1.0),
            size=problem.x0.shape,
        )
        start[finite_box] = rng.uniform(lo[finite_box], hi[finite_box])
        yield _project(start, problem.bounds)


def solve_equations(
    problem: EquationProblem,
    *,
    method: str = "newton",
    starts: Optional[Iterable[Array]] = None,
    n_starts: int = 1,
    seed: Optional[int] = 0,
    start_scale: float = 0.1,
    options: Optional[Mapping[str, Any]] = None,
) -> EquationResult:
    """Try one or more starting points and return the first verified feasible candidate.

    Returns ``accepted=False`` when no starting point yields a feasible solution;
    ``accepted=False`` indicates an inconclusive local search, not infeasibility.
    """
    if not isinstance(problem, EquationProblem):
        raise TypeError(f"problem must be an EquationProblem; got {type(problem).__name__}")
    n_starts = _positive_integer(n_starts, "n_starts")
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must not be a boolean")
    if isinstance(start_scale, (bool, np.bool_)) or start_scale < 0.0 or not np.isfinite(start_scale):
        raise ValueError("start_scale must be a non-negative finite number")
    if starts is not None:
        if isinstance(starts, (str, bytes, bytearray)):
            raise TypeError("starts must be an iterable of starting vectors, not a string")
        if isinstance(starts, np.ndarray) and starts.ndim == 1:
            raise ValueError("starts array must be 2-D of shape (n_starts, n_var)")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    algorithm = EQUATION_SOLVERS.get(method)
    opts = dict(options or {})
    start_time = time.perf_counter()
    tried = 0
    iterations = 0
    last_x: Optional[Array] = None
    last_report: Optional[ConstraintResiduals] = None
    last_converged = False
    last_message = "no starting points were supplied"

    for initial in _starts(problem, starts, n_starts, seed, start_scale):
        tried += 1
        try:
            raw_initial = np.asarray(initial)
            if raw_initial.dtype == bool:
                raise ValueError("starting point must not be boolean")
            initial = np.asarray(raw_initial, dtype=float)
        except (TypeError, ValueError, OverflowError):
            last_message = "ignored a start that cannot be converted to finite numbers"
            continue
        if initial.shape != problem.x0.shape or not np.all(np.isfinite(initial)):
            last_message = "ignored a start with the wrong shape or non-finite values"
            continue
        report: Optional[ConstraintResiduals] = None
        try:
            candidate, converged, count, message = algorithm(problem, initial, opts)
            count = _iteration_count(count)
            if candidate is not None:
                candidate = np.asarray(candidate, dtype=float)
                report = problem.check(candidate)
                if not isinstance(report, ConstraintResiduals):
                    raise TypeError("candidate validation must return ConstraintResiduals")
        except Exception as exc:
            candidate, converged, count = None, False, 0
            report = None
            message = f"equation algorithm or candidate validation raised {type(exc).__name__}: {exc}"
        iterations += count
        last_converged, last_message = converged, message
        if candidate is None or report is None:
            continue
        last_x, last_report = candidate, report
        if report.maximum <= problem.tolerance:
            return EquationResult(
                method, candidate, True, converged, tried, iterations, report,
                time.perf_counter() - start_time, message,
            )

    return EquationResult(
        method, last_x, False, last_converged, tried, iterations, last_report,
        time.perf_counter() - start_time, last_message,
    )


class PreparedEquationSolver:
    """Reusable equation pre-solver for one affine :class:`ConicForm`.

    For a square affine form, ``newton`` is exactly one sparse LU solve.  The LU
    factors are cached and only the injection-dependent right-hand side changes
    between queries.  Gauss-Seidel likewise caches its row matching.  General,
    under-determined or user-registered methods use :func:`solve_equations`.
    """

    def __init__(
        self,
        form: Any,
        *,
        method: str = "newton",
        n_starts: int = 1,
        seed: Optional[int] = 0,
        options: Optional[Mapping[str, Any]] = None,
        tolerance: float = 1e-9,
    ) -> None:
        if not (
            hasattr(form, "A")
            and callable(getattr(form, "rhs", None))
            and callable(getattr(form, "scipy_problem", None))
        ):
            raise TypeError(f"form must be a ConicForm; got {type(form).__name__}")
        if method not in EQUATION_SOLVERS:
            # Registry.get gives the more useful list of available methods.
            EQUATION_SOLVERS.get(method)
        n_starts = _positive_integer(n_starts, "n_starts")
        if isinstance(seed, (bool, np.bool_)):
            raise TypeError("seed must not be a boolean")
        if isinstance(tolerance, (bool, np.bool_)) or tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be a positive finite number")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("options must be a mapping")
        self.form = form
        self.method = method
        self.n_starts = n_starts
        self.seed = seed
        self.options = dict(options or {})
        self.tolerance = float(tolerance)
        self._factor: Optional[Callable[[Array], Array]] = None
        self._row_order: Optional[Array] = None
        self._gs_matrix: Any = None
        self._gs_diagonal: Optional[Array] = None
        self._gs_lower: Any = None
        self._gs_upper: Any = None

        if form.A.shape[0] == form.A.shape[1] and not any(
            c.exact for c in getattr(form, "cones", ())
        ):
            if method == "newton":
                from scipy.sparse.linalg import factorized

                try:
                    self._factor = factorized(form.A.tocsc())
                except (RuntimeError, ValueError):
                    self._factor = None
            elif method == "gauss-seidel":
                self._row_order = _matched_rows(form.A)
                if self._row_order is not None:
                    from scipy import sparse

                    self._gs_matrix = form.A.tocsr()[self._row_order, :].tocsr()
                    self._gs_diagonal = np.asarray(self._gs_matrix.diagonal(), dtype=float)
                    self._gs_lower = sparse.tril(self._gs_matrix, format="csr")
                    self._gs_upper = sparse.triu(self._gs_matrix, k=1, format="csr")

    def solve(self, p_mw: Array, q_mvar: Array) -> EquationResult:
        """Try a candidate for one injection vector and validate it strictly."""
        problem = self.form.scipy_problem(
            p_mw, q_mvar, tolerance=self.tolerance
        )
        started = time.perf_counter()

        if self.method == "newton" and self._factor is not None:
            try:
                x = np.asarray(self._factor(self.form.rhs(p_mw, q_mvar)), dtype=float)
                report = problem.check(x)
                accepted = report.maximum <= problem.tolerance
                converged = report.equality <= problem.tolerance
                return EquationResult(
                    self.method, x, accepted, converged, 1, 1, report,
                    time.perf_counter() - started,
                    "cached sparse Newton factorisation",
                )
            except (ArithmeticError, RuntimeError, ValueError):
                # The generic path records an inconclusive result and lets the
                # oracle fall back to the optimiser.
                pass

        if self.method == "gauss-seidel" and self._row_order is not None:
            return self._solve_affine_gauss_seidel(problem, p_mw, q_mvar, started)

        return solve_equations(
            problem, method=self.method, n_starts=self.n_starts, seed=self.seed,
            options=self.options,
        )

    def _solve_affine_gauss_seidel(
        self, problem: EquationProblem, p_mw: Array, q_mvar: Array, started: float
    ) -> EquationResult:
        """Gauss-Seidel with cached row order and incremental residual updates."""
        ordered = self._gs_matrix
        diagonal = self._gs_diagonal
        lower, upper = self._gs_lower, self._gs_upper
        assert ordered is not None and diagonal is not None
        assert lower is not None and upper is not None
        rhs = np.asarray(self.form.rhs(p_mw, q_mvar))[self._row_order]
        x = np.asarray(problem.x0, dtype=float).copy()
        max_iterations = _non_negative_integer(
            self.options.get("max_iterations", 500), "max_iterations"
        )
        residual_tolerance = _finite_option(
            self.options.get("residual_tolerance", 1e-10), "residual_tolerance must be positive"
        )
        omega = _finite_option(
            self.options.get("omega", 1.0), "Gauss-Seidel omega must be between 0 and 2"
        )
        if residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive")
        if not (0.0 < omega < 2.0):
            raise ValueError("Gauss-Seidel omega must be between 0 and 2")
        if np.any(np.abs(diagonal) <= np.finfo(float).tiny):
            return EquationResult(
                self.method, x, False, False, 1, 0, problem.check(x),
                time.perf_counter() - started, "zero diagonal after row matching",
            )
        from scipy.sparse.linalg import spsolve_triangular

        converged = False
        count = 0
        for count in range(max_iterations + 1):
            residual = ordered @ x - rhs
            if np.linalg.norm(residual, ord=np.inf) <= residual_tolerance:
                converged = True
                break
            if count == max_iterations:
                break
            sweep = np.asarray(
                spsolve_triangular(lower, rhs - upper @ x, lower=True), dtype=float
            )
            x = (1.0 - omega) * x + omega * sweep
        report = problem.check(x)
        accepted = converged and report.maximum <= problem.tolerance
        return EquationResult(
            self.method, x, accepted, converged, 1, count, report,
            time.perf_counter() - started,
            "cached affine Gauss-Seidel" if converged else "iteration limit reached",
        )


__all__ = [
    "ConstraintResiduals", "EQUATION_SOLVERS", "STRUCTURED_SOLVERS",
    "EquationAlgorithm", "EquationProblem",
    "EquationResult", "EquationSolverStats", "PreparedEquationSolver",
    "list_equation_solvers", "list_structured_solvers",
    "solve_equations",
]


class CandidateMixin:
    """Mixin providing optional local pre-solving ahead of an oracle's optimiser.

    A verified candidate returns ``Verdict.FEASIBLE``; rejected candidates,
    exceptions, or disabled pre-solvers return ``None`` so the caller falls back
    to its optimiser.  Oracles call :meth:`_init_candidate` in ``__init__`` and
    :meth:`_candidate_verdict` before invoking their optimiser.
    """

    case: Any
    _lock: Any
    _direct: Any
    _solver: Any
    _model: Any
    _candidate: Any

    def close(self) -> None:
        """Release direct/Pyomo solvers and optional candidates under the lock.

        Clear the reference before disposal so a repeated close never invokes
        a native solver handle twice.  The next query can build a fresh model.
        """
        from OpenRC.powerflow.base import _close_solver

        with self._lock:
            direct, self._direct = self._direct, None
            solver, self._solver = getattr(self, "_solver", None), None
            candidate, self._candidate = getattr(self, "_candidate", None), None
            self._model = None
            for name in ("_p_param", "_q_param"):
                if hasattr(self, name):
                    setattr(self, name, None)
            with ExitStack() as resources:
                if direct is not None:
                    resources.callback(direct.close)
                resources.callback(_close_solver, solver)
                close_candidate = getattr(candidate, "close", None)
                if callable(close_candidate):
                    resources.callback(close_candidate)

    def _init_candidate(self, name: Optional[str], *, starts: int = 1,
                        seed: Optional[int] = 0, options: Optional[dict] = None,
                        tolerance: float = 1e-9, min_rate: float = 0.05,
                        patience: int = 20, structured: bool = False) -> None:
        if not isinstance(structured, (bool, np.bool_)):
            raise ValueError("structured must be a boolean")
        if name is not None:
            check_candidate(name, structured=bool(structured))
        starts = _positive_integer(starts, "candidate_starts")
        if isinstance(seed, (bool, np.bool_)):
            raise TypeError("candidate_seed must not be a boolean")
        if isinstance(tolerance, (bool, np.bool_)) or tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError("candidate_tolerance must be a positive finite number")
        if isinstance(min_rate, (bool, np.bool_)) or not 0.0 <= min_rate <= 1.0:
            raise ValueError("candidate_min_rate must lie in [0, 1]")
        patience = _positive_integer(patience, "candidate_patience")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("candidate_options must be a mapping")
        self.candidate_solver = name
        self.candidate_starts = starts
        self.candidate_seed = seed
        self.candidate_options = dict(options or {})
        self.candidate_tolerance = float(tolerance)
        self.candidate_min_rate = float(min_rate)
        self.candidate_patience = patience
        self.candidate_stats = EquationSolverStats()
        self.last_candidate: Optional[EquationResult] = None
        self._candidate = None

    def _conic_form(self) -> Any:     # provided by the oracle
        raise NotImplementedError

    def _candidate_verdict(self, p_mw: Array, q_mvar: Array) -> Optional[Any]:
        """``Verdict.FEASIBLE`` for a verified candidate, else ``None`` (fall back)."""
        from OpenRC.powerflow.base import Verdict

        if self.candidate_solver is None or self.candidate_stats.disabled:
            return None
        started = time.perf_counter()
        try:
            if self._candidate is None:
                name = self.candidate_solver
                if name in STRUCTURED_SOLVERS and name not in EQUATION_SOLVERS:
                    self._candidate = STRUCTURED_SOLVERS.get(name)(
                        self.case, **self.candidate_options)
                else:
                    self._candidate = PreparedEquationSolver(
                        self._conic_form(), method=name, n_starts=self.candidate_starts,
                        seed=self.candidate_seed, options=self.candidate_options,
                        tolerance=self.candidate_tolerance)
            # The oracle has already applied case.branch_flow_q; a structured
            # solver that applies it itself exposes solve_branch_flow.
            solve = getattr(self._candidate, "solve_branch_flow", self._candidate.solve)
            result = solve(p_mw, q_mvar)
            if not isinstance(result, EquationResult):
                raise TypeError("candidate solvers must return EquationResult")
            _iteration_count(result.iterations)
            _iteration_count(result.starts_tried)
            if not isinstance(result.accepted, (bool, np.bool_)):
                raise TypeError("candidate acceptance must be boolean")
            if not math.isfinite(result.seconds) or result.seconds < 0.0:
                raise ValueError("candidate timing must be finite and non-negative")
        except Exception as exc:
            # Catch candidate-solver errors and record an unaccepted EquationResult so
            # the caller falls back to the reference optimiser.
            result = EquationResult(
                str(self.candidate_solver), None, False, False, 0, 0, None,
                time.perf_counter() - started,
                f"candidate solver raised {type(exc).__name__}: {exc}",
            )
        self.last_candidate = result
        self.candidate_stats.record(result)
        self.candidate_stats.gate(self.candidate_min_rate, self.candidate_patience)
        return Verdict.FEASIBLE if result.accepted else None

    def _candidate_info(self) -> dict:
        certificate = (getattr(self._candidate, "last_certificate", None)
                       if self._candidate is not None else None)
        return {
            "candidate_solver": self.candidate_solver,
            "candidate_stats": (self.candidate_stats.as_dict()
                                if self.candidate_solver is not None else None),
            "last_candidate": (self.last_candidate.as_dict()
                               if self.last_candidate is not None else None),
            "last_candidate_certificate": (
                certificate.as_dict() if certificate is not None else None
            ),
        }
