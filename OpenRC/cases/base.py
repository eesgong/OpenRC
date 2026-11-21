"""Network case container (:class:`GridCase`) and configuration hashing.

Oracles, region builders, and applications consume a :class:`GridCase` snapshot
wrapping a ``pandapower`` network, per-bus base injections, and voltage limits.

Sign and unit conventions
-------------------------
* **Units.** Public API powers are three-phase **MW / Mvar**, line-to-line
  voltages **kV**, and currents **kA**.  Per-phase SI conversions (W, V, A,
  ohm) for branch-flow models are performed in :mod:`OpenRC.powerflow.distflow`.

* **Sign.** A positive nodal value is *consumption*.  A region coordinate is the
  *additional* load a POC may draw; a negative coordinate is generation/export.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from functools import cached_property, partial
import hashlib
import pickle
from types import (BuiltinFunctionType, CodeType, FunctionType,
                   MappingProxyType, MethodType, ModuleType)
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from OpenRC.cases.axes import _AXIS_PATTERN, AxisSpec, RegionAxes
from OpenRC.cases._copy import (_callable_state, _function_globals, _slot_names,
                                copy_configuration)

#: Fixed-power tables whose P/Q follow ``load_scale``.
_SCALED_TABLES = ("load", "sgen", "storage")
#: Net dictionary key storing original unscaled P/Q tables so ``rebuild()`` /
#: ``dataclasses.replace`` rescales from the unscaled baseline.
_UNSCALED_KEY = "_openrc_unscaled_injections"


def _configuration_token(value: Any, *, network: Any = None) -> Any:
    """Value-based hashing input, independent of pandas' internal caches.

    Recurse through object columns: geodata and extension tables may contain
    dictionaries, arrays or numerical callbacks. Python functions carry their
    code and captured parameters; import modules remain named dependencies.
    Pickle is only written into a digest here; no external pickle is loaded.
    """
    import pandas as pd

    active: Dict[int, int] = {}

    def token(item: Any) -> Any:
        if network is not None and item is network:
            # Emit a fixed token for back-references to the root network object;
            # the filtered physical tables are already hashed at the root.
            return ("network-reference",)
        identity = id(item)
        if identity in active:
            return ("cycle", len(active) - active[identity])
        active[identity] = len(active)
        try:
            if isinstance(item, ModuleType):
                return ("module", item.__name__, getattr(item, "__version__", None))
            if isinstance(item, CodeType):
                # Source paths/line numbers do not change a curve's physics.
                # Exception tables arrived in 3.11; all other fields below are
                # available throughout the supported Python 3.9+ range.
                return ("code", item.co_argcount, item.co_posonlyargcount,
                        item.co_kwonlyargcount, item.co_nlocals, item.co_stacksize,
                        item.co_flags, item.co_code, token(item.co_consts),
                        token(item.co_names), token(item.co_varnames),
                        token(item.co_freevars), token(item.co_cellvars),
                        getattr(item, "co_exceptiontable", b""))
            if isinstance(item, FunctionType):
                captures = []
                for cell in item.__closure__ or ():
                    try:
                        captures.append(token(cell.cell_contents))
                    except ValueError:
                        captures.append(("empty-cell",))
                return ("function", item.__module__, item.__globals__.get("__name__"),
                        item.__globals__.get("__package__"),
                        token(item.__code__), token(item.__defaults__),
                        token(item.__kwdefaults__), tuple(captures),
                        token(_function_globals(item)), token(vars(item)))
            if isinstance(item, MethodType):
                return ("method", token(item.__func__), token(item.__self__))
            if isinstance(item, partial):
                return ("partial", token(item.func), token(item.args),
                        token(item.keywords), token(vars(item)))
            if isinstance(item, BuiltinFunctionType):
                owner = getattr(item, "__self__", None)
                return ("builtin", item.__module__, item.__qualname__, token(owner))
            if isinstance(item, type):
                return ("type", item.__module__, item.__qualname__)
            if isinstance(item, np.ufunc):
                return ("ufunc", item.__name__, item.nin, item.nout)
            if (type(item).__name__ == "_ArrayFunctionDispatcher"
                    and type(item).__module__.startswith("numpy")):
                return ("numpy-function", item.__module__, item.__qualname__, np.__version__)
            if isinstance(item, pd.DataFrame):
                return ("frame", token(item.index), token(item.columns),
                        tuple(token(dtype) for dtype in item.dtypes),
                        token(item.to_numpy(copy=False)))
            if isinstance(item, pd.Series):
                return ("series", token(item.index), token(item.name),
                        token(item.dtype), token(item.to_numpy(copy=False)))
            if isinstance(item, pd.Index):
                return ("index", type(item).__name__, token(item.dtype),
                        token(item.to_numpy()), token(list(item.names)))
            if isinstance(item, pd.CategoricalDtype):
                return ("category", token(item.categories), item.ordered)
            if isinstance(item, (np.dtype, pd.api.extensions.ExtensionDtype)):
                return ("dtype", str(item))
            if isinstance(item, np.ndarray):
                if item.dtype.hasobject:
                    return ("objects", item.shape, tuple(token(child) for child in item.flat))
                return ("array", item.dtype.str, item.shape, item.tobytes(order="C"))
            if isinstance(item, dict):
                return ("mapping", tuple((token(key), token(child)) for key, child in
                                         sorted(item.items(), key=lambda pair: repr(pair[0]))))
            if isinstance(item, MappingProxyType):
                return ("mappingproxy", tuple((token(key), token(child)) for key, child in
                                              sorted(item.items(), key=lambda pair: repr(pair[0]))))
            if isinstance(item, (tuple, list)):
                return (type(item).__name__, tuple(token(child) for child in item))
            if isinstance(item, (set, frozenset)):
                return (type(item).__name__, tuple(token(child) for child in sorted(item, key=repr)))
            if callable(item):
                state = _callable_state(item)
                slots = tuple((name, token(getattr(item, name))) for name in _slot_names(item)
                              if hasattr(item, name))
                return ("callable", type(item).__module__, type(item).__qualname__,
                        getattr(item, "__module__", None), getattr(item, "__qualname__", None),
                        token(type(item).__call__), token(state), slots)
            if hasattr(item, "__dict__"):
                return ("object", type(item).__module__, type(item).__qualname__, token(vars(item)))
            return item
        finally:
            active.pop(identity)

    return token(value)


#: Values of :attr:`GridCase.line_charging`.
LINE_CHARGING = ("omit", "constant")


@dataclass
class GridCase:
    """A radial distribution network plus everything OpenRC needs to use it.

    Attributes
    ----------
    name:
        Registry key, e.g. ``"9bus"``.
    net:
        Underlying ``pandapower`` network used by AC/DC/OPF oracles and
        topology extraction.
    base_p_mw, base_q_mvar:
        Per-bus aggregated base injection (consumption positive), length
        ``n_bus``.  Several ``pandapower`` loads on one bus are already summed.
    v_nom_kv:
        Nominal line-to-line voltage, the base of every per-unit quantity.
    vm_min_pu, vm_max_pu:
        Voltage envelope applied to every non-slack bus.
    axis_presets:
        Named region axis sets, e.g. ``{"247": [1, 3, 6]}``.  Bus indices are
        0-based while the preset names follow the 1-based numbering of the
        papers, so "POC 2,4,7" is stored under the key ``"247"``.
    load_scale:
        Proportional scaling applied to ``load``, ``sgen``, and ``storage`` P/Q
        (while voltage-controlled ``gen`` dispatch is unchanged) and to
        :attr:`effective_base_p_mw` / :attr:`effective_base_q_mvar`.  ``1.0``
        uses recorded loads; ``< 1.0`` scales heavily loaded feeders into a
        feasible operating range; ``0.0`` clears fixed base loads.

    Configuration lifetime
    ----------------------
    Constructed ``GridCase`` instances are treated as immutable configurations.
    Use :meth:`rebuild` for scalar changes, or edit a private net and import it
    again with :func:`~OpenRC.cases.case_from_pandapower`. Oracles own detached
    snapshots; later edits of the source cannot change an existing oracle.
    Construction/task boundaries detect edits without hashing on every query.
    """

    name: str
    net: Any                       # pandapower net (untyped library)
    base_p_mw: np.ndarray
    base_q_mvar: np.ndarray
    v_nom_kv: float
    vm_min_pu: float = 0.9
    vm_max_pu: float = 1.1
    axis_presets: Dict[str, List[Union[int, str]]] = field(default_factory=dict)
    description: str = ""
    #: Multiplier applied to ``load``, ``sgen``, and ``storage`` P/Q (``gen`` dispatch is unchanged).
    load_scale: float = 1.0
    #: Explicit slack/infeed bus index.  When ``None``, the slack bus is inferred
    #: automatically (see :attr:`slack_bus`).
    slack_bus_override: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Keyword arguments passed to the case factory, serialized for worker reconstruction.
    factory_kwargs: Dict[str, Any] = field(default_factory=dict, repr=False)
    #: Set by :func:`get_case`; ``None`` means workers cannot safely assume a
    #: same-named registry entry reproduces this live object.
    registry_name: Optional[str] = field(default=None, repr=False)
    #: Voltage magnitude the infeed holds, per unit of :attr:`v_nom_kv`.  The
    #: branch-flow models (DistFlow, SOCP, exact, the sweep, the certificate)
    #: fix the slack at this value; the AC/DC oracles read the network's own
    #: external grid.  ``1.0`` is the nominal voltage.
    slack_vm_pu: float = 1.0
    #: Line shunt capacitance in the branch-flow models, which otherwise omit
    #: it (AC/DC keep the full line model).  ``"omit"`` (default) leaves it
    #: out.  ``"constant"`` uses each line's pi model - half its capacitance at
    #: each end - as constant reactive sources ``omega C / 2 V^2`` at the slack
    #: voltage (:attr:`charging_q_mvar`).
    line_charging: str = "omit"

    def __post_init__(self) -> None:
        """Normalise numeric data once, apply the load scale, reject bad cases."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"case name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.description, str):
            raise ValueError(f"description must be a string, got {self.description!r}")
        if not isinstance(self.axis_presets, dict):
            raise TypeError(
                f"axis_presets must be a dict, got {type(self.axis_presets).__name__}"
            )
        for preset_key, preset_axes in self.axis_presets.items():
            if not isinstance(preset_key, str) or not preset_key.strip():
                raise ValueError(
                    f"axis preset key must be a non-empty string, got {preset_key!r}"
                )
            if isinstance(preset_axes, (str, bytes, bytearray)):
                raise TypeError(
                    f"axis preset {preset_key!r} must be a sequence of axes, not str or bytes"
                )
        # Own one complete graph, before scaling. A second independent net
        # copy during scaling would break metadata/controller back-references.
        inputs = copy_configuration({item.name: getattr(self, item.name)
                                     for item in fields(self) if item.init})
        for name, value in inputs.items():
            setattr(self, name, value)
        if (isinstance(self.load_scale, (bool, np.bool_))
                or not np.isfinite(self.load_scale)
                or self.load_scale < 0.0):
            raise ValueError(
                f"load_scale must be finite and non-negative, got {self.load_scale!r}"
            )
        if (np.asarray(self.base_p_mw).dtype == bool
                or np.asarray(self.base_q_mvar).dtype == bool):
            raise ValueError("base injection arrays must be numeric, not boolean")
        marker = self.net.get(_UNSCALED_KEY) if hasattr(self.net, "get") else None
        if self.load_scale != 1.0 or marker is not None:
            # Scale fixed-power pandapower tables (load, sgen, storage) to match
            # effective_base_p_mw / effective_base_q_mvar.  Original unscaled
            # values are stored under _UNSCALED_KEY alongside base_p / base_q so
            # dataclasses.replace rescales from the original baseline unless new
            # base arrays were supplied.
            net = self.net
            net.pop(_UNSCALED_KEY, None)
            p_now = np.asarray(self.base_p_mw, dtype=float)
            q_now = np.asarray(self.base_q_mvar, dtype=float)
            if marker is not None and not (
                    np.shape(marker["base_p"]) == p_now.shape
                    and np.array_equal(marker["base_p"], p_now)
                    and np.array_equal(marker["base_q"], q_now)):
                marker = None
            if marker is not None:
                unscaled = marker["tables"]
            else:
                unscaled = {}
                for table in _SCALED_TABLES:
                    frame = getattr(net, table, None)
                    if frame is not None and len(frame):
                        columns = [c for c in ("p_mw", "q_mvar") if c in frame.columns]
                        unscaled[table] = frame[columns].copy()
            for table, values in unscaled.items():
                frame = getattr(net, table)
                for column in values.columns:
                    frame.loc[values.index, column] = values[column] * self.load_scale
            if self.load_scale != 1.0:
                net[_UNSCALED_KEY] = {
                    "tables": {table: values.copy() for table, values in unscaled.items()},
                    "base_p": p_now.copy(), "base_q": q_now.copy()}
            self.net = net

        self.base_p_mw = np.array(self.base_p_mw, dtype=float, copy=True)
        self.base_q_mvar = np.array(self.base_q_mvar, dtype=float, copy=True)
        if getattr(self.net, "bus", None) is None or len(self.net.bus) == 0:
            raise ValueError(f"case {self.name!r} has no buses")
        if not np.array_equal(self.net.bus.index.to_numpy(), np.arange(self.n_bus)):
            raise ValueError(
                f"case {self.name!r} bus index must be contiguous 0..{self.n_bus - 1}; "
                "use case_from_pandapower(net, ...) to renumber"
            )
        expected = (self.n_bus,)
        if self.base_p_mw.shape != expected or self.base_q_mvar.shape != expected:
            raise ValueError(
                f"base injection arrays must both have shape {expected}; "
                f"got {self.base_p_mw.shape} and {self.base_q_mvar.shape}"
            )
        if not np.all(np.isfinite(self.base_p_mw)) or not np.all(np.isfinite(self.base_q_mvar)):
            raise ValueError("base injection arrays must contain only finite values")
        if (isinstance(self.v_nom_kv, (bool, np.bool_))
                or not np.isfinite(self.v_nom_kv)
                or self.v_nom_kv <= 0.0):
            raise ValueError("v_nom_kv must be finite and strictly positive")
        if (isinstance(self.vm_min_pu, (bool, np.bool_))
                or isinstance(self.vm_max_pu, (bool, np.bool_))
                or not (np.isfinite(self.vm_min_pu) and np.isfinite(self.vm_max_pu)
                        and 0.0 < self.vm_min_pu < self.vm_max_pu)):
            raise ValueError("voltage limits must be finite and satisfy 0 < min < max")
        if isinstance(self.slack_vm_pu, (bool, np.bool_)) or not (
                np.isfinite(self.slack_vm_pu) and self.slack_vm_pu > 0.0):
            raise ValueError("slack_vm_pu must be finite and strictly positive")
        self.slack_vm_pu = float(self.slack_vm_pu)
        if self.line_charging not in LINE_CHARGING:
            raise ValueError(f"line_charging must be one of {LINE_CHARGING}, "
                             f"got {self.line_charging!r}")
        if self.slack_bus_override is not None:
            if (isinstance(self.slack_bus_override, (bool, np.bool_))
                    or not isinstance(self.slack_bus_override, (int, np.integer))):
                raise ValueError(
                    f"slack_bus_override must be an integer bus index, got {self.slack_bus_override!r}"
                )
            bus = int(self.slack_bus_override)
            if not 0 <= bus < self.n_bus:
                raise ValueError(
                    f"slack_bus_override={bus} is out of range for case {self.name!r} "
                    f"(0..{self.n_bus - 1})"
                )
            self.slack_bus_override = bus
        if hasattr(self.net, "get") and "_openrc_case_vm_limits" not in self.net:
            self.net["_openrc_case_vm_limits"] = (float(self.vm_min_pu), float(self.vm_max_pu))
        self._configuration_id = self._configuration_digest()

    def _configuration_digest(self) -> str:
        """Fingerprint physical inputs, excluding pandapower's result state."""
        net = {key: value for key, value in self.net.items()
               if (not key.startswith(("_", "res_")) or key.startswith("_openrc_"))
               and key not in ("converged", "OPF_converged")}
        # Include future/subclass dataclass inputs automatically. Descriptive
        # metadata and worker recipes are not physical model parameters.
        inputs = {item.name: getattr(self, item.name) for item in fields(self)
                  if item.init and item.name not in
                  {"net", "metadata", "description", "factory_kwargs", "registry_name"}}
        inputs["net"] = net
        digest = hashlib.sha256()

        def feed(token: Any) -> None:
            # Pickle leaf values individually rather than the whole tuple tree so
            # object memoization ids do not affect the digest across copies.
            if isinstance(token, tuple):
                digest.update(b"T" + len(token).to_bytes(8, "big"))
                for child in token:
                    feed(child)
            else:
                data = pickle.dumps(token, protocol=4)
                digest.update(b"V" + len(data).to_bytes(8, "big") + data)

        feed(_configuration_token(inputs, network=self.net))
        return digest.hexdigest()

    @property
    def configuration_id(self) -> str:
        """SHA-256 hex digest of the case's physical configuration inputs at construction."""
        return self._configuration_id

    def assert_unchanged(self) -> None:
        """Reject stale caches after an in-place configuration edit.

        Runs in time linear in the network table size and is intended for task
        initialization.  Running pandapower may update results/internals without
        changing a case; its public input tables and OpenRC modelling markers
        must remain unchanged.
        """
        if self._configuration_digest() != self._configuration_id:
            raise ValueError(
                f"case {self.name!r} configuration changed after construction; cached "
                "data may be stale. Use original_case.rebuild(load_scale=..., "
                "vm_min_pu=...) before editing it, or edit a private net and use "
                "case_from_pandapower(net, ...) to rebuild injections and topology. "
                "Existing oracles retain their original configuration snapshot."
            )

    def snapshot(self) -> GridCase:
        """Return an isolated, validated case without copying derived caches."""
        self.assert_unchanged()
        # Reconstruct only dataclass inputs: topology/base caches may be large
        # and must never carry over from a previously used case.
        result = replace(self)
        if result.configuration_id != self.configuration_id:
            raise ValueError("case changed while creating its snapshot; retry without concurrent edits")
        return result

    def rebuild(self, **changes: Any) -> GridCase:
        """Derive an isolated configuration and discard all derived caches.

        ``lighter = case.rebuild(load_scale=0.4)`` rescales from recorded
        injections, including recovery from zero load. Net and base-vector
        changes belong in ``case_from_pandapower`` instead, so this helper
        cannot change the two representations independently. Registry recipes
        are updated for factory-supported knobs; non-reproducible variants use
        the safe serial fallback.
        """
        if {"net", "base_p_mw", "base_q_mvar"}.intersection(changes):
            raise ValueError("use case_from_pandapower(net, ...) to rebuild network/base injections")
        self.assert_unchanged()
        recipe_knobs = {"load_scale", "line_charging", "axis_presets", "description"}
        if (self.registry_name and self.registry_name == self.name
                and "factory_kwargs" not in changes
                and set(changes) <= recipe_knobs):
            changes = dict(changes, factory_kwargs=dict(self.factory_kwargs, **changes))
        return replace(self, **changes)

    # ------------------------------------------------------------- basic info
    @property
    def n_bus(self) -> int:
        return int(len(self.net.bus))

    @property
    def n_line(self) -> int:
        return int(len(self.net.line))

    @cached_property
    def _slack(self) -> tuple[int, str]:
        """``(bus, provenance)``, inferred via :func:`~OpenRC.powerflow.topology.infer_slack_bus`
        unless :attr:`slack_bus_override` is set."""
        from OpenRC.powerflow.topology import infer_slack_bus

        if self.slack_bus_override is not None:
            if (isinstance(self.slack_bus_override, (bool, np.bool_))
                    or not isinstance(self.slack_bus_override, (int, np.integer))):
                raise ValueError(
                    f"slack_bus_override must be an integer bus index, got {self.slack_bus_override!r}"
                )
            bus = int(self.slack_bus_override)
            if not 0 <= bus < self.n_bus:
                raise ValueError(
                    f"slack_bus_override={bus} is out of range for case {self.name!r} "
                    f"(0..{self.n_bus - 1})"
                )
            return bus, "explicit slack_bus_override"
        try:
            return infer_slack_bus(self.net)
        except ValueError as exc:
            raise type(exc)(f"case {self.name!r}: {exc}") from None

    @property
    def slack_bus(self) -> int:
        """Index of the infeed bus, detected automatically."""
        return self._slack[0]

    @property
    def slack_source(self) -> str:
        """How :attr:`slack_bus` was decided, for provenance and debugging."""
        return self._slack[1]

    @cached_property
    def infeed_buses(self) -> set[int]:
        """All buses that act as voltage-angle references.

        A radial DistFlow model needs one root and therefore uses
        :attr:`slack_bus`.  AC/DC power flow and OPF, however, legitimately
        support several external grids.  Axis validation must exclude every
        such infeed without requiring the network to have a unique root.
        """
        buses: set[int] = set()
        ext = getattr(self.net, "ext_grid", None)
        if ext is not None and len(ext):
            live = (ext[ext["in_service"].astype(bool)]
                    if "in_service" in ext.columns else ext)
            buses.update(int(bus) for bus in live.bus)
        gen = getattr(self.net, "gen", None)
        if gen is not None and len(gen) and "slack" in gen.columns:
            live = (gen[gen["in_service"].astype(bool)]
                    if "in_service" in gen.columns else gen)
            buses.update(int(bus) for bus in live.loc[live["slack"].astype(bool), "bus"])
        if self.slack_bus_override is not None:
            buses.add(self.slack_bus)
        if buses:
            return buses
        return {self.slack_bus}

    @cached_property
    def radiality(self):
        """:class:`~OpenRC.powerflow.topology.RadialityReport` for this network.

        Evaluated from graph connectivity and element tables without requiring a
        unique slack bus.
        """
        from OpenRC.powerflow.topology import radiality_report

        return radiality_report(self.net, line_charging=self.line_charging)

    @property
    def is_radial(self) -> bool:
        """``True`` when the DistFlow oracles can be used on this network."""
        return self.radiality.is_radial

    @cached_property
    def non_slack_buses(self) -> List[int]:
        """Every bus that is not an infeed, ascending.

        Computed from :attr:`infeed_buses` without building :attr:`topology`,
        supporting meshed networks and multi-infeed AC/DC/OPF cases.
        """
        return [b for b in range(self.n_bus) if b not in self.infeed_buses]

    @cached_property
    def topology(self):
        """Lazily built :class:`~OpenRC.powerflow.topology.RadialTopology`."""
        from OpenRC.powerflow.topology import RadialTopology

        return RadialTopology.from_case(self)

    # -------------------------------------------------------------- base load
    @cached_property
    def effective_base_p_mw(self) -> np.ndarray:
        """Scaled per-bus active base load ``base_p_mw * load_scale`` (MW)."""
        return self.base_p_mw.astype(float) * self.load_scale

    @cached_property
    def effective_base_q_mvar(self) -> np.ndarray:
        """Scaled per-bus reactive base load ``base_q_mvar * load_scale`` (Mvar)."""
        return self.base_q_mvar.astype(float) * self.load_scale

    # ------------------------------------------------------------------- axes
    def resolve_axes(self, spec: Sequence[AxisSpec] | str) -> RegionAxes:
        """Turn a preset name or an explicit axis list into :class:`RegionAxes`.

        All of these are equivalent for the 9-bus case::

            case.resolve_axes("247")
            case.resolve_axes([1, 3, 6])
            case.resolve_axes(["1p", "3p", "6p"])

        which is what lets the CLI accept either ``--axes 247`` or
        ``--axes 1,3,6`` or a mixed ``--axes 1p,1q,3p``.
        """
        if isinstance(spec, (bytes, bytearray)):
            raise TypeError("axis specification must not be bytes")
        if isinstance(spec, str):
            if spec in self.axis_presets:
                axes = RegionAxes(self.axis_presets[spec])
            elif "," in spec or _AXIS_PATTERN.match(spec.strip()):
                tokens = [s.strip() for s in spec.split(",")]
                if any(not t for t in tokens):
                    raise ValueError(f"cannot parse axis list {spec!r}: empty axis entry")
                axes = RegionAxes(tokens)
            else:
                raise KeyError(
                    f"case {self.name!r} has no axis preset {spec!r}; available: "
                    f"{', '.join(sorted(self.axis_presets)) or '(none)'}"
                )
        else:
            axes = RegionAxes(list(spec))

        for axis in axes:
            if not 0 <= axis.bus < self.n_bus:
                hint = ""
                if isinstance(spec, str) and len(spec.strip()) > 1 and spec.strip().isdigit():
                    # "247" is a preset on one case and a bus number on another.
                    hint = (f"; {spec!r} is not a preset of this case, so it was read as one "
                            f"bus number (presets: {', '.join(sorted(self.axis_presets)) or 'none'})")
                raise ValueError(f"bus {axis.bus} out of range for case {self.name!r}{hint}")
            if axis.bus in self.infeed_buses:
                raise ValueError(
                    f"bus {axis.bus} is a slack/infeed bus of {self.name!r} and cannot "
                    "be an axis"
                )
        return axes

    # ---------------------------------------------------------------- vectors
    def injection_vector(
        self,
        axes: RegionAxes,
        delta: Optional[ArrayLike] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Base injection plus a region-space offset, as full per-bus arrays.

        Parameters
        ----------
        axes:
            The region axes, one per entry of ``delta``.
        delta:
            Additional injection in MW / Mvar along each axis (``None`` = zero).

        Returns
        -------
        ``(p_mw, q_mvar)``, each of length ``n_bus``, ready for an oracle.
        """
        p = self.effective_base_p_mw.copy()
        q = self.effective_base_q_mvar.copy()
        if delta is None:
            return p, q

        raw = np.asarray(delta)
        if raw.dtype == bool:
            raise ValueError("region coordinates must be a numeric vector, not boolean")
        delta = np.asarray(raw, dtype=float)
        if delta.shape != (len(axes),):
            raise ValueError(
                f"delta has shape {delta.shape}, expected ({len(axes)},) to match the axes"
            )
        if not np.all(np.isfinite(delta)):
            raise ValueError("region coordinates must contain only finite values")
        for axis, value in zip(axes, delta):
            if axis.quantity == "p":
                p[axis.bus] += value
            else:
                q[axis.bus] += value
        return p, q

    # ------------------------------------------------------------------ repr
    @cached_property
    def charging_q_mvar(self) -> np.ndarray:
        """Reactive power (Mvar, per bus) the lines' shunt capacitance supplies.

        The pi model of every in-service line puts half its capacitance ``C``
        at each end; at the slack voltage ``V = slack_vm_pu * v_nom_kv`` that
        half supplies ``omega C / 2 * V**2`` (kV**2 x S = Mvar).  The
        branch-flow models subtract it from the reactive load only with
        ``line_charging="constant"``; AC power flow models the lines itself.
        """
        out = np.zeros(self.n_bus)
        lines = getattr(self.net, "line", None)
        if lines is None or len(lines) == 0 or "c_nf_per_km" not in lines.columns:
            return out
        if "in_service" in lines.columns:
            lines = lines[lines["in_service"].astype(bool)]
        length = lines.length_km.to_numpy(dtype=float)
        parallel = (lines["parallel"].to_numpy(dtype=float) if "parallel" in lines.columns
                    else np.ones(len(lines)))
        parallel = np.where(np.isfinite(parallel) & (parallel > 0.0), parallel, 1.0)
        farad = lines.c_nf_per_km.to_numpy(dtype=float) * 1e-9 * length * parallel
        farad = np.where(np.isfinite(farad) & (farad > 0.0), farad, 0.0)
        f_hz = float(self.net.get("f_hz", 50.0) or 50.0) if hasattr(self.net, "get") else 50.0
        f_hz = f_hz if np.isfinite(f_hz) and f_hz > 0.0 else 50.0
        vn_kv = self.net.bus["vn_kv"].to_numpy(dtype=float)
        from_buses = lines["from_bus"].to_numpy(dtype=int)
        to_buses = lines["to_bus"].to_numpy(dtype=int)
        valid_from = (from_buses >= 0) & (from_buses < self.n_bus)
        line_kv = np.where(valid_from, vn_kv[np.clip(from_buses, 0, self.n_bus - 1)], self.v_nom_kv)
        volts = line_kv * self.slack_vm_pu
        half = np.pi * f_hz * farad * volts * volts           # omega C / 2 * V**2

        switches = getattr(self.net, "switch", None)
        opened: Dict[int, set[int]] = {}
        if switches is not None and len(switches):
            open_l = switches[(switches.et == "l") & ~switches.closed.astype(bool)]
            for sw_el, sw_bus in zip(open_l.element.to_numpy(dtype=int),
                                     open_l.bus.to_numpy(dtype=int)):
                opened.setdefault(int(sw_el), set()).add(int(sw_bus))

        for line_idx, f_bus, t_bus, h_q in zip(lines.index.to_numpy(dtype=int),
                                               from_buses, to_buses, half):
            if h_q == 0.0:
                continue
            open_ends = opened.get(int(line_idx))
            if not open_ends:
                if 0 <= f_bus < self.n_bus:
                    out[f_bus] += h_q
                if 0 <= t_bus < self.n_bus:
                    out[t_bus] += h_q
                continue
            f_closed = f_bus not in open_ends
            t_closed = t_bus not in open_ends
            if f_closed and t_closed:
                if 0 <= f_bus < self.n_bus:
                    out[f_bus] += h_q
                if 0 <= t_bus < self.n_bus:
                    out[t_bus] += h_q
            elif f_closed and 0 <= f_bus < self.n_bus:
                out[f_bus] += 2.0 * h_q
            elif t_closed and 0 <= t_bus < self.n_bus:
                out[t_bus] += 2.0 * h_q
        return out

    def branch_flow_q(self, q_mvar):
        """``q_mvar`` adjusted for branch-flow models.

        Subtracts :attr:`charging_q_mvar` when ``line_charging="constant"``;
        returns ``q_mvar`` unchanged when ``line_charging="omit"``.  Supports
        1-D and 2-D arrays.
        """
        raw = np.asarray(q_mvar)
        if raw.dtype == bool:
            raise ValueError("q_mvar must be a numeric array, not boolean")
        if self.line_charging == "omit":
            return q_mvar
        return np.asarray(raw, dtype=float) - self.charging_q_mvar

    def summary(self) -> str:
        presets = ", ".join(f"{k}={v}" for k, v in sorted(self.axis_presets.items()))
        flag = (f"  [load_scale={self.load_scale:g}]" if self.load_scale != 1.0 else "")
        raw = ""
        if self.load_scale != 1.0:
            raw = (f"    recorded load {self.base_p_mw.sum():+.2f} MW / "
                   f"{self.base_q_mvar.sum():+.2f} Mvar before scaling\n")
        try:
            slack = f"slack at bus {self.slack_bus} (by {self.slack_source})"
        except ValueError as exc:
            slack = f"slack undetermined: {exc}"
        shape = "radial" if self.is_radial else "NOT radial"
        return (
            f"{self.name}: {self.n_bus} buses, {self.n_line} lines, {self.v_nom_kv} kV, "
            f"{shape}, {slack}{flag}\n"
            f"    effective base load {self.effective_base_p_mw.sum():+.2f} MW / "
            f"{self.effective_base_q_mvar.sum():+.2f} Mvar\n"
            f"{raw}"
            f"    voltage envelope [{self.vm_min_pu}, {self.vm_max_pu}] p.u.\n"
            f"{self._charging_line()}"
            f"    axis presets: {presets or '(none)'}\n"
            f"    {self.description}"
        )

    def _charging_line(self) -> str:
        total = float(self.charging_q_mvar.sum())
        if total <= 0.0:
            return ""
        use = ("constant reactive sources in the branch-flow models"
               if self.line_charging == "constant" else
               "omitted by the branch-flow models; line_charging='constant' adds it")
        return f"    line charging {total:.3f} Mvar at the slack voltage ({use})\n"

    def __repr__(self) -> str:
        return f"<GridCase {self.name!r} n_bus={self.n_bus} n_line={self.n_line}>"


def validate_case_configuration(case: Any, oracle: Any = None) -> None:
    """Check a task's case and, when supplied, its oracle configuration.

    Duck-typed extension cases/oracles keep their own configuration contract.
    Compares ``GridCase`` configurations by :attr:`GridCase.configuration_id`
    because oracles store isolated case snapshots.  Intended to be called once
    per task.
    """
    if isinstance(case, GridCase):
        case.assert_unchanged()
        bound = getattr(oracle, "case", None)
        if isinstance(bound, GridCase):
            if bound is not case:
                bound.assert_unchanged()
            if bound.configuration_id != case.configuration_id:
                raise ValueError("oracle belongs to a different case configuration; "
                                 "rebuild the oracle for this case")
