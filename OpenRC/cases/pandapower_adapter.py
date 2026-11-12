"""Convert ``pandapower`` networks into :class:`~OpenRC.cases.base.GridCase` instances.

Converts an arbitrary ``pandapowerNet`` (user-constructed, loaded from JSON, or
from ``pandapower.networks``) into a :class:`~OpenRC.cases.base.GridCase`::

    case = orc.case_from_pandapower(net, name="my-feeder")
    case = orc.get_case("pp:create_kerber_dorfnetz")     # built-in, by name

Inferred network properties
---------------------------
Three properties are inferred during conversion and recorded in
``case.metadata["conversion"]``:

**Where the infeed is.**  Handled by
:func:`~OpenRC.powerflow.topology.infer_slack_bus`; ``case.slack_source`` records
which rule matched.

**Whether the DistFlow family applies.**  Handled by
:func:`~OpenRC.powerflow.topology.radiality_report`; ``case.is_radial`` and
``case.radiality.message`` record eligibility, while meshed networks remain
usable with the AC/DC and OPF oracles.

**How transformers are represented.**  See below.

Single-voltage transformer collapse
-----------------------------------
The branch-flow models in OpenRC are formulated in physical units against a
single nominal voltage.  To express a multi-voltage radial feeder on a single
voltage base, branches and two-winding transformers are converted to per-unit
and referred to :math:`V_{ref}`.

For a branch with per-unit impedance :math:`z_{pu}` on base :math:`S_{base}`, the
equivalent impedance at reference voltage :math:`V_{ref}` is
:math:`z_{pu} \\, V_{ref}^2 / S_{base}`, and a rating :math:`s^{max}_{pu}` becomes
a current limit :math:`s^{max}_{pu} S_{base} / (\\sqrt{3} V_{ref})`.  Applying
this transformation to lines and two-winding transformers yields an electrically
equivalent single-voltage network (matching the substation-line representation
used by the tabular 9-bus cases).

Off-nominal tap ratios, phase shifts, and no-load shunt branches are not
represented by a series impedance and are recorded in ``ConversionReport.warnings``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

from OpenRC.cases.base import LINE_CHARGING, GridCase
from OpenRC.utils import LOGGER

#: How :func:`case_from_pandapower` may treat two-winding transformers.
TRANSFORMER_HANDLING = ("auto", "collapse", "keep")

# Older pandapower topology utilities omit these native FACTS/converter
# references.  DC bus IDs belong to another namespace and must not be remapped.
_NATIVE_AC_BUS_COLUMNS = {
    "svc": ("bus",), "ssc": ("bus",),
    "vsc": ("bus", "ref_bus"),
    "vsc_stacked": ("bus",), "vsc_bipolar": ("bus",), "b2b_vsc": ("bus",),
    "tcsc": ("from_bus", "to_bus"),
}


@dataclass
class ConversionReport:
    """What the adapter did, and what it had to assume.

    Stored on ``case.metadata["conversion"]`` so a result can always be traced
    back to the modelling decisions behind it.
    """

    n_bus: int = 0
    n_line: int = 0
    n_trafo_collapsed: int = 0
    n_trafo_kept: int = 0
    reference_kv: Optional[float] = None
    voltage_levels: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dropped_out_of_service: Dict[str, int] = field(default_factory=dict)
    switches_applied: Dict[str, int] = field(default_factory=dict)
    #: Lines whose rating was a placeholder (see
    #: :func:`OpenRC.powerflow.topology.unrated_lines`), and how many of them
    #: were given ``default_rating_ka``.
    n_unrated: int = 0
    n_rating_filled: int = 0
    #: Voltage the external grid holds (``None``: none, or several differing),
    #: and the slack voltage the branch-flow models were given.
    slack_setpoint_pu: Optional[float] = None
    slack_vm_pu: float = 1.0
    #: ``{original bus index: OpenRC bus index}``.  OpenRC numbers buses
    #: contiguously from 0; a model that does not (custom IDs, fused buses) is
    #: renumbered, and this is how to translate back.
    bus_index_map: Dict[int, int] = field(default_factory=dict)

    @property
    def collapsed(self) -> bool:
        return self.n_trafo_collapsed > 0

    def summary(self) -> str:
        bits = [f"{self.n_bus} buses", f"{self.n_line} lines"]
        if self.n_trafo_collapsed:
            bits.append(f"{self.n_trafo_collapsed} transformer(s) collapsed to "
                        f"{self.reference_kv:g} kV")
        if self.n_trafo_kept:
            bits.append(f"{self.n_trafo_kept} transformer(s) kept")
        if self.switches_applied:
            bits.append("switches applied (" + ", ".join(
                f"{v} {k}" for k, v in sorted(self.switches_applied.items()) if v) + ")")
        text = ", ".join(bits)
        if self.warnings:
            text += "\n    " + "\n    ".join(self.warnings)
        return text

    def as_dict(self) -> dict:
        return {
            "n_bus": self.n_bus, "n_line": self.n_line,
            "n_trafo_collapsed": self.n_trafo_collapsed,
            "n_trafo_kept": self.n_trafo_kept,
            "reference_kv": self.reference_kv,
            "voltage_levels": self.voltage_levels,
            "warnings": list(self.warnings),
            "dropped_out_of_service": dict(self.dropped_out_of_service),
            "switches_applied": dict(self.switches_applied),
            "n_unrated": self.n_unrated,
            "n_rating_filled": self.n_rating_filled,
            "slack_setpoint_pu": self.slack_setpoint_pu,
            "slack_vm_pu": self.slack_vm_pu,
            "bus_index_map": dict(self.bus_index_map),
            "renumbered": any(k != v for k, v in self.bus_index_map.items()),
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _finite_or(value, default: float) -> float:
    """Return ``float(value)`` when finite, else ``default`` (guards against pandas ``NaN``)."""
    if value is None:
        return default
    val = float(value)
    return val if math.isfinite(val) else default


def _branch_impedance_pu(row, s_base: float) -> Tuple[float, float, float]:
    """Per-unit ``(r, x, s_max)`` of a two-winding transformer."""
    sn = float(row.sn_mva)
    if not np.isfinite(sn) or sn <= 0:
        raise ValueError("transformer has no usable sn_mva")
    vk = float(row.vk_percent)
    vkr = float(row.vkr_percent)
    if not np.isfinite(vk) or vk <= 0.0:
        raise ValueError("transformer has no usable vk_percent")
    if not np.isfinite(vkr) or vkr < 0.0 or vkr > vk * (1.0 + 1e-9):
        raise ValueError("transformer has no usable vkr_percent")
    parallel = _finite_or(getattr(row, "parallel", 1.0), 1.0)
    if parallel <= 0.0:
        parallel = 1.0
    # vk/vkr are given on the transformer's own rating, so rebase to s_base.
    z_pu = (vk / 100.0) * (s_base / sn) / parallel
    r_pu = (vkr / 100.0) * (s_base / sn) / parallel
    x_pu = math.sqrt(max(z_pu ** 2 - r_pu ** 2, 0.0))
    max_loading = _finite_or(getattr(row, "max_loading_percent", 100.0), 100.0)
    if max_loading <= 0.0:
        raise ValueError("transformer has non-positive max_loading_percent")
    derating = _finite_or(getattr(row, "df", 1.0), 1.0)
    if derating <= 0.0:
        derating = 1.0
    # Nominal 100 % apparent-power rating on s_base; per-element
    # max_loading_percent is copied to the collapsed line so neither AC/OPF nor
    # RadialTopology applies it twice.
    s_max_pu = sn * parallel * derating / s_base
    return r_pu, x_pu, s_max_pu


def _line_impedance_pu(row, v_base_kv: float, s_base: float) -> Tuple[float, float, float]:
    """Per-unit ``(r, x, s_max)`` of a line at its own voltage base."""
    z_base = v_base_kv ** 2 / s_base
    parallel = _finite_or(getattr(row, "parallel", 1.0), 1.0)
    if parallel <= 0.0:
        parallel = 1.0
    length = float(row.length_km)
    r_pu = float(row.r_ohm_per_km) * length / z_base / parallel
    x_pu = float(row.x_ohm_per_km) * length / z_base / parallel
    # sqrt(3) * kV * kA = MVA
    derating = _finite_or(getattr(row, "df", 1.0), 1.0)
    if derating <= 0.0:
        derating = 1.0
    s_max_pu = (math.sqrt(3.0) * v_base_kv * float(row.max_i_ka)
                * parallel * derating / s_base)
    return r_pu, x_pu, s_max_pu


def _pick_reference_kv(net) -> float:
    """Reference voltage for the collapsed network.

    The level carrying the most buses is chosen: on a feeder that is the
    distribution level, so the collapsed network keeps the voltages an operator
    would recognise, and the single upstream bus is the one that moves.
    """
    counts = net.bus.vn_kv.value_counts()
    return float(counts.idxmax())


def _voltage_envelope(
    net, default_min: float, default_max: float
) -> Tuple[float, float, bool]:
    """Return the strictest common voltage envelope and whether it exists.

    ``pandapower`` fills unset limits with permissive defaults (0.0 / 2.0), which
    are not an operating envelope; those are ignored.  ``GridCase`` stores one
    pair of scalar limits for the DistFlow family, so using the strictest lower
    and upper limits is conservative when buses have different envelopes.  AC
    power-flow validation keeps each bus's native pandapower limits.
    """
    lo, hi = default_min, default_max
    # A voltage-controlled infeed normally has min == max == its setpoint.  It
    # is not a DistFlow state variable and must not collapse the operating
    # envelope of downstream buses (case33bw's slack is exactly 1.0 pu).
    infeed_buses: set[int] = set()
    ext_grid = getattr(net, "ext_grid", None)
    if ext_grid is not None and len(ext_grid):
        live = (ext_grid[ext_grid["in_service"].astype(bool)]
                if "in_service" in ext_grid.columns else ext_grid)
        infeed_buses.update(int(bus) for bus in live.bus)
    gen = getattr(net, "gen", None)
    if gen is not None and len(gen) and "slack" in gen.columns:
        live = gen[gen["in_service"].astype(bool)] \
            if "in_service" in gen.columns else gen
        infeed_buses.update(int(bus) for bus in live.loc[live["slack"].astype(bool), "bus"])
    buses = net.bus.drop(index=list(infeed_buses), errors="ignore")

    if "min_vm_pu" in buses.columns:
        values = buses.min_vm_pu.to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            lo = float(np.max(values))
    if "max_vm_pu" in buses.columns:
        values = buses.max_vm_pu.to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values < 2.0)]
        if values.size:
            hi = float(np.min(values))
    if not lo < hi:
        return default_min, default_max, False
    return lo, hi, True


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------
def case_from_pandapower(
    net,
    *,
    name: str = "pandapower",
    axis_presets: Optional[Dict[str, Sequence]] = None,
    description: str = "",
    load_scale: float = 1.0,
    transformer_handling: str = "auto",
    s_base_mva: float = 1.0,
    reference_kv: Optional[float] = None,
    vm_min_pu: float = 0.9,
    vm_max_pu: float = 1.1,
    feeder: Optional[int] = None,
    default_rating_ka: Optional[float] = None,
    slack_vm_pu: Optional[Union[float, str]] = None,
    line_charging: str = "omit",
    copy: bool = True,
) -> GridCase:
    """Build a :class:`GridCase` from any ``pandapower`` network.

    Parameters
    ----------
    net:
        The network.  Copied by default, so the caller's object is untouched.
    name:
        Case name, used in provenance and in filenames.
    axis_presets:
        Named POC sets in the network's **own** bus IDs, e.g.
        ``{"three": [12, 40, 77]}``.  They are translated to OpenRC's contiguous
        numbering automatically; ``case.metadata["conversion"]`` records the map.
    load_scale:
        Proportional scaling of the recorded load; see :class:`GridCase`.
    transformer_handling:
        ``"auto"`` (default)
            Collapse two-winding transformers into equivalent branches when the
            network is otherwise a single-infeed radial tree, enabling DistFlow
            oracles.  Retain native transformers when the network is meshed or
            contains other non-DistFlow elements.
        ``"collapse"``
            Always collapse.
        ``"keep"``
            Keep native transformers (supported by AC/DC and OPF oracles;
            incompatible with DistFlow).
    s_base_mva, reference_kv:
        Per-unit base for the collapse, and the voltage the result is expressed
        at.  ``reference_kv=None`` picks the level carrying the most buses.
    vm_min_pu, vm_max_pu:
        Fallback voltage envelope, used when the network does not state one.
    feeder:
        Keep only the ``feeder``-th independently fed part of the network.  A
        network with several infeeds - ``mv_oberrhein`` has two substations,
        ``lv_schutterwald`` fourteen - may contain several fed components.
        Each selected part is checked again for DistFlow eligibility. Parts are
        numbered by their infeed bus, ascending; :func:`list_feeders` shows them.
        ``None`` (default) keeps the whole network, which the AC/DC and OPF
        oracles can use as it is.  Extraction refuses an HVDC link, converter
        DC network, or remote voltage-control reference crossing the selected
        AC part's boundary; these require the full coupled network.
    default_rating_ka:
        Thermal rating given to every line whose rating is a placeholder.
        Transmission test cases derived from MATPOWER usually carry none -
        ``case33bw`` has ``max_i_ka = 99999`` on every line.  Such a line is
        treated as unconstrained, which Model-I, SOCP and the AC oracles handle
        correctly, but Model-II and both Method-2 models bound losses by the
        rating and cannot be used without one.  When ``None`` (default),
        placeholder ratings are left unmodified and marked unrated.
    slack_vm_pu:
        Voltage, per unit, at which the branch-flow models (DistFlow, SOCP,
        exact, the sweep and the certificate) hold the infeed.  ``None``
        (default) keeps 1.0 p.u.; ``"setpoint"`` takes the external grid's
        ``vm_pu``, the voltage the AC/DC oracles use; a number sets it.  When
        the setpoint differs from the value used, the conversion report warns:
        the two oracle families then describe different operating points.
    line_charging:
        How branch-flow models treat line shunt capacitance: ``"omit"``
        (default) or ``"constant"``; see :attr:`GridCase.line_charging`.

    Returns
    -------
    GridCase

    Notes
    -----
    Collapsing accepts the standard per-unit idealisation of a transformer: an
    off-nominal tap position or a phase shift is not represented.  Both are
    detected and recorded in ``case.metadata["conversion"]["warnings"]``.
    """
    import copy as _copy

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"case name must be a non-empty string, got {name!r}")
    if not isinstance(description, str):
        raise ValueError(f"description must be a string, got {description!r}")
    if (isinstance(load_scale, (bool, np.bool_))
            or not np.isfinite(load_scale)
            or load_scale < 0.0):
        raise ValueError(f"load_scale must be finite and non-negative, got {load_scale!r}")
    if (isinstance(vm_min_pu, (bool, np.bool_))
            or isinstance(vm_max_pu, (bool, np.bool_))
            or not (np.isfinite(vm_min_pu) and np.isfinite(vm_max_pu)
                    and 0.0 < vm_min_pu < vm_max_pu)):
        raise ValueError("voltage limits must be finite and satisfy 0 < min < max")
    if transformer_handling not in TRANSFORMER_HANDLING:
        raise ValueError(
            f"transformer_handling must be one of {TRANSFORMER_HANDLING}, "
            f"got {transformer_handling!r}"
        )
    if line_charging not in LINE_CHARGING:
        raise ValueError(
            f"line_charging must be one of {LINE_CHARGING}, got {line_charging!r}"
        )
    if (isinstance(s_base_mva, (bool, np.bool_))
            or not np.isfinite(s_base_mva)
            or s_base_mva <= 0.0):
        raise ValueError(f"s_base_mva must be finite and strictly positive, got {s_base_mva!r}")
    if reference_kv is not None and (
            isinstance(reference_kv, (bool, np.bool_))
            or not np.isfinite(reference_kv)
            or reference_kv <= 0.0):
        raise ValueError(f"reference_kv must be finite and strictly positive, got {reference_kv!r}")
    if default_rating_ka is not None and (
            isinstance(default_rating_ka, (bool, np.bool_))
            or not np.isfinite(default_rating_ka)
            or default_rating_ka <= 0.0):
        raise ValueError(
            f"default_rating_ka must be finite and positive, got {default_rating_ka!r}"
        )
    if not isinstance(copy, (bool, np.bool_)):
        raise ValueError(f"copy must be a boolean, got {copy!r}")
    if copy:
        net = _copy.deepcopy(net)

    if len(net.bus) == 0:
        raise ValueError(f"network {name!r} has no buses")

    report = ConversionReport()
    report.voltage_levels = sorted(float(v) for v in net.bus.vn_kv.unique())

    # ---- operating topology: switches, out-of-service elements, numbering --
    # Apply switch states before radiality analysis so open tie switches in
    # open-ring feeders are not counted as closed loops.
    _apply_switch_states(net, report)
    _drop_out_of_service_branches(net, report)
    _balance_asymmetric_elements(net, report)
    if feeder is not None:
        net = _select_feeder(net, feeder, report)
    _renumber_buses(net, report)

    # ---- decide whether to collapse transformers ---------------------------
    from OpenRC.powerflow.topology import radiality_report

    trafos = getattr(net, "trafo", None)
    n_trafo = 0 if trafos is None else len(trafos)

    collapse = transformer_handling == "collapse"
    before = None
    if transformer_handling == "auto" and n_trafo:
        before = radiality_report(net, line_charging=line_charging)
        _, _, common_vm = _voltage_envelope(net, vm_min_pu, vm_max_pu)
        # Collapsing turns a transformer into a line; it changes no connectivity,
        # so it only helps when the network is a single-infeed tree with no
        # other elements outside the DistFlow model.
        collapse = (before.is_connected and before.n_loops == 0
                    and before.n_infeeds <= 1
                    and set(before.unsupported) <= {"two-winding transformers"}
                    and common_vm)

    if collapse and n_trafo:
        v_ref = float(reference_kv) if reference_kv is not None else _pick_reference_kv(net)
        report.reference_kv = v_ref
        net = _collapse_to_single_voltage(net, v_ref, float(s_base_mva), report)
    else:
        report.n_trafo_kept = n_trafo
        if n_trafo and transformer_handling == "auto":
            not_tree = (before is not None and not (
                before.is_connected and before.n_loops == 0 and before.n_infeeds <= 1
            ))
            why = ("the network is not one single-infeed tree"
                   if not_tree else
                   "the network carries other elements outside the DistFlow model")
            report.warnings.append(
                f"{n_trafo} transformer(s) kept: {why}, so "
                "collapsing them would not make the DistFlow oracles applicable. "
                "Use the AC/DC or OPF oracles."
            )

    _handle_unrated_lines(net, report, default_rating_ka)

    # ---- base injection ----------------------------------------------------
    n_bus = len(net.bus)
    base_p = np.zeros(n_bus)
    base_q = np.zeros(n_bus)
    for table, sign in (("load", +1.0), ("sgen", -1.0), ("storage", +1.0)):
        frame = getattr(net, table, None)
        if frame is None or len(frame) == 0:
            continue
        live = frame[frame.in_service.astype(bool)] if "in_service" in frame.columns else frame
        scaling = live["scaling"].to_numpy(dtype=float) if "scaling" in live.columns \
            else np.ones(len(live))
        scaling = np.where(np.isfinite(scaling), scaling, 1.0)
        for row, scale in zip(live.itertuples(), scaling):
            bus = int(row.bus)
            base_p[bus] += sign * _finite_or(getattr(row, "p_mw", 0.0), 0.0) * scale
            base_q[bus] += sign * _finite_or(getattr(row, "q_mvar", 0.0), 0.0) * scale

    lo, hi, has_common_voltage_envelope = _voltage_envelope(
        net, vm_min_pu, vm_max_pu
    )
    if not has_common_voltage_envelope:
        # Record _openrc_distflow_unsupported so DistFlow rejects incompatible
        # per-bus voltage intervals while AC/DC/OPF retain native per-bus limits.
        net["_openrc_distflow_unsupported"] = {
            "incompatible bus voltage envelopes": 1
        }
        report.warnings.append(
            "bus-specific voltage bounds have no common interval; the scalar "
            f"GridCase envelope retains the requested defaults [{lo:g}, {hi:g}] "
            "pu and the DistFlow family is disabled. AC/DC/OPF keep the native "
            "per-bus limits."
        )
    report.n_bus = n_bus
    report.n_line = int(len(net.line))
    report.slack_setpoint_pu = _slack_setpoint(net)
    report.slack_vm_pu = _choose_slack_vm(slack_vm_pu, report)

    case = GridCase(
        name=name,
        net=net,
        base_p_mw=base_p,
        base_q_mvar=base_q,
        v_nom_kv=float(report.reference_kv or net.bus.vn_kv.max()),
        vm_min_pu=lo,
        vm_max_pu=hi,
        axis_presets=_translate_presets(axis_presets, report.bus_index_map),
        description=description or _short_description(report),
        load_scale=load_scale,
        metadata={"source": "pandapower", "conversion": report.as_dict()},
        slack_vm_pu=report.slack_vm_pu,
        line_charging=line_charging,
    )
    for message in report.warnings:
        LOGGER.info("%s: %s", name, message)
    return case


def _slack_setpoint(net) -> Optional[float]:
    """The voltage magnitude the infeed holds, when the network states one."""
    buses: set[int] = set()
    values: set[float] = set()
    for table, column in (("ext_grid", None), ("gen", "slack")):
        frame = getattr(net, table, None)
        if frame is None or len(frame) == 0 or "vm_pu" not in frame.columns:
            continue
        live = frame[frame["in_service"].astype(bool)] if "in_service" in frame.columns else frame
        if column is not None:
            if column not in live.columns:
                continue
            live = live[live[column].astype(bool)]
        if len(live) == 0:
            continue
        for bus, vm in zip(live["bus"], live["vm_pu"]):
            value = float(vm)
            if not (np.isfinite(value) and value > 0.0):
                return None
            buses.add(int(bus))
            values.add(value)
    if len(buses) == 1 and len(values) == 1:
        return values.pop()
    return None


def _choose_slack_vm(option, report: ConversionReport) -> float:
    """Resolve ``slack_vm_pu`` and warn when the families would disagree."""
    setpoint = report.slack_setpoint_pu
    if option is None:
        chosen = 1.0
    elif isinstance(option, str):
        if option != "setpoint":
            raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")
        chosen = 1.0 if setpoint is None else setpoint
        if setpoint is None:
            report.warnings.append("slack_vm_pu='setpoint' but the network states no single "
                                   "infeed voltage; the branch-flow models use 1.0 pu")
    else:
        chosen = float(option)
        if isinstance(option, (bool, np.bool_)) or not (np.isfinite(chosen) and chosen > 0.0):
            raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")
    if setpoint is not None and abs(setpoint - chosen) > 1e-9:
        report.warnings.append(
            f"the external grid holds {setpoint:g} pu but the branch-flow models (DistFlow, "
            f"SOCP, exact, the certificate) hold the infeed at {chosen:g} pu, so they and the "
            "AC/DC oracles describe different operating points; pass "
            "slack_vm_pu='setpoint' to make them agree")
    return chosen


def _collapse_to_single_voltage(net, v_ref: float, s_base: float,
                                report: ConversionReport):
    """Replace every transformer with an equivalent line at ``v_ref``.

    Lines already at another voltage level are rebased too, so the whole network
    ends up electrically equivalent at one voltage - which is the form the
    branch-flow models are written for.
    """
    import pandapower as pp

    from OpenRC.powerflow.topology import UNRATED_MVA_THRESHOLD

    z_ref = v_ref ** 2 / s_base

    switches = getattr(net, "switch", None)
    if switches is not None and len(switches):
        opened = switches[(switches.et == "t") & ~switches.closed.astype(bool)]
        if len(opened):
            report.n_trafo_kept += len(net.trafo)
            report.reference_kv = None
            report.warnings.append(
                "transformer collapse skipped: an open terminal switch leaves "
                "the connected winding's native no-load branch energised. "
                "AC/DC/OPF retain the transformer and its switch."
            )
            return net

    # Validate all transformer parameters before mutating net so invalid data
    # aborts collapse atomically.
    converted = {}
    invalid = []
    for idx, row in net.trafo.iterrows():
        try:
            converted[idx] = _branch_impedance_pu(row, s_base)
        except ValueError as exc:
            invalid.append((idx, str(exc)))
    if invalid:
        report.n_trafo_kept += len(net.trafo)
        detail = "; ".join(f"transformer {idx}: {reason}" for idx, reason in invalid)
        report.warnings.append(
            "transformer collapse skipped atomically because not every transformer "
            f"has usable data ({detail})."
        )
        return net

    # --- rebase existing lines that sit at another voltage level ------------
    for idx, row in net.line.iterrows():
        v_base = float(net.bus.vn_kv[int(row.from_bus)])
        if abs(v_base - v_ref) < 1e-12:
            continue
        r_pu, x_pu, s_max_pu = _line_impedance_pu(row, v_base, s_base)
        length = _finite_or(getattr(row, "length_km", 1.0), 1.0)
        if length <= 0.0:
            length = 1.0
        parallel = _finite_or(getattr(row, "parallel", 1.0), 1.0)
        if parallel <= 0.0:
            parallel = 1.0
        net.line.at[idx, "r_ohm_per_km"] = r_pu * z_ref / length
        net.line.at[idx, "x_ohm_per_km"] = x_pu * z_ref / length
        net.line.at[idx, "max_i_ka"] = s_max_pu * s_base / (math.sqrt(3.0) * v_ref)
        if "c_nf_per_km" in net.line.columns:
            c_nf = _finite_or(getattr(row, "c_nf_per_km", 0.0), 0.0)
            net.line.at[idx, "c_nf_per_km"] = c_nf * parallel * (v_base / v_ref) ** 2
        if "g_us_per_km" in net.line.columns:
            g_us = _finite_or(getattr(row, "g_us_per_km", 0.0), 0.0)
            net.line.at[idx, "g_us_per_km"] = g_us * parallel * (v_base / v_ref) ** 2
        net.line.at[idx, "parallel"] = 1
        if "df" in net.line.columns:
            # ``max_i_ka`` now contains the original parallel and derating
            # factors, so reset them to avoid applying either one twice.
            net.line.at[idx, "df"] = 1.0

    # --- transformers become lines ------------------------------------------
    for idx, row in list(net.trafo.iterrows()):
        tap = getattr(row, "tap_pos", None)
        neutral = getattr(row, "tap_neutral", None)
        if tap is not None and neutral is not None and np.isfinite(tap) \
                and np.isfinite(neutral) and float(tap) != float(neutral):
            report.warnings.append(
                f"transformer {idx} is at tap {tap:g} (neutral {neutral:g}); the "
                "collapsed equivalent represents the nominal ratio only."
            )
        shift = _finite_or(getattr(row, "shift_degree", 0.0), 0.0)
        if shift:
            # Radial: the shift rotates every voltage and current downstream by
            # the same angle, leaving magnitudes, currents and losses - all the
            # branch-flow models and every limit use - unchanged.  Meshed: it
            # drives loop flows, which the collapsed line cannot reproduce.
            report.warnings.append(
                f"transformer {idx} has a {shift:g} degree phase shift; dropping it is "
                "exact on a radial network (it only rotates the downstream angles) but "
                "changes loop flows on a meshed one."
            )
        r_pu, x_pu, s_max_pu = converted[idx]
        # vk is stated on the transformer's own rated voltages; refer it to the
        # bus base.  A rated ratio that differs from the bus ratio is an
        # off-nominal ratio, which the series equivalent cannot represent.
        vn_lv = _finite_or(getattr(row, "vn_lv_kv", np.nan), np.nan)
        vn_hv = _finite_or(getattr(row, "vn_hv_kv", np.nan), np.nan)
        bus_lv = float(net.bus.vn_kv.at[int(row.lv_bus)])
        bus_hv = float(net.bus.vn_kv.at[int(row.hv_bus)])
        if np.isfinite(vn_lv) and vn_lv > 0.0:
            ratio_sq = (vn_lv / bus_lv) ** 2
            r_pu, x_pu = r_pu * ratio_sq, x_pu * ratio_sq
        # The rated current of each winding is sn / (sqrt(3) vn_rated); referred
        # to the bus base it is scaled by bus / rated voltage, and pandapower
        # loads a transformer by its worse winding.
        sides = [bus / rated for bus, rated in ((bus_hv, vn_hv), (bus_lv, vn_lv))
                 if np.isfinite(rated) and rated > 0.0]
        if sides and s_max_pu * s_base < UNRATED_MVA_THRESHOLD:
            s_max_pu *= min(sides)
        if (np.isfinite(vn_lv) and np.isfinite(vn_hv)
                and abs((vn_hv / vn_lv) / (bus_hv / bus_lv) - 1.0) > 1e-6):
            report.warnings.append(
                f"transformer {idx} is rated {vn_hv:g}/{vn_lv:g} kV on {bus_hv:g}/{bus_lv:g} kV "
                "buses; that off-nominal ratio is not represented by the collapsed equivalent.")
        pfe_kw = _finite_or(getattr(row, "pfe_kw", 0.0), 0.0)
        i0_percent = _finite_or(getattr(row, "i0_percent", 0.0), 0.0)
        if pfe_kw or i0_percent:
            report.warnings.append(
                f"transformer {idx} no-load branch (pfe_kw={pfe_kw:g}, "
                f"i0_percent={i0_percent:g}) is not represented by the series "
                "equivalent."
            )

        line_idx = pp.create_line_from_parameters(
            net,
            from_bus=int(row.hv_bus), to_bus=int(row.lv_bus), length_km=1.0,
            r_ohm_per_km=r_pu * z_ref, x_ohm_per_km=x_pu * z_ref, c_nf_per_km=0.0,
            max_i_ka=s_max_pu * s_base / (math.sqrt(3.0) * v_ref),
            name=f"trafo:{row.name if isinstance(row.name, str) else idx}",
            in_service=bool(getattr(row, "in_service", True)),
        )
        raw_loading = _finite_or(getattr(row, "max_loading_percent", np.nan), np.nan)
        if np.isfinite(raw_loading):
            if "max_loading_percent" not in net.line.columns:
                net.line["max_loading_percent"] = 100.0
            net.line.at[line_idx, "max_loading_percent"] = raw_loading
        report.n_trafo_collapsed += 1

    if report.n_trafo_collapsed:
        net.trafo = net.trafo.drop(index=list(converted))

    # --- rebase other elements parameterized in ohms or rated bus voltage ---
    bus_vn = net.bus["vn_kv"]
    shunts = getattr(net, "shunt", None)
    if shunts is not None and len(shunts) and "vn_kv" in shunts.columns:
        for idx, row in shunts.iterrows():
            old_bus_vn = float(bus_vn.at[int(row.bus)])
            if abs(old_bus_vn - v_ref) < 1e-12:
                continue
            vn_shunt = _finite_or(getattr(row, "vn_kv", np.nan), old_bus_vn)
            if vn_shunt <= 0.0:
                vn_shunt = old_bus_vn
            shunts.at[idx, "vn_kv"] = vn_shunt * (v_ref / old_bus_vn)

    for table, bus_col, z_cols in (
        ("xward", "bus", ("r_ohm", "x_ohm")),
        ("switch", "bus", ("z_ohm",)),
        ("svc", "bus", ("x_l_ohm", "x_cvar_ohm")),
        ("ssc", "bus", ("r_ohm", "x_ohm")),
        ("tcsc", "from_bus", ("x_l_ohm", "x_cvar_ohm")),
    ):
        frame = getattr(net, table, None)
        if frame is None or len(frame) == 0:
            continue
        present = [c for c in z_cols if c in frame.columns]
        if not present:
            continue
        for idx, row in frame.iterrows():
            if table == "switch" and getattr(row, "et", None) != "b":
                continue
            old_bus_vn = float(bus_vn.at[int(getattr(row, bus_col))])
            if abs(old_bus_vn - v_ref) < 1e-12:
                continue
            scale_sq = (v_ref / old_bus_vn) ** 2
            for col in present:
                val = _finite_or(getattr(row, col, np.nan), np.nan)
                if np.isfinite(val):
                    frame.at[idx, col] = val * scale_sq

    trafo3w = getattr(net, "trafo3w", None)
    if trafo3w is not None and len(trafo3w):
        for idx, row in trafo3w.iterrows():
            for bus_col, vn_col in (("hv_bus", "vn_hv_kv"),
                                    ("mv_bus", "vn_mv_kv"),
                                    ("lv_bus", "vn_lv_kv")):
                if vn_col not in trafo3w.columns:
                    continue
                old_bus_vn = float(bus_vn.at[int(getattr(row, bus_col))])
                vn_w = _finite_or(getattr(row, vn_col, np.nan), np.nan)
                if np.isfinite(vn_w) and abs(old_bus_vn - v_ref) >= 1e-12:
                    trafo3w.at[idx, vn_col] = vn_w * (v_ref / old_bus_vn)

    # every bus now lives at the reference voltage
    net.bus["vn_kv"] = v_ref
    return net


# --------------------------------------------------------------------------
# operating topology
# --------------------------------------------------------------------------
def _apply_switch_states(net, report: ConversionReport) -> None:
    """Turn switch states into the topology they imply.

    ``pandapower`` switch semantics:

    * ``et == "l"`` / ``"t"``, open at one end - retain the native open-end
      branch (its connected-end charging/no-load branch remains energised);
    * both ends open - the whole line or transformer is disconnected;
    * ``et == "b"``, closed and ideal - the buses are one electrical node;
    * a closed bus switch with impedance remains a native electrical branch;
    * ``et == "b"``, open - no connection, nothing to do.

    Closed bus-bus switches are resolved by fusing the buses, chains included,
    so no zero-impedance branch ever reaches a solver.
    """
    import pandapower as pp

    switches = getattr(net, "switch", None)
    if switches is None or len(switches) == 0:
        return

    counts = {"open line": 0, "open transformer": 0, "fused bus pairs": 0}
    applied = np.zeros(len(switches), dtype=bool)
    closed = switches.closed.astype(bool).to_numpy()
    kinds = switches.et.to_numpy()
    elements = switches.element.to_numpy()
    buses = switches.bus.to_numpy()
    impedance = (switches.z_ohm.fillna(0.0).to_numpy(dtype=float)
                 if "z_ohm" in switches.columns else np.zeros(len(switches)))
    opened: Dict[Tuple[str, int], set[int]] = {}
    for is_closed, kind, element, bus in zip(closed, kinds, elements, buses):
        if not is_closed and kind in ("l", "t"):
            opened.setdefault((kind, int(element)), set()).add(int(bus))

    for k, (is_closed, kind, element) in enumerate(zip(closed, kinds, elements)):
        if is_closed:
            applied[k] = kind in ("l", "t") or (kind == "b" and impedance[k] == 0.0)
            continue
        if kind == "l" and int(element) in net.line.index:
            counts["open line"] += 1
            row = net.line.loc[int(element)]
            ends = {int(row.from_bus), int(row.to_bus)}
            if ends <= opened[(kind, int(element))]:
                net.line.at[int(element), "in_service"] = False
                applied[k] = True
        elif kind == "t" and int(element) in net.trafo.index:
            counts["open transformer"] += 1
            row = net.trafo.loc[int(element)]
            ends = {int(row.hv_bus), int(row.lv_bus)}
            if ends <= opened[(kind, int(element))]:
                net.trafo.at[int(element), "in_service"] = False
                applied[k] = True
        elif kind == "b":
            applied[k] = True

    # Closed bus-bus switches: union-find, then fuse each group onto one bus.
    parent: Dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # pandapower energises no out-of-service bus through a closed coupler, and
    # a fused group keeps its lowest bus's row: fusing across that switch would
    # carry the out-of-service flag to energised buses.
    live_bus = (set(net.bus.index[net.bus.in_service.astype(bool)])
                if "in_service" in net.bus.columns else set(net.bus.index))
    for is_closed, kind, bus, element, z in zip(closed, kinds, buses, elements, impedance):
        if (is_closed and kind == "b" and z == 0.0
                and int(bus) in live_bus and int(element) in live_bus):
            ra, rb = find(int(bus)), find(int(element))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups: Dict[int, List[int]] = {}
    for bus in list(parent):
        groups.setdefault(find(bus), []).append(bus)

    for keep, members in groups.items():
        others = [b for b in members if b != keep and b in net.bus.index]
        if others and keep in net.bus.index:
            group_rows = net.bus.loc[[keep] + others]
            min_vm: Optional[float] = None
            max_vm: Optional[float] = None
            if "min_vm_pu" in group_rows.columns:
                lo_vals = group_rows["min_vm_pu"].to_numpy(dtype=float)
                lo_vals = lo_vals[np.isfinite(lo_vals) & (lo_vals > 0.0)]
                if lo_vals.size:
                    min_vm = float(np.max(lo_vals))
            if "max_vm_pu" in group_rows.columns:
                hi_vals = group_rows["max_vm_pu"].to_numpy(dtype=float)
                hi_vals = hi_vals[np.isfinite(hi_vals) & (hi_vals < 2.0)]
                if hi_vals.size:
                    max_vm = float(np.min(hi_vals))
            references = {
                (table, column): frame[column].copy()
                for table, columns in _NATIVE_AC_BUS_COLUMNS.items()
                if (frame := getattr(net, table, None)) is not None
                for column in columns if column in frame.columns
            }
            pp.toolbox.fuse_buses(net, keep, others)
            if min_vm is not None:
                net.bus.at[keep, "min_vm_pu"] = min_vm
            if max_vm is not None:
                net.bus.at[keep, "max_vm_pu"] = max_vm
            for (table, column), values in references.items():
                frame = getattr(net, table)
                frame[column] = values.reindex(frame.index).replace({bus: keep for bus in others})
            counts["fused bus pairs"] += len(others)

    # Drop only switches already expressed exactly as topology.  Open-ended
    # branches, finite-impedance bus couplers, and unconverted switch kinds
    # (for example a 't3' winding switch) retain their native AC semantics.
    keep = switches.index[~applied]
    net.switch = net.switch.loc[[i for i in keep if i in net.switch.index]]
    report.switches_applied = {k: v for k, v in counts.items() if v}


def _drop_out_of_service_branches(net, report: ConversionReport) -> None:
    """Remove branches that are switched out.

    The radial topology is built by walking the branch tables, so an
    out-of-service line left in place would be walked as if it conducted.
    """
    import pandapower as pp

    # An out-of-service bus is de-energised in pandapower, together with
    # every element on it; left in place it would be walked as a live node.
    if "in_service" in net.bus.columns:
        dead_buses = net.bus.index[~net.bus.in_service.astype(bool)].tolist()
        if dead_buses:
            for table, columns in _NATIVE_AC_BUS_COLUMNS.items():
                frame = getattr(net, table, None)
                if frame is None or not len(frame):
                    continue
                attached = np.zeros(len(frame), dtype=bool)
                for column in columns:
                    if column in frame.columns:
                        attached |= frame[column].isin(dead_buses).to_numpy()
                if np.any(attached):
                    setattr(net, table, frame.loc[~attached].copy())
                    report.dropped_out_of_service[table] = int(np.sum(attached))
            pp.toolbox.drop_buses(net, dead_buses, drop_elements=True)
            report.dropped_out_of_service["bus"] = len(dead_buses)

    for table, dropper in (("line", pp.toolbox.drop_lines),
                           ("trafo", pp.toolbox.drop_trafos)):
        frame = getattr(net, table, None)
        if frame is None or len(frame) == 0 or "in_service" not in frame.columns:
            continue
        dead = frame.index[~frame.in_service.astype(bool)].tolist()
        if dead:
            dropper(net, dead)
            report.dropped_out_of_service[table] = len(dead)


def _renumber_buses(net, report: ConversionReport) -> None:
    """Number buses contiguously from 0, recording the translation.

    OpenRC indexes its per-bus arrays by bus number, so a model with custom IDs
    or with gaps left by fused buses is renumbered.  ``report.bus_index_map``
    records ``{original_bus_id: new_bus_id}``.
    """
    import pandapower as pp

    original = [int(b) for b in net.bus.index]
    report.bus_index_map = {old: new for new, old in enumerate(original)}
    if original != list(range(len(original))):
        references = {
            (table, column): frame[column].copy()
            for table, columns in _NATIVE_AC_BUS_COLUMNS.items()
            if (frame := getattr(net, table, None)) is not None
            for column in columns if column in frame.columns
        }
        # create_continuous_bus_index sorts the bus table by index first, so
        # the translation is pandapower's lookup, not the table order.
        lookup = pp.toolbox.create_continuous_bus_index(net, start=0)
        if not isinstance(lookup, dict):          # older pandapower returns nothing
            lookup = {old: new for new, old in enumerate(sorted(original))}
        report.bus_index_map = {int(old): int(new) for old, new in lookup.items()}
        for (table, column), values in references.items():
            getattr(net, table)[column] = values.map(
                lambda bus: report.bus_index_map.get(bus, bus))


def _translate_presets(presets, bus_index_map: Dict[int, int]
                       ) -> Dict[str, List[Union[int, str]]]:
    """Axis presets given in original bus IDs, expressed in OpenRC numbering.

    Plain bus numbers stay integers; entries with a ``p``/``q`` marker stay
    strings such as ``"3q"``.
    """
    from OpenRC.cases.axes import RegionAxis

    if presets is not None and not isinstance(presets, dict):
        raise TypeError(f"axis_presets must be a dict, got {type(presets).__name__}")
    out: Dict[str, List[Union[int, str]]] = {}
    for name, buses in (presets or {}).items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"axis preset key must be a non-empty string, got {name!r}")
        if isinstance(buses, (str, bytes, bytearray)):
            raise TypeError(f"axis preset {name!r} must be a sequence of axes, not str or bytes")
        translated: List[Union[int, str]] = []
        for bus in buses:
            axis = RegionAxis.parse(bus)
            if axis.bus not in bus_index_map:
                raise ValueError(
                    f"axis preset {name!r} names bus {axis.bus}, which the network does "
                    "not contain (or which a closed bus-bus switch fused away)"
                )
            mapped = bus_index_map[axis.bus]
            plain_int = isinstance(bus, int) or (isinstance(bus, str) and bus.strip().isdigit())
            translated.append(mapped if plain_int and axis.quantity == "p"
                              else f"{mapped}{axis.quantity}")
        out[name] = translated
    return out


# --------------------------------------------------------------------------
# feeders
# --------------------------------------------------------------------------
def _feeder_components(net) -> List[Tuple[int, List[int]]]:
    """``[(infeed bus, [buses in its part]), ...]`` sorted by infeed bus.

    Excludes de-energised islands without an external grid or slack generator.
    """
    from OpenRC.powerflow.topology import branch_endpoints

    buses = [int(b) for b in net.bus.index]
    parent = {b: b for b in buses}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v, _, _ in branch_endpoints(net):
        if u in parent and v in parent:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

    members: Dict[int, List[int]] = {}
    for b in buses:
        members.setdefault(find(b), []).append(b)

    infeed_set: set[int] = set()
    ext = getattr(net, "ext_grid", None)
    if ext is not None and len(ext):
        live = ext[ext["in_service"].astype(bool)] if "in_service" in ext.columns else ext
        infeed_set.update(int(b) for b in live.bus)
    gen = getattr(net, "gen", None)
    if gen is not None and len(gen) and "slack" in gen.columns:
        live = gen[gen["in_service"].astype(bool)] if "in_service" in gen.columns else gen
        infeed_set.update(int(b) for b in live.loc[live["slack"].astype(bool), "bus"])
    infeeds = sorted(infeed_set)
    out = []
    seen_components = set()
    for infeed in infeeds:
        if infeed in parent:
            component = find(infeed)
            if component in seen_components:
                continue
            seen_components.add(component)
            out.append((infeed, sorted(members[component])))
    return out


def list_feeders(net) -> "pd.DataFrame":
    """Table of the AC-connected, externally fed parts of a network.

    Returns a ``pandas.DataFrame`` with one row per feeder: its index (what to
    pass as ``feeder=``), infeed bus, and size.  Switch states are applied first,
    so the table reflects the network as operated.  Extraction additionally
    checks DC links and remote controls before treating a part as independent.
    """
    import copy as _copy

    import pandas as pd

    work = _copy.deepcopy(net)
    scratch = ConversionReport()
    _apply_switch_states(work, scratch)
    _drop_out_of_service_branches(work, scratch)
    rows = [{"feeder": k, "infeed_bus": infeed, "buses": len(buses)}
            for k, (infeed, buses) in enumerate(_feeder_components(work))]
    return pd.DataFrame(rows)


def _select_feeder(net, feeder: int, report: ConversionReport):
    """Reduce ``net`` to the ``feeder``-th independently fed part."""
    import copy as _copy

    import pandapower as pp

    if isinstance(feeder, (bool, np.bool_)) or not isinstance(feeder, (int, np.integer)):
        raise ValueError(f"feeder must be an integer index, got {feeder!r}")
    parts = _feeder_components(net)
    if not parts:
        raise ValueError("the network has no external grid, so it has no feeders to select")
    if not 0 <= int(feeder) < len(parts):
        raise ValueError(
            f"feeder={feeder} is out of range: the network has {len(parts)} "
            f"independently fed part(s), numbered 0..{len(parts) - 1}"
        )
    infeed, buses = parts[int(feeder)]
    selected = set(buses)
    links = getattr(net, "dcline", None)
    if links is not None and len(links):
        live = links[links.in_service.astype(bool)] if "in_service" in links else links
        if (live.from_bus.isin(selected) ^ live.to_bus.isin(selected)).any():
            raise ValueError("the selected AC feeder is coupled to another feeder by an HVDC "
                             "link; use the full network or an explicit subnet")
    sub = pp.toolbox.select_subnet(net, buses, include_switch_buses=False,
                                   include_results=False, keep_everything_else=False)
    for table, columns in _NATIVE_AC_BUS_COLUMNS.items():
        frame = getattr(net, table, None)
        if frame is None:
            continue
        keep = np.ones(len(frame), dtype=bool)
        for column in columns:
            if column in frame.columns:
                inside = frame[column].isin(selected)
                if column == "ref_bus":
                    inside |= ~frame[column].isin(net.bus.index)
                    active = (frame.in_service.astype(bool) if "in_service" in frame
                              else np.ones(len(frame), dtype=bool))
                    if (frame.bus.isin(selected) & ~inside & active).any():
                        raise ValueError("a selected converter uses an AC voltage reference "
                                         "outside this feeder; use the full network")
                keep &= inside.to_numpy()
        sub[table] = frame.loc[keep].copy()

    # A converter's DC terminal is part of the selected physical network.
    # Follow DC lines from retained live converters and retain that subsystem;
    # an active converter to another AC feeder means the parts are coupled,
    # so calling this an independent feeder extraction would be incorrect.
    converter_tables = ("vsc", "vsc_stacked", "vsc_bipolar", "b2b_vsc")
    dc_buses: set[int] = set()
    for table in converter_tables:
        frame = getattr(sub, table, None)
        if frame is None or not len(frame):
            continue
        live = frame[frame.in_service.astype(bool)] if "in_service" in frame.columns else frame
        for column in ("bus_dc", "bus_dc_plus", "bus_dc_minus"):
            if column in live.columns:
                dc_buses.update(int(bus) for bus in live[column].dropna())
    dc_lines = getattr(net, "line_dc", None)
    if dc_lines is not None and len(dc_lines):
        live = dc_lines[dc_lines.in_service.astype(bool)] if "in_service" in dc_lines else dc_lines
        changed = True
        while changed:
            old_count = len(dc_buses)
            for row in live.itertuples():
                if int(row.from_bus_dc) in dc_buses or int(row.to_bus_dc) in dc_buses:
                    dc_buses.update((int(row.from_bus_dc), int(row.to_bus_dc)))
            changed = len(dc_buses) != old_count
    for table in converter_tables:
        original = getattr(net, table, None)
        if original is None or not len(original):
            continue
        live = original[original.in_service.astype(bool)] if "in_service" in original else original
        outside = live.loc[~live.index.isin(sub[table].index)]
        for column in ("bus_dc", "bus_dc_plus", "bus_dc_minus"):
            if column in outside.columns and outside[column].isin(dc_buses).any():
                raise ValueError("the selected AC feeder is coupled to another feeder by a DC "
                                 "converter network; use the full network or an explicit subnet")
        retained = sub[table]
        for column in ("bus_dc", "bus_dc_plus", "bus_dc_minus"):
            if column in retained.columns:
                retained = retained.loc[retained[column].isin(dc_buses)]
        sub[table] = retained.copy()
    for table, columns in (("bus_dc", ()), ("line_dc", ("from_bus_dc", "to_bus_dc")),
                           ("source_dc", ("bus_dc",)), ("load_dc", ("bus_dc",))):
        frame = getattr(net, table, None)
        if frame is None:
            continue
        keep = frame.index.isin(dc_buses) if table == "bus_dc" else np.ones(len(frame), dtype=bool)
        for column in columns:
            keep &= frame[column].isin(dc_buses).to_numpy()
        sub[table] = frame.loc[keep].copy()
    # Copy network-level power-flow settings omitted by pandapower select_subnet.
    for key in ("sn_mva", "f_hz", "user_pf_options", "q_capability_characteristic"):
        if key in net:
            sub[key] = _copy.deepcopy(net[key])
    report.warnings.append(
        f"kept feeder {feeder} of {len(parts)} (infeed bus {infeed}, {len(buses)} buses)"
    )
    return sub


# --------------------------------------------------------------------------
# asymmetric elements
# --------------------------------------------------------------------------
def _balance_asymmetric_elements(net, report: ConversionReport) -> None:
    """Replace per-phase loads and generators by their three-phase totals.

    OpenRC uses balanced single-phase equivalents.  Explicit ordinary loads
    and static generators carrying the phase totals keep every backend on the
    same injection data, including pandapower versions and DC/OPF paths that
    do not aggregate asymmetric tables.  Phase imbalance is not modelled;
    conversion emits a warning in ``report.warnings``.
    """
    import pandapower as pp

    converted = 0
    for table, target in (("asymmetric_load", "load"), ("asymmetric_sgen", "sgen")):
        frame = getattr(net, table, None)
        if frame is None or len(frame) == 0:
            continue
        for row in frame.itertuples():
            if hasattr(row, "in_service") and not bool(row.in_service):
                continue
            p_total = sum(_finite_or(getattr(row, f"p_{ph}_mw", 0.0), 0.0) for ph in "abc")
            q_total = sum(_finite_or(getattr(row, f"q_{ph}_mvar", 0.0), 0.0) for ph in "abc")
            # Preserve explicit scaling=0.0; only fall back to 1.0 when None or non-finite.
            scaling = _finite_or(getattr(row, "scaling", 1.0), 1.0)
            creator = pp.create_load if target == "load" else pp.create_sgen
            creator(net, bus=int(row.bus), p_mw=p_total * scaling,
                    q_mvar=q_total * scaling, name=f"{table}:{row.Index}")
            converted += 1
        setattr(net, table, frame.iloc[0:0])

    if converted:
        report.warnings.append(
            f"{converted} asymmetric load/generator(s) replaced by their three-phase "
            "totals; OpenRC's models are balanced, so phase imbalance is not "
            "represented."
        )


# --------------------------------------------------------------------------
# ratings
# --------------------------------------------------------------------------
def _handle_unrated_lines(net, report: ConversionReport,
                          default_rating_ka: Optional[float]) -> None:
    """Record unrated lines and replace placeholder ratings when ``default_rating_ka`` is given."""
    from OpenRC.powerflow.topology import unrated_lines

    unrated = unrated_lines(net)
    report.n_unrated = len(unrated)
    if not unrated:
        return

    if default_rating_ka is not None:
        rating = float(default_rating_ka)
        if not np.isfinite(rating) or rating <= 0.0:
            raise ValueError(f"default_rating_ka must be finite and positive, got {rating!r}")
        net.line.loc[unrated, "max_i_ka"] = rating
        report.n_rating_filled = len(unrated)
        report.warnings.append(
            f"{len(unrated)} line(s) carried a placeholder rating and were given "
            f"default_rating_ka={rating:g} kA."
        )
        return

    report.warnings.append(
        f"{len(unrated)} line(s) carry a placeholder rather than a thermal rating "
        "(MATPOWER's 9900 MVA or pandapower's 99999 kA) and are treated as "
        "unconstrained. 'lindistflow', 'socp' and the AC oracles handle that; "
        "Model-II and both Method-2 models need real ratings - pass "
        "default_rating_ka to supply one."
    )


def _short_description(report: ConversionReport) -> str:
    """One line for ``case.summary()``; the full notes stay in the metadata."""
    bits = [f"imported from pandapower ({report.n_bus} buses"]
    if report.n_trafo_collapsed:
        bits.append(f"{report.n_trafo_collapsed} transformer(s) collapsed to "
                    f"{report.reference_kv:g} kV")
    if report.n_unrated and not report.n_rating_filled:
        bits.append(f"{report.n_unrated} unrated line(s)")
    text = ", ".join(bits) + ")"
    if report.warnings:
        text += f"; {len(report.warnings)} note(s) in metadata['conversion']"
    return text
