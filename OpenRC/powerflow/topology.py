"""Radial topology extraction - turns a ``pandapower`` net into a parent/child tree.

A branch-flow model needs two facts about its network: *which bus is the
slack* and *where each bus sits in the variable vector*.  :class:`RadialTopology`
derives them once, by a breadth-first search from the slack bus, so the same
LinDistFlow / SOCP code runs unmodified on any radial network.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ==========================================================================
# Radiality and root detection - both independent of any declared slack bus
# ==========================================================================
class MeshedNetworkError(ValueError):
    """Raised by radial branch-flow models when the network is not a valid single-root tree.

    Raised by DistFlow, SOCP, exact branch-flow, and radial sweep models when
    the network is meshed, disconnected, multi-infeed, or contains unsupported
    shunt/FACTS/transformer elements.  AC/DC power-flow and OPF oracles do not
    require radiality.

    Subclass of :class:`ValueError`.
    """


class AmbiguousSlackError(ValueError):
    """The infeed bus could not be determined and must be given explicitly."""


class UnratedBranchError(ValueError):
    """A model needs thermal ratings that some branches do not have."""


#: Apparent-power rating above which a branch is treated as *unrated*.
#:
#: MATPOWER (``rateA = 9900`` MVA) and ``pandapower`` (``max_i_ka = 99999``)
#: encode unconstrained branches with large sentinel values that would otherwise
#: produce ~1e11 Mvar loss bounds in ``r * I_max^2`` terms.  Because physical
#: transmission circuits remain well below 9 GVA, this threshold separates
#: sentinel placeholders from nameplate thermal ratings.
UNRATED_MVA_THRESHOLD = 9000.0


def effective_line_current_ka(row) -> float:
    """Usable current of a pandapower line, including derating and parallels.

    ``max_i_ka`` is the rating of one circuit.  Pandapower applies ``df`` as a
    derating factor and ``parallel`` as the number of identical circuits; the
    DistFlow topology must use the same effective limit.
    """
    rating = float(row.max_i_ka)
    raw_derating = getattr(row, "df", 1.0)
    raw_parallel = getattr(row, "parallel", 1.0)
    derating = (1.0 if raw_derating is None or not math.isfinite(float(raw_derating))
                else float(raw_derating))
    parallel = (1.0 if raw_parallel is None or not math.isfinite(float(raw_parallel))
                or float(raw_parallel) <= 0.0 else float(raw_parallel))
    return rating * derating * parallel


def unrated_lines(net) -> List[int]:
    """Return indices of conductive in-service lines whose apparent-power rating meets or exceeds :data:`UNRATED_MVA_THRESHOLD`.

    Evaluates :math:`\\sqrt{3} V I` at each line's nominal voltage because a
    fixed apparent-power sentinel (e.g. 9900 MVA) corresponds to widely
    different currents across voltage levels (41 kA at 138 kV vs. 27 000 kA at
    0.2 kV).
    """
    if not len(getattr(net, "line", [])):
        return []
    vn = net.bus.vn_kv
    opened = _open_switch_elements(net, "l")
    out = []
    for idx, row in net.line.iterrows():
        # Skip out-of-service and open-switched lines (e.g. open tie lines in case33bw).
        if "in_service" in net.line.columns and not bool(row.in_service):
            continue
        if int(idx) in opened:
            continue
        i_ka = effective_line_current_ka(row)
        if not np.isfinite(i_ka):
            out.append(int(idx))
            continue
        v_kv = float(vn.get(int(row.from_bus), np.nan))
        s_mva = math.sqrt(3.0) * v_kv * i_ka if np.isfinite(v_kv) else np.inf
        if s_mva >= UNRATED_MVA_THRESHOLD:
            out.append(int(idx))
    return out


#: Branch-like ``pandapower`` tables, with the columns naming their endpoints.
#: Every one of them makes a connection, so every one counts towards
#: radiality; checking ``line`` alone would call a transformer-connected network
#: radial however many loops it had.
_BRANCH_TABLES = {
    "line": ("from_bus", "to_bus"),
    "trafo": ("hv_bus", "lv_bus"),
    "impedance": ("from_bus", "to_bus"),
    "tcsc": ("from_bus", "to_bus"),
}

#: Tables that connect buses but that the DistFlow models cannot represent.
_UNSUPPORTED_TABLES = {
    "trafo": "two-winding transformers",
    "trafo3w": "three-winding transformers",
    "dcline": "DC lines",
    "impedance": "impedance branches",
    # These are not topological branches, but their electrical behaviour is
    # absent from OpenRC's fixed-P/Q branch-flow equations.
    "gen": "voltage-controlled generators",
    "shunt": "voltage-dependent shunts",
    "ward": "Ward equivalents",
    "xward": "extended Ward equivalents",
    "motor": "induction motors",
    "asymmetric_load": "unconverted asymmetric loads",
    "asymmetric_sgen": "unconverted asymmetric static generators",
    "svc": "static var compensators (SVC)",
    "ssc": "static synchronous compensators (SSC)",
    "tcsc": "thyristor-controlled series capacitors (TCSC)",
    "vsc": "voltage-source converters (VSC)",
    "vsc_stacked": "stacked voltage-source converters",
    "vsc_bipolar": "bipolar voltage-source converters",
    "b2b_vsc": "back-to-back voltage-source converters",
    "bus_dc": "DC buses",
    "line_dc": "DC network lines",
    "source_dc": "DC voltage sources",
    "load_dc": "DC network loads",
}


def _open_switch_elements(net, kind: str) -> set[int]:
    """Elements with an open terminal switch, hence no end-to-end path."""
    switches = getattr(net, "switch", None)
    if switches is None or not len(switches):
        return set()
    selected = switches[(switches.et == kind) & ~switches.closed.astype(bool)]
    return {int(element) for element in selected.element}


@dataclass
class RadialityReport:
    """Topological and element-support classification for radial branch-flow models.

    Evaluated from graph connectivity and element counts without requiring a
    slack bus: a connected graph is a tree iff ``n_components == 1`` and
    ``n_branch == n_bus - 1``.  When ``is_radial`` is ``False``, branch-flow
    oracles are inapplicable while AC/DC and OPF oracles remain available.
    """

    n_bus: int
    n_branch: int
    n_components: int
    n_loops: int
    branch_counts: Dict[str, int] = field(default_factory=dict)
    unsupported: Dict[str, int] = field(default_factory=dict)
    #: Count of in-service external grids / slack generators (DistFlow requires <= 1).
    n_infeeds: int = 1

    @property
    def is_connected(self) -> bool:
        return self.n_components == 1

    @property
    def is_radial(self) -> bool:
        """One component, no loops, nothing unrepresentable, at most one infeed."""
        return (self.is_connected and self.n_loops == 0 and not self.unsupported
                and self.n_infeeds <= 1)

    def __bool__(self) -> bool:
        return self.is_radial

    @property
    def message(self) -> str:
        """Human-readable summary listing all conditions preventing radial DistFlow use."""
        reasons = []
        if self.unsupported:
            what = ", ".join(f"{n} {k}" for k, n in sorted(self.unsupported.items()))
            reasons.append(
                f"it contains {what}, which the DistFlow oracles cannot represent "
                "directly (on a radial network, OpenRC.case_from_pandapower collapses "
                "two-winding transformers into equivalent branches automatically)"
            )
        if not self.is_connected:
            reasons.append(
                f"it splits into {self.n_components} disconnected parts, and a "
                "security region is only defined for one connected network"
            )
        if self.n_infeeds > 1:
            reasons.append(
                f"it has {self.n_infeeds} infeeds (external grids), and a branch-flow "
                "model has a single root; a network fed from several substations is "
                "not one radial feeder (extract a single feeder with "
                "pandapower.toolbox.select_subnet to use DistFlow on it)"
            )
        if self.n_loops:
            reasons.append(
                f"it is MESHED: {self.n_bus} buses and {self.n_branch} branches "
                f"leave {self.n_loops} independent loop(s), where a radial network "
                f"has exactly {self.n_bus - 1} branches and none (open the loops to "
                "a radial configuration)"
            )
        if reasons:
            body = "; ".join(reasons)
            return (
                f"the DistFlow oracles cannot be used on this network because "
                f"{body}. This restricts the DistFlow family only - the network "
                "still has a security region. Compute it with an oracle that "
                "carries no radiality assumption: 'dcopf'/'acopf' if the "
                "generation is dispatchable, or 'dcpf'/'acpf' to evaluate a fixed "
                "dispatch."
            )
        return (
            f"radial: {self.n_bus} buses, {self.n_branch} branches, one component, "
            "no loops."
        )

    def summary(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.branch_counts.items()) if v)
        return f"{'radial' if self.is_radial else 'NOT radial'} [{counts}] - {self.message}"


def branch_endpoints(net) -> List[Tuple[int, int, str, int]]:
    """Every bus-to-bus connection as ``(u, v, table, row)``.

    Covers lines, transformers, impedances and closed bus-bus switches.
    Out-of-service rows and branches with an open terminal are skipped,
    because they provide no end-to-end path (their native shunts may remain).
    """
    edges: List[Tuple[int, int, str, int]] = []
    # Represent each in-service three-winding transformer as two edges (HV-MV,
    # HV-LV) so connected multi-voltage components are preserved.
    trafo3w = getattr(net, "trafo3w", None)
    if trafo3w is not None and len(trafo3w):
        live = (trafo3w["in_service"].to_numpy(dtype=bool)
                if "in_service" in trafo3w.columns else np.ones(len(trafo3w), dtype=bool))
        for row, ok in zip(trafo3w.itertuples(), live):
            if ok:
                edges.append((int(row.hv_bus), int(row.mv_bus), "trafo3w", int(row.Index)))
                edges.append((int(row.hv_bus), int(row.lv_bus), "trafo3w", int(row.Index)))
    for table, (col_u, col_v) in _BRANCH_TABLES.items():
        frame = getattr(net, table, None)
        if frame is None or len(frame) == 0:
            continue
        in_service = (frame["in_service"].to_numpy(dtype=bool)
                      if "in_service" in frame.columns
                      else np.ones(len(frame), dtype=bool))
        us = frame[col_u].to_numpy()
        vs = frame[col_v].to_numpy()
        opened = _open_switch_elements(net, {"line": "l", "trafo": "t"}.get(table, ""))
        for row, (u, v, live) in enumerate(zip(us, vs, in_service)):
            if live and int(frame.index[row]) not in opened:
                edges.append((int(u), int(v), table, int(frame.index[row])))
    switches = getattr(net, "switch", None)
    if switches is not None and len(switches):
        coupled = switches[(switches.et == "b") & switches.closed.astype(bool)]
        for row in coupled.itertuples():
            edges.append((int(row.bus), int(row.element), "switch", int(row.Index)))
    return edges


def radiality_report(net, *, line_charging: str = "omit") -> RadialityReport:
    """Classify a network as radial, meshed, disconnected or unsupported."""
    n_bus = int(len(net.bus))
    edges = branch_endpoints(net)

    counts: Dict[str, int] = {}
    for _, _, table, _ in edges:
        counts[table] = counts.get(table, 0) + 1

    unsupported: Dict[str, int] = {}
    switches = getattr(net, "switch", None)
    if switches is not None and len(switches):
        coupled = switches[(switches.et == "b") & switches.closed.astype(bool)]
        if len(coupled):
            unsupported["unconverted bus-bus switches"] = len(coupled)
        # The constant-charging approximation normally allocates half to each
        # endpoint.  An open-ended pi branch instead charges through its one
        # connected end, which this series-only tree cannot represent.
        if line_charging == "constant":
            opened = switches[(switches.et == "l") & ~switches.closed.astype(bool)]
            count = 0
            for idx in opened.element.unique():
                if int(idx) not in net.line.index:
                    continue
                row = net.line.loc[int(idx)]
                ends = {int(row.from_bus), int(row.to_bus)}
                open_ends = set(opened.loc[opened.element == idx, "bus"].astype(int))
                capacitance = float(getattr(row, "c_nf_per_km", 0.0))
                if (bool(getattr(row, "in_service", True)) and not ends <= open_ends
                        and np.isfinite(capacitance) and capacitance != 0.0):
                    count += 1
            if count:
                unsupported["open-ended line charging with constant approximation"] = count
    for table, label in _UNSUPPORTED_TABLES.items():
        frame = getattr(net, table, None)
        if frame is not None and len(frame):
            live = (frame[frame["in_service"].astype(bool)]
                    if "in_service" in frame.columns else frame)
            if len(live):
                unsupported[label] = int(len(live))
    # ZIP loads change their P/Q with voltage.  A fixed-P/Q branch-flow model
    # cannot certify the original network when those components are present.
    # Include both legacy common-P/Q and newer separate-P/Q column names.
    loads = getattr(net, "load", None)
    if loads is not None and len(loads):
        live = loads[loads["in_service"].astype(bool)] if "in_service" in loads else loads
        columns = [name for name in (
            "const_z_percent", "const_i_percent", "const_z_p_percent",
            "const_i_p_percent", "const_z_q_percent", "const_i_q_percent",
        ) if name in live.columns]
        if columns and len(live):
            fractions = live[columns].to_numpy(dtype=float)
            dependent = np.any((fractions != 0.0) | ~np.isfinite(fractions), axis=1)
            if np.any(dependent):
                unsupported["voltage-dependent ZIP loads"] = int(np.sum(dependent))
    # The pandapower adapter can attach non-topological reasons why its scalar
    # DistFlow representation is unsafe (for example, bus voltage intervals
    # with no common intersection).  Keep the marker on the copied network so
    # every later radiality check reaches the same decision.
    extra_unsupported = getattr(net, "_openrc_distflow_unsupported", None)
    if isinstance(extra_unsupported, dict):
        for label, count in extra_unsupported.items():
            if int(count) > 0:
                unsupported[str(label)] = int(count)

    # Connected components by union-find; no root needed.
    parent = list(range(n_bus))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v, _, _ in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    n_components = len({find(b) for b in range(n_bus)}) if n_bus else 0
    n_branch = len(edges)
    # Cyclomatic number: loops = E - V + C.
    n_loops = max(0, n_branch - n_bus + n_components)

    infeed_buses: set[int] = set()
    ext = getattr(net, "ext_grid", None)
    if ext is not None and len(ext):
        live = ext[ext["in_service"].astype(bool)] if "in_service" in ext.columns else ext
        infeed_buses.update(int(b) for b in live.bus)
    gen = getattr(net, "gen", None)
    if gen is not None and len(gen) and "slack" in gen.columns:
        live = gen[gen["in_service"].astype(bool)] if "in_service" in gen.columns else gen
        infeed_buses.update(int(b) for b in live.loc[live["slack"].astype(bool), "bus"])
    n_infeeds = len(infeed_buses)

    return RadialityReport(
        n_bus=n_bus, n_branch=n_branch, n_components=n_components,
        n_loops=n_loops, branch_counts=counts, unsupported=unsupported,
        n_infeeds=n_infeeds,
    )


def infer_slack_bus(net) -> Tuple[int, str]:
    """Find the infeed bus automatically, and say how it was decided.

    A radial feeder has one connection to the upstream system, and that bus is
    the root of the tree.  The rules are tried in order of how directly they
    state the intent:

    1. a single external grid - what ``pandapower`` uses for exactly this;
    2. a single slack generator;
    3. the unique bus at the highest nominal voltage, which on a distribution
       feeder is the substation busbar;
    4. if several buses share that voltage, the unique one among them with a
       single branch, i.e. the head of the feeder.

    Returns ``(bus, how)``; ``how`` is recorded in the case metadata so the
    choice can be inspected.

    Raises
    ------
    AmbiguousSlackError
        When none of the four rules identifies a unique slack bus.
    """
    ext = getattr(net, "ext_grid", None)
    if ext is not None and len(ext):
        live = ext[ext["in_service"].astype(bool)] if "in_service" in ext.columns else ext
        buses = sorted({int(b) for b in live.bus})
        if len(buses) == 1:
            return buses[0], "external grid"
        if len(buses) > 1:
            raise AmbiguousSlackError(
                f"the network declares {len(buses)} in-service external grids "
                f"(buses {buses}). A radial feeder has a single infeed; keep one, "
                "or build the GridCase with an explicit slack_bus."
            )

    gen = getattr(net, "gen", None)
    if gen is not None and len(gen) and "slack" in gen.columns:
        live = gen[gen["in_service"].astype(bool)] if "in_service" in gen.columns else gen
        slack_gens = live[live["slack"].astype(bool)]
        buses = sorted({int(b) for b in slack_gens.bus})
        if len(buses) == 1:
            return buses[0], "slack generator"
        if len(buses) > 1:
            raise AmbiguousSlackError(
                f"the network declares {len(buses)} slack generators (buses "
                f"{buses}); keep one, or pass slack_bus explicitly."
            )

    vn = net.bus["vn_kv"].to_numpy(dtype=float)
    if len(vn):
        highest = float(np.nanmax(vn))
        candidates = [int(b) for b, v in zip(net.bus.index, vn) if v == highest]
        if len(candidates) == 1:
            return candidates[0], f"highest nominal voltage ({highest:g} kV)"

        # Among multiple buses at the highest voltage, select a unique degree-1 terminal bus.
        degree: Dict[int, int] = {}
        for u, v, _, _ in branch_endpoints(net):
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        heads = [b for b in candidates if degree.get(b, 0) == 1]
        if len(heads) == 1:
            return heads[0], f"single feeder head at {highest:g} kV"

        raise AmbiguousSlackError(
            f"{len(candidates)} buses share the highest nominal voltage "
            f"({highest:g} kV) and {len(heads)} of them are feeder heads, so the "
            "infeed cannot be inferred. Declare an external grid on the "
            "substation bus, or build the GridCase with an explicit slack_bus."
        )

    raise AmbiguousSlackError("the network has no buses, so it has no infeed.")


@dataclass(frozen=True)
class Branch:
    """One oriented branch of the radial tree, with its electrical parameters.

    The orientation is always *away from the slack*: ``parent`` is one step
    closer to the substation than ``child``.  ``reversed_in_net`` records whether
    that orientation is opposite to the ``from_bus``/``to_bus`` order stored in
    ``pandapower``, which matters when reading results back out of an AC solve.
    """

    index: int          # row index in net.line
    parent: int         # bus index on the substation side
    child: int          # bus index on the downstream side
    r_ohm: float        # total resistance  (length x r_ohm_per_km)
    x_ohm: float        # total reactance   (length x x_ohm_per_km)
    i_max_a: float      # thermal limit in amperes (per phase)
    reversed_in_net: bool = False
    #: ``False`` when the branch's rating is a placeholder (see
    #: :func:`unrated_lines`).  An unrated branch has no thermal constraint, and a
    #: model that bounds losses by the rating cannot be used on it.
    rated: bool = True

    def __post_init__(self) -> None:
        for field_name, val in (("index", self.index), ("parent", self.parent), ("child", self.child)):
            if (
                isinstance(val, (bool, np.bool_))
                or not isinstance(val, (int, np.integer))
                or int(val) < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        if int(self.parent) == int(self.child):
            raise ValueError("branch parent and child must be distinct buses")
        for field_name, num_val in (("r_ohm", self.r_ohm), ("x_ohm", self.x_ohm)):
            if (
                isinstance(num_val, (bool, np.bool_))
                or not isinstance(num_val, (int, float, np.number))
                or not math.isfinite(float(num_val))
            ):
                raise ValueError(f"{field_name} must be a finite number")
        if (
            isinstance(self.i_max_a, (bool, np.bool_))
            or not isinstance(self.i_max_a, (int, float, np.number))
            or math.isnan(float(self.i_max_a))
            or float(self.i_max_a) < 0.0
        ):
            raise ValueError("i_max_a must be a non-negative number")
        if not isinstance(self.reversed_in_net, (bool, np.bool_)):
            raise ValueError("reversed_in_net must be a boolean")
        if not isinstance(self.rated, (bool, np.bool_)):
            raise ValueError("rated must be a boolean")

    @property
    def l_max(self) -> float:
        """Squared thermal current limit ``I_max^2`` in A^2, as used by DistFlow."""
        return self.i_max_a ** 2


class RadialTopology:
    """Parent/child structure of a radial network, rooted at the slack bus.

    Attributes
    ----------
    slack:
        Bus index of the external grid.
    branches:
        One :class:`Branch` per line, **ordered from the root outwards** (a valid
        topological order).  Iterating in reverse therefore visits every child
        before its parent, which is what the downstream power summation needs.
    buses:
        Non-slack bus indices in ascending order.
    bus_pos:
        ``bus index -> 0-based position`` inside :attr:`buses`.
    """

    def __init__(self, slack: int, branches: Sequence[Branch], n_bus: int) -> None:
        if (
            isinstance(n_bus, (bool, np.bool_))
            or not isinstance(n_bus, (int, float, np.number))
            or not math.isfinite(n_bus)
            or int(n_bus) != n_bus
            or int(n_bus) < 1
        ):
            raise ValueError("n_bus must be a positive integer")
        if (
            isinstance(slack, (bool, np.bool_))
            or not isinstance(slack, (int, float, np.number))
            or not math.isfinite(slack)
            or int(slack) != slack
            or not (0 <= int(slack) < int(n_bus))
        ):
            raise ValueError(f"slack must be an integer in 0..{int(n_bus) - 1}")
        if isinstance(branches, (str, bytes, bytearray)):
            raise TypeError("branches must be a sequence of Branch objects, not a string")
        self.slack = int(slack)
        self.branches: List[Branch] = list(branches)
        self.n_bus = int(n_bus)
        for br in self.branches:
            if not isinstance(br, Branch):
                raise TypeError("every entry in branches must be a Branch instance")
            if not (0 <= br.parent < self.n_bus and 0 <= br.child < self.n_bus):
                raise ValueError(
                    f"branch endpoints ({br.parent}, {br.child}) out of range 0..{self.n_bus - 1}"
                )
        self.buses: List[int] = [b for b in range(self.n_bus) if b != self.slack]
        self.bus_pos: Dict[int, int] = {b: k for k, b in enumerate(self.buses)}

        # branch feeding each bus, and the list of branches leaving it
        #: Filled in by :meth:`from_case`; ``None`` when built directly.
        self.report: Optional[RadialityReport] = None
        #: Branches whose rating is a placeholder - see :func:`unrated_lines`.
        self.unrated_branches: List[Branch] = [br for br in self.branches if not br.rated]
        self.branch_to_child: Dict[int, Branch] = {br.child: br for br in self.branches}
        self.children_of: Dict[int, List[Branch]] = {b: [] for b in range(self.n_bus)}
        for br in self.branches:
            self.children_of[br.parent].append(br)

    # ------------------------------------------------------------- construction
    @classmethod
    def from_case(cls, case) -> "RadialTopology":
        """Build the tree from a :class:`~OpenRC.cases.base.GridCase`.

        Raises
        ------
        MeshedNetworkError
            If the network is not a single-infeed radial tree without unsupported elements.
        """
        net = case.net
        n_bus = int(len(net.bus))

        # Validate radiality and element support via radiality_report() before BFS.
        report = radiality_report(net, line_charging=case.line_charging)
        if not report.is_radial:
            raise MeshedNetworkError(
                f"case {case.name!r}: {report.message}"
            )

        slack = case.slack_bus
        n_line = int(len(net.line))
        # Out-of-service lines conduct nothing; the radiality report already
        # skips them, so the tree must too.
        live = (net.line.in_service.to_numpy(dtype=bool) if "in_service" in net.line.columns
                else np.ones(n_line, dtype=bool))
        # Not in place: ``to_numpy`` can return a view of the case's own table.
        live = live & ~np.isin(net.line.index.to_numpy(), list(_open_switch_elements(net, "l")))

        # Undirected adjacency: bus -> [(neighbour, line row index), ...]
        adjacency: Dict[int, List[tuple]] = {b: [] for b in range(n_bus)}
        f_bus = net.line.from_bus.to_numpy()
        t_bus = net.line.to_bus.to_numpy()
        for row in range(n_line):
            if not live[row]:
                continue
            i, j = int(f_bus[row]), int(t_bus[row])
            adjacency[i].append((j, row))
            adjacency[j].append((i, row))

        unrated = set(unrated_lines(net))
        line_index = net.line.index.to_numpy()
        r_km = net.line.r_ohm_per_km.to_numpy(dtype=float)
        x_km = net.line.x_ohm_per_km.to_numpy(dtype=float)
        length = net.line.length_km.to_numpy(dtype=float)
        i_max_ka = net.line.max_i_ka.to_numpy(dtype=float)
        derating = (net.line.df.to_numpy(dtype=float)
                    if "df" in net.line.columns else np.ones(n_line))
        derating = np.where(np.isfinite(derating), derating, 1.0)
        parallel = (net.line.parallel.to_numpy(dtype=float)
                    if "parallel" in net.line.columns else np.ones(n_line))
        parallel = np.where(np.isfinite(parallel) & (parallel > 0.0), parallel, 1.0)
        # The AC/DC oracles enforce min(100, max_loading_percent); so must the
        # branch-flow models (a collapsed transformer already carries it).
        loading = (net.line.max_loading_percent.to_numpy(dtype=float)
                   if "max_loading_percent" in net.line.columns else np.full(n_line, 100.0))
        loading = np.where(np.isfinite(loading) & (loading >= 0.0),
                           np.minimum(loading, 100.0), 100.0) / 100.0

        # BFS from the slack fixes the orientation of every branch.
        branches: List[Branch] = []
        visited = {slack}
        queue = deque([slack])
        while queue:
            parent = queue.popleft()
            for child, row in adjacency[parent]:
                if child in visited:
                    continue
                visited.add(child)
                branches.append(
                    Branch(
                        index=int(line_index[row]),
                        parent=parent,
                        child=child,
                        # Parallel circuits divide series impedance and add
                        # current capacity.  ``df`` derates only the capacity.
                        r_ohm=float(r_km[row] * length[row] / parallel[row]),
                        x_ohm=float(x_km[row] * length[row] / parallel[row]),
                        # pandapower stores kA; DistFlow is written in amperes
                        i_max_a=float(i_max_ka[row] * derating[row]
                                      * parallel[row] * loading[row] * 1000.0),
                        rated=int(line_index[row]) not in unrated,
                        reversed_in_net=(int(f_bus[row]) != parent),
                    )
                )
                queue.append(child)

        if len(visited) != n_bus:
            unreachable = sorted(set(range(n_bus)) - visited)
            raise MeshedNetworkError(  # ruled out by the report
                f"case {case.name!r}: buses {unreachable[:10]}... are not connected to "
                f"the slack bus {slack}"
            )
        if len(branches) != int(live.sum()):  # ruled out by the report
            raise MeshedNetworkError(
                f"case {case.name!r} contains a cycle: the BFS tree used "
                f"{len(branches)} of {n_line} lines"
            )
        topology = cls(slack, branches, n_bus)
        topology.report = report
        return topology

    # ----------------------------------------------------------------- queries
    def path_to_root(self, bus: int) -> List[Branch]:
        """Branches from ``bus`` up to (but excluding) the slack, child-first.

        Used by the tightened conservative model, which charges every branch with
        the losses accumulated along its own upstream path.
        """
        if isinstance(bus, (bool, np.bool_)) or not isinstance(bus, (int, np.integer)):
            raise ValueError(f"bus must be an integer bus index, got {bus!r}")
        path: List[Branch] = []
        node = int(bus)
        while node != self.slack:
            if len(path) >= self.n_bus:
                raise ValueError(f"cycle detected in path from bus {bus} to the slack bus")
            branch = self.branch_to_child.get(node)
            if branch is None:  # guarded by the constructor
                raise ValueError(f"bus {bus} has no path to the slack bus")
            path.append(branch)
            node = branch.parent
        return path

    def depth(self, bus: int) -> int:
        """Number of branches between ``bus`` and the slack (0 for the slack)."""
        return len(self.path_to_root(bus))

    def electrical_distance(self, bus: int) -> float:
        """Total series impedance magnitude ``sum |r + j x|`` along the path from ``bus`` to the slack bus, in ohms."""
        return float(sum(np.hypot(br.r_ohm, br.x_ohm) for br in self.path_to_root(bus)))

    def __len__(self) -> int:
        return len(self.branches)

    def __repr__(self) -> str:
        return f"<RadialTopology slack={self.slack} n_bus={self.n_bus} n_branch={len(self)}>"
