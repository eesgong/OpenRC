"""Registration of the networks that ship with OpenRC.

Adding a case
-------------
1. Put ``bus.csv``, ``line.csv``, ``load.csv`` and ``meta.json`` in
   ``OpenRC/cases/data/<name>/`` (the schema is in :mod:`OpenRC.cases.tabular`).
2. Add one ``_register_tabular_case(...)`` call below.

Registered cases are available via :func:`get_case` and ``--case`` on the CLI.
"""

from __future__ import annotations

import copy
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from OpenRC.cases.base import GridCase
from OpenRC.cases.tabular import build_case
from OpenRC.registry import Registry

DATA_ROOT = Path(__file__).resolve().parent / "data"

#: Registry of case *factories*.  Factories are called with no arguments and
#: return a fresh :class:`GridCase`; :func:`get_case` memoises the result so the
#: (relatively slow) ``pandapower`` construction happens at most once per name.
CASES: Registry = Registry("case")


def _register_tabular_case(
    name: str,
    data_dir: Path,
    *,
    doc: str,
    axis_presets: Optional[Dict[str, List[int]]] = None,
    load_scale: float = 1.0,
    v_nom_kv_override: Optional[float] = None,
) -> None:
    """Bind ``name`` to a factory that builds the case from its CSV directory.

    The factory accepts keyword overrides, so a registered case can be
    instantiated at a different setting without registering a second name -
    ``get_case("9bus", load_scale=0.70)``.
    """

    def _factory(**overrides) -> GridCase:
        settings: Dict[str, Any] = dict(axis_presets=axis_presets, load_scale=load_scale,
                                        v_nom_kv_override=v_nom_kv_override)
        settings.update(overrides)
        return build_case(name, data_dir, **settings)

    _factory.__doc__ = doc
    CASES.add(name, _factory, doc=doc)


# --------------------------------------------------------------------------
# Built-in networks
# --------------------------------------------------------------------------
_register_tabular_case(
    "9bus",
    DATA_ROOT / "9bus",
    doc="Dutch 10.5 kV MV feeder, 8 load buses - the paper's small case.",
)

_register_tabular_case(
    "9bus-scaled",
    DATA_ROOT / "9bus-scaled",
    doc="9bus with every rating x6 and every length /3 - the 8-D region study's variant.",
)


# Register pandapower.networks factories lazily; networks are instantiated only
# when requested via get_case().
def _register_pandapower_networks_once() -> None:
    try:
        from OpenRC.cases.pp_networks import register_pandapower_networks

        register_pandapower_networks(CASES)
    except Exception as exc:  # pandapower is always present in practice
        from OpenRC.utils import LOGGER

        LOGGER.warning("could not register the pandapower network library: %s", exc)


_register_pandapower_networks_once()


@lru_cache(maxsize=None)
def _build_case(name: str, overrides: tuple) -> GridCase:
    """Memoised construction; ``overrides`` is a hashable tuple of kwargs."""
    factory = CASES.get(name)
    case = factory(**dict(overrides)) if overrides else factory()
    case.factory_kwargs = dict(overrides)
    case.registry_name = name
    return case


_UNHASHABLE_REFERENCES: Dict[bytes, GridCase] = {}


def reference_case(name: str, overrides: Optional[dict] = None) -> GridCase:
    """Return the cached read-only template :class:`GridCase` for ``(name, overrides)``.

    :func:`get_case` returns an isolated :meth:`~GridCase.snapshot` of this
    instance; parallel checks compare against it to detect post-load edits.
    """
    overrides = dict(overrides or {})
    items = tuple(sorted(overrides.items()))
    try:
        hash(items)
    except TypeError:
        key = pickle.dumps((name, sorted(overrides.items())))
        if key not in _UNHASHABLE_REFERENCES:
            case = CASES.get(name)(**overrides)
            case.factory_kwargs = copy.deepcopy(overrides)
            case.registry_name = name
            _UNHASHABLE_REFERENCES[key] = case
        return _UNHASHABLE_REFERENCES[key]
    return _build_case(name, items)


def get_case(name: str, **overrides) -> GridCase:
    """Return the :class:`GridCase` registered under ``name``.

    Construction is memoised, but every call returns a private copy: changing
    a case (or its net) never leaks into later calls.  Keyword overrides go to
    the case factory and are part of the cache key, so
    ``get_case("9bus", load_scale=0.70)`` is built once as well.
    """
    return reference_case(name, overrides).snapshot()


def list_cases() -> List[tuple]:
    """``[(name, one-line description), ...]`` for ``demo.py list``."""
    return CASES.describe()


# --------------------------------------------------------------------------
# Networks built programmatically rather than from tables
# --------------------------------------------------------------------------
def _cigre_hv(
    *,
    load_scale: float = 1.0,
    vm_min_pu: float = 0.9,
    vm_max_pu: float = 1.1,
    axis_presets: Optional[Dict[str, List[int]]] = None,
    slack_vm_pu: Optional[Union[float, str]] = 1.0,
    line_charging: str = "omit",
) -> GridCase:
    """CIGRE HV transmission benchmark configured for optimal power flow.

    Because this network is meshed and contains transformers, radial DistFlow
    oracles raise :class:`~OpenRC.powerflow.MeshedNetworkError`; use ``dcopf``,
    ``acopf``, ``acpf``, or ``dcpf``::

        case = orc.get_case("cigre-hv")
        region = ExpansiveBuilder().build(
            case, case.resolve_axes("026"), orc.get_oracle("dcopf", case),
            max_vertices=40).polytope

    OPF parameters added to ``pandapower.networks.create_cigre_network_hv()``
    (with base loads scaled by 0.5 and zero-load POC markers at buses 6, 7, 8):

    * generator active power is dispatchable in ``[0, 3 x nameplate]`` (at least 100 MW);
    * generator reactive power is dispatchable in ``[-500, +500]`` Mvar;
    * the external grid is dispatchable in ``[-3000, +3000]`` MW / Mvar;
    * every generator and the external grid carry a unit linear cost, so the OPF
      minimises total dispatch;
    * the connection points are firm, i.e. not curtailable by the optimiser.
    """
    import numpy as np
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.create_cigre_network_hv()
    # The benchmark's own loading is halved, as in the original study.
    for i in range(5):
        net.load.at[i, "p_mw"] /= 2.0
        net.load.at[i, "q_mvar"] /= 2.0
    # Three zero-power loads mark the points of connection under study.
    for bus in (6, 7, 8):
        pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"POC_{bus}")
    if "controllable" in net.load.columns:
        net.load["controllable"] = False

    # --- make the network OPF-ready (see the docstring for the assumptions) ---
    for i in net.gen.index:
        nameplate = float(net.gen.at[i, "p_mw"])
        net.gen.at[i, "min_p_mw"] = 0.0
        net.gen.at[i, "max_p_mw"] = max(3.0 * nameplate, 100.0)
        net.gen.at[i, "min_q_mvar"] = -500.0
        net.gen.at[i, "max_q_mvar"] = 500.0
        net.gen.at[i, "controllable"] = True
        pp.create_poly_cost(net, i, "gen", cp1_eur_per_mw=1.0)
    for column, value in (("min_p_mw", -3000.0), ("max_p_mw", 3000.0),
                          ("min_q_mvar", -3000.0), ("max_q_mvar", 3000.0)):
        net.ext_grid[column] = value
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=1.0)

    n_bus = len(net.bus)
    base_p = np.zeros(n_bus)
    base_q = np.zeros(n_bus)
    for row in net.load.itertuples():
        base_p[int(row.bus)] += float(row.p_mw)
        base_q[int(row.bus)] += float(row.q_mvar)

    from OpenRC.cases.pandapower_adapter import _slack_setpoint

    if slack_vm_pu is None:
        resolved_slack = 1.0
    elif isinstance(slack_vm_pu, str):
        if slack_vm_pu != "setpoint":
            raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")
        setpoint = _slack_setpoint(net)
        resolved_slack = 1.0 if setpoint is None else setpoint
    elif isinstance(slack_vm_pu, (bool, np.bool_)):
        raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")
    else:
        resolved_slack = float(slack_vm_pu)
        if not (np.isfinite(resolved_slack) and resolved_slack > 0.0):
            raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")

    return GridCase(
        name="cigre-hv",
        net=net,
        base_p_mw=base_p,
        base_q_mvar=base_q,
        v_nom_kv=float(net.bus.vn_kv.max()),
        vm_min_pu=vm_min_pu,
        vm_max_pu=vm_max_pu,
        load_scale=load_scale,
        axis_presets={"026": [0, 2, 6]} if axis_presets is None else {k: list(v) for k, v in axis_presets.items()},
        description=("CIGRE HV benchmark - meshed, with transformers. DistFlow "
                     "oracles refuse it; use 'dcopf'/'acopf' (or 'acpf'/'dcpf' "
                     "for a fixed dispatch)."),
        metadata={"source": "pandapower.networks.create_cigre_network_hv",
                  "opf_ready": True},
        slack_vm_pu=resolved_slack,
        line_charging=line_charging,
    )


CASES.add(
    "cigre-hv",
    _cigre_hv,
    doc="CIGRE HV benchmark, meshed with transformers - use the OPF oracles.",
)
