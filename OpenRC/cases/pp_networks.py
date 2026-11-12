"""Every network built into ``pandapower``, available by name.

``pandapower.networks`` ships around sixty ready-made systems - IEEE and MATPOWER
transmission cases, the CIGRE benchmarks, the Kerber LV feeder library, the IEEE
European LV test feeder, and more.  This module registers all of them, so any of
them can be used exactly like a curated OpenRC case:

    orc.get_case("pp:case118")
    orc.get_case("pp:create_kerber_dorfnetz")
    orc.get_case("pp:create_cigre_network_mv", network_kwargs={"with_der": "pv_wind"})
    orc.list_pandapower_networks()          # what is there, and what applies to it

Registration is lazy: networks are constructed and transformers collapsed only
on first :func:`~OpenRC.cases.get_case` call, then memoised.

Oracle compatibility is determined by :func:`~OpenRC.powerflow.topology.radiality_report`
and :func:`~OpenRC.powerflow.opf.opf_ready`.  Meshed transmission cases are
supported by the AC/DC and OPF oracles, while radial distribution feeders (once
their substation transformer is collapsed) are also supported by the DistFlow
family.  :func:`classify_pandapower_networks` constructs and inspects each
network on demand.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from OpenRC.cases.base import GridCase
from OpenRC.utils import LOGGER

#: Prefix that distinguishes a network imported from ``pandapower.networks``
#: from a case curated by OpenRC.
PP_PREFIX = "pp:"

#: Non-case callables in ``pandapower.networks`` excluded from automatic registration.
_NOT_NETWORKS = {
    "create_empty_network",
    "create_dickert_lv_feeders",
}


def _network_factories() -> Dict[str, Callable]:
    """Zero-argument factories defined in ``pandapower.networks`` itself.

    Filtering on ``__module__`` keeps out the helpers that the package
    re-exports from elsewhere, and the required-argument check keeps out the
    builders that need a network to work on.
    """
    import pandapower.networks as pn

    out: Dict[str, Callable] = {}
    for name in dir(pn):
        if name.startswith("_") or name in _NOT_NETWORKS:
            continue
        function = getattr(pn, name)
        if not callable(function):
            continue
        if not (getattr(function, "__module__", "") or "").startswith("pandapower.networks"):
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            continue
        required = [p for p in signature.parameters.values()
                    if p.default is p.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if required:
            continue
        out[name] = function
    return out


def _first_line(text: Optional[str]) -> str:
    """First meaningful line of a docstring, for the registry listing."""
    if not text:
        return ""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            return line[:110]
    return ""


def register_pandapower_networks(registry=None) -> List[str]:
    """Register every built-in network under ``pp:<factory name>``.

    Called once at import time by :mod:`OpenRC.cases.builtin`.  Returns the
    list of newly registered case keys.
    """
    from OpenRC.cases.builtin import CASES

    registry = CASES if registry is None else registry
    registered: List[str] = []

    for factory_name, factory in _network_factories().items():
        case_name = f"{PP_PREFIX}{factory_name}"
        if case_name in registry:
            continue

        def _make(_factory=factory, _case_name=case_name, network_kwargs=None,
                  **overrides) -> GridCase:
            from OpenRC.cases.pandapower_adapter import case_from_pandapower

            if network_kwargs is not None and not isinstance(network_kwargs, dict):
                raise TypeError(
                    f"network_kwargs must be a dict, got {type(network_kwargs).__name__}"
                )
            # ``network_kwargs`` go to the pandapower factory (for instance
            # ``{"with_der": "pv_wind"}`` for the CIGRE MV feeder); every other
            # option goes to the adapter.
            return case_from_pandapower(_factory(**dict(network_kwargs or {})),
                                        name=_case_name, **overrides)

        doc = _first_line(factory.__doc__) or f"pandapower.networks.{factory_name}"
        registry.add(case_name, _make, doc=doc)
        registered.append(case_name)

    LOGGER.debug("registered %d pandapower networks", len(registered))
    return registered


def list_pandapower_networks() -> List[Tuple[str, str]]:
    """``[(case name, one-line description), ...]`` for the built-in networks."""
    from OpenRC.cases.builtin import CASES

    return [(name, doc) for name, doc in CASES.describe() if name.startswith(PP_PREFIX)]


def _oracle_families(radial: bool, opf: bool) -> str:
    """The oracle families that apply: DistFlow needs a tree, OPF needs costs."""
    power_flow = "AC/DC/OPF" if opf else "AC/DC"
    return f"DistFlow + {power_flow}" if radial else f"{power_flow} only"


def classify_pandapower_networks(names: Optional[List[str]] = None) -> "pd.DataFrame":
    """Build every registered network and report what each one is.

    Returns a ``pandas.DataFrame`` with one row per network: bus and branch
    counts, whether it is radial after transformer collapse, how the infeed was
    found, and which oracle families apply (OPF only where the network carries
    generator costs and limits, see :func:`~OpenRC.powerflow.opf.opf_ready`).

    Instantiates every requested network (several minutes for the full library);
    pass ``names`` to inspect a subset.
    """
    import pandas as pd

    from OpenRC.cases.builtin import get_case
    from OpenRC.powerflow.opf import opf_ready

    if names is None:
        names = [name for name, _ in list_pandapower_networks()]
    elif isinstance(names, (str, bytes, bytearray)):
        raise TypeError("names must be a sequence of case names, not str or bytes")

    rows = []
    for name in names:
        row: Dict[str, Any] = {"case": name}
        try:
            case = get_case(name)
            report = case.radiality
            raw = case.metadata.get("conversion")
            conversion: Dict[str, Any] = raw if isinstance(raw, dict) else {}
            # Multi-infeed networks have no unique radial slack_bus; record infeed count instead.
            try:
                slack, slack_source = case.slack_bus, case.slack_source
            except ValueError:
                slack, slack_source = None, f"{report.n_infeeds} infeeds"
            row.update(
                buses=case.n_bus,
                branches=report.n_branch,
                loops=report.n_loops,
                components=report.n_components,
                trafo_collapsed=conversion.get("n_trafo_collapsed", 0),
                radial=report.is_radial,
                infeeds=report.n_infeeds,
                oracles=_oracle_families(report.is_radial, opf_ready(case.net)),
                slack=slack,
                slack_source=slack_source,
                error="",
            )
        except Exception as exc:                      # a network OpenRC cannot import
            row.update(buses=None, branches=None, loops=None, components=None,
                       trafo_collapsed=None, radial=None, infeeds=None, oracles="",
                       slack=None, slack_source="",
                       error=f"{type(exc).__name__}: {str(exc)[:80]}")
        rows.append(row)
    return pd.DataFrame(rows)
