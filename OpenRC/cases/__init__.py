"""Network test cases, pandapower adapters, and region-axis specifications.

Typical use::

    from OpenRC.cases import get_case

    case = get_case("9bus")
    axes = case.resolve_axes("247")      # -> RegionAxes([b1p, b3p, b6p])
"""

from OpenRC.cases.axes import RegionAxes, RegionAxis
from OpenRC.cases.base import GridCase
from OpenRC.cases.builtin import CASES, get_case, list_cases
from OpenRC.cases.pandapower_adapter import (
    ConversionReport,
    case_from_pandapower,
    list_feeders,
)
from OpenRC.cases.pp_networks import (
    PP_PREFIX,
    classify_pandapower_networks,
    list_pandapower_networks,
)
from OpenRC.cases.tabular import build_case

__all__ = [
    "CASES",
    "ConversionReport",
    "PP_PREFIX",
    "case_from_pandapower",
    "classify_pandapower_networks",
    "list_feeders",
    "list_pandapower_networks",
    "GridCase",
    "RegionAxes",
    "RegionAxis",
    "build_case",
    "get_case",
    "list_cases",
]
