"""Build a :class:`~OpenRC.cases.base.GridCase` from four CSV/JSON files.

Each tabular case directory under ``OpenRC/cases/data/<name>/`` contains
``bus.csv``, ``line.csv``, ``load.csv``, and ``meta.json`` and is registered via
:mod:`OpenRC.cases.builtin`.

Table schema
------------
``bus.csv``   ``bus, name, vn_kv, vm_min_pu, vm_max_pu, is_slack``
``line.csv``  ``line, name, from_bus, to_bus, length_km, r_ohm_per_km,
              x_ohm_per_km, c_nf_per_km, max_i_ka, std_type``
``load.csv``  ``bus, name, p_mw, q_mvar``   (several rows per bus are summed)
``meta.json`` ``{"v_nom_kv": ..., "poc_presets": {...}, "notes": ...}``

Line parameters are specified explicitly per branch (no ``pandapower``
``std_type`` lookup required).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from OpenRC.cases.base import GridCase


def load_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load ``bus.csv``, ``line.csv``, ``load.csv``, and ``meta.json`` from ``data_dir``."""
    data_dir = Path(data_dir)
    required = ["bus.csv", "line.csv", "load.csv", "meta.json"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{data_dir} is missing {', '.join(missing)}")
    bus = pd.read_csv(data_dir / "bus.csv")
    line = pd.read_csv(data_dir / "line.csv")
    load = pd.read_csv(data_dir / "load.csv")
    meta = json.loads((data_dir / "meta.json").read_text())
    return bus, line, load, meta


def _num(cell: Any) -> float:
    """A CSV cell as a float (pandas types cells as a union of scalar kinds)."""
    return float(cell)


def _int(cell: Any) -> int:
    """A CSV cell as an integer index."""
    return int(cell)


def build_case(
    name: str,
    data_dir: Path,
    *,
    axis_presets: Optional[Dict[str, List[int]]] = None,
    description: str = "",
    load_scale: float = 1.0,
    v_nom_kv_override: Optional[float] = None,
    vm_min_pu: Optional[float] = None,
    vm_max_pu: Optional[float] = None,
    slack_vm_pu: Optional[Any] = 1.0,
    line_charging: str = "omit",
) -> GridCase:
    """Assemble a :class:`GridCase` (and its ``pandapower`` net) from a directory.

    Parameters
    ----------
    name:
        Registry key for the case.
    data_dir:
        Directory holding the four tables.
    axis_presets:
        Overrides ``meta.json``'s ``poc_presets`` when given.
    load_scale:
        Proportional scaling of every recorded load and static generator.
        See :class:`GridCase`.
    v_nom_kv_override:
        Optional solver base voltage, for reproducing a model whose base
        voltage differs from the bus table.
    """
    import pandapower as pp

    bus_t, line_t, load_t, meta = load_tables(data_dir)

    net = pp.create_empty_network(name=name)

    # --- buses: the CSV row order defines the bus index, so insert in order ---
    bus_t = bus_t.sort_values("bus").reset_index(drop=True)
    expected = np.arange(len(bus_t))
    if not np.array_equal(bus_t["bus"].to_numpy(), expected):
        raise ValueError(
            f"{data_dir}/bus.csv must use contiguous 0-based indices 0..{len(bus_t) - 1}"
        )
    for row in bus_t.itertuples(index=False):
        pp.create_bus(
            net,
            vn_kv=_num(row.vn_kv),
            name=str(row.name),
            min_vm_pu=_num(row.vm_min_pu),
            max_vm_pu=_num(row.vm_max_pu),
        )

    # --- slack ------------------------------------------------------------
    slack_rows = bus_t.index[bus_t["is_slack"].astype(bool)].tolist()
    if len(slack_rows) != 1:
        raise ValueError(f"{data_dir}/bus.csv marks {len(slack_rows)} slack buses, expected 1")
    if slack_vm_pu is None:
        resolved_slack = 1.0
    elif isinstance(slack_vm_pu, str):
        # 'setpoint' names the external grid's voltage, which this builder sets.
        if slack_vm_pu != "setpoint":
            raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")
        resolved_slack = 1.0
    elif isinstance(slack_vm_pu, (bool, np.bool_)):
        raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")
    else:
        resolved_slack = float(slack_vm_pu)
        if not (np.isfinite(resolved_slack) and resolved_slack > 0.0):
            raise ValueError("slack_vm_pu must be None, 'setpoint' or a positive number")
    # The AC/DC oracles read the external grid; hold it where the branch-flow
    # models hold the slack, so both oracle families see one operating point.
    pp.create_ext_grid(net, bus=int(slack_rows[0]), vm_pu=resolved_slack,
                       name="Grid Connection")

    # --- branches: explicit r/x/i_max, no std_type registry needed ---------
    for row in line_t.sort_values("line").itertuples(index=False):
        raw_c = getattr(row, "c_nf_per_km", 0.0)
        c_nf = 0.0 if raw_c is None or not np.isfinite(float(raw_c)) else _num(raw_c)
        pp.create_line_from_parameters(
            net,
            from_bus=_int(row.from_bus),
            to_bus=_int(row.to_bus),
            length_km=_num(row.length_km),
            r_ohm_per_km=_num(row.r_ohm_per_km),
            x_ohm_per_km=_num(row.x_ohm_per_km),
            c_nf_per_km=c_nf,
            max_i_ka=_num(row.max_i_ka),
            name=str(getattr(row, "name", "")),
        )
        # keep the human-readable cable name for reporting
        net.line.at[net.line.index[-1], "std_type"] = str(getattr(row, "std_type", ""))

    # --- loads ------------------------------------------------------------
    n_bus = len(bus_t)
    base_p = np.zeros(n_bus, dtype=float)
    base_q = np.zeros(n_bus, dtype=float)
    for row in load_t.itertuples(index=False):
        b = _int(row.bus)
        pp.create_load(net, bus=b, p_mw=_num(row.p_mw), q_mvar=_num(row.q_mvar),
                       name=str(getattr(row, "name", "")))
        base_p[b] += _num(row.p_mw)
        base_q[b] += _num(row.q_mvar)

    if v_nom_kv_override is not None and isinstance(v_nom_kv_override, (bool, np.bool_)):
        raise ValueError("v_nom_kv_override must be finite and strictly positive")
    presets = axis_presets if axis_presets is not None else meta.get("poc_presets", {})
    if not isinstance(presets, dict):
        raise TypeError(f"axis_presets must be a dict, got {type(presets).__name__}")
    for preset_key, preset_axes in presets.items():
        if isinstance(preset_axes, (str, bytes, bytearray)):
            raise TypeError(
                f"axis preset {preset_key!r} must be a sequence of axes, not str or bytes"
            )
    default_vm_min = float(bus_t.loc[~bus_t["is_slack"].astype(bool), "vm_min_pu"].max())
    default_vm_max = float(bus_t.loc[~bus_t["is_slack"].astype(bool), "vm_max_pu"].min())
    return GridCase(
        name=name,
        net=net,
        base_p_mw=base_p,
        base_q_mvar=base_q,
        v_nom_kv=(float(meta["v_nom_kv"]) if v_nom_kv_override is None
                  else float(v_nom_kv_override)),
        # The envelope is taken over non-slack buses: the slack is voltage-controlled
        # and its own limits are not an operating constraint.
        vm_min_pu=default_vm_min if vm_min_pu is None else vm_min_pu,
        vm_max_pu=default_vm_max if vm_max_pu is None else vm_max_pu,
        axis_presets={k: list(v) for k, v in presets.items()},
        description=description if isinstance(description, str) and description else str(meta.get("notes", "")) if description == "" else description,
        load_scale=load_scale,
        metadata={"data_dir": str(data_dir), "source": meta.get("source", "")},
        slack_vm_pu=resolved_slack,
        line_charging=line_charging,
    )
