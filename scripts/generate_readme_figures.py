#!/usr/bin/env python3
"""Regenerate the README figures from OpenRC itself.

Builds a deterministic 14-vertex security region on the 9-bus network
(LinDistFlow, axes ``247``), renders its 3-D convex hull, and writes the SVG
diagrams in ``docs/images/``.  Run from any working directory::

    python scripts/generate_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
sys.path.insert(0, str(ROOT))

import OpenRC as orc  # noqa: E402
from OpenRC.region.expansive import ExpansiveBuilder  # noqa: E402


WORKFLOW_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1212" viewBox="0 0 900 1212" role="img" aria-labelledby="title desc">
  <title id="title">OpenRC Architecture &amp; Feasibility Oracle Workflow</title>
  <desc id="desc">Vertical architectural workflow of OpenRC: Stage I (Case and POC Axes, covering curated built-in cases 9bus, 9bus-scaled, cigre-hv and primary tested pandapower cases pp:case33bw, pp:mv_oberrhein feeder=1, pp:create_cigre_network_mv, pp:case9, plus all 60 pandapower grids) feeds Stage II (Feasibility Oracle, a power-flow feasibility interface supporting LinDistFlow, SOCP, Exact Gurobi, and AC/DC OPF models with three-valued verdicts: FEASIBLE, INFEASIBLE, INCONCLUSIVE). Through a decoupled query-verdict loop, Stage III (Region Builder) constructs Stage IV (Convex Polytope), which then branches into Stage V (Compression, Validation &amp; Certification, and Downstream Linear Optimization Applications).</desc>

  <defs>
    <!-- Background coordinate grid -->
    <pattern id="bg-grid" width="36" height="36" patternUnits="userSpaceOnUse">
      <path d="M 36 0 L 0 0 0 36" fill="none" stroke="#e6e1d6" stroke-width="0.75"/>
    </pattern>

    <!-- Diagonal Hatch for Outer Nonlinear Envelope in Polytope Plot -->
    <pattern id="envelope-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="7" stroke="#c8963e" stroke-width="1.1" stroke-opacity="0.32"/>
    </pattern>

    <!-- Card drop shadow -->
    <filter id="card-shadow" x="-2%" y="-2%" width="104%" height="106%">
      <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#1e2530" flood-opacity="0.06"/>
    </filter>

    <style>
      @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&amp;family=JetBrains+Mono:wght@400;500;600;700&amp;family=Space+Grotesk:wght@400;500;600;700&amp;display=swap');

      /* Surface and frame styles */
      .canvas-bg { fill: #f7f5f0; }
      .outer-frame { fill: none; stroke: #2c3542; stroke-width: 1.5; }
      .inner-brass-rule { fill: none; stroke: #c5b69e; stroke-width: 0.9; }
      .divider-rule { stroke: #dcd6c9; stroke-width: 1; fill: none; }

      .card-base {
        fill: #fdfcf9;
        stroke: #c6bfb0;
        stroke-width: 1.2;
        filter: url(#card-shadow);
      }
      .card-oracle {
        fill: #fdfcf9;
        stroke: #b88c42;
        stroke-width: 1.5;
        filter: url(#card-shadow);
      }
      .subpanel {
        fill: #f6f4ee;
        stroke: #dcd6c8;
        stroke-width: 1;
      }

      /* Typography styles */
      .kicker {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.8px;
        font-weight: 600;
        letter-spacing: 0.1em;
        fill: #5c6573;
      }
      .stage-num {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.6px;
        font-weight: 700;
        letter-spacing: 0.08em;
      }
      .main-heading {
        font-family: "Instrument Serif", "Palatino Linotype", "Iowan Old Style", "Baskerville", Georgia, serif;
        font-size: 18.2px;
        font-weight: 400;
        letter-spacing: 0.005em;
        fill: #1a212b;
      }
      .serif-italic {
        font-family: "Instrument Serif", "Palatino Linotype", "Iowan Old Style", "Baskerville", Georgia, serif;
        font-style: italic;
      }
      .card-title {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 15.5px;
        font-weight: 700;
        letter-spacing: -0.01em;
        fill: #1a212b;
      }
      .card-title-sm {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: -0.01em;
        fill: #1a212b;
      }
      .module-tag {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 10px;
        font-weight: 500;
        fill: #5c6573;
      }
      .section-label {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.05em;
        fill: #2c3542;
      }
      .body-text {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 11.3px;
        font-weight: 400;
        fill: #39424e;
      }
      .body-sm {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 10.5px;
        font-weight: 400;
        fill: #495361;
      }
      .body-bold {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 11.3px;
        font-weight: 700;
        fill: #1a212b;
      }
      .mono-code {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 10px;
        font-weight: 600;
        fill: #1a212b;
      }
      .mono-sm {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.3px;
        font-weight: 500;
        fill: #39424e;
      }
      .verdict-badge {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.04em;
      }

      /* Accent palette */
      .c-slate     { fill: #1a212b; }
      .c-muted     { fill: #5c6573; }
      .c-verdigris { fill: #265951; }
      .c-amber     { fill: #8c5a0d; }
      .c-fjord     { fill: #1d3c63; }
      .c-brick     { fill: #7c2d2d; }
      .fw-bold     { font-weight: 700; }
    </style>
  </defs>

  <!-- ====================================================================
       0. BACKGROUND CANVAS & FRAME
       ==================================================================== -->
  <rect class="canvas-bg" x="0" y="0" width="900" height="1212" rx="6"/>
  <rect x="14" y="14" width="872" height="1184" fill="url(#bg-grid)"/>
  <rect class="outer-frame" x="14" y="14" width="872" height="1184" rx="4"/>
  <rect class="inner-brass-rule" x="18" y="18" width="864" height="1176" rx="2"/>

  <!-- ====================================================================
       1. HEADER BANNER (y = 18..108)
       ==================================================================== -->
  <rect x="19" y="19" width="862" height="88" fill="#fdfcf9"/>
  <line class="outer-frame" x1="18" y1="107" x2="882" y2="107"/>
  <line class="inner-brass-rule" x1="18" y1="110" x2="882" y2="110"/>

  <!-- Polytope emblem -->
  <g transform="translate(38, 36)">
    <circle cx="24" cy="24" r="22" fill="#f7f5f0" stroke="#2c3542" stroke-width="1.4"/>
    <circle cx="24" cy="24" r="18.5" fill="none" stroke="#b89c6d" stroke-width="0.8"/>
    <polygon points="24,8 37,15 39,29 29,39 16,38 9,26 13,13" fill="#edf2f8" stroke="#1d3c63" stroke-width="1.4"/>
    <line x1="24" y1="8"  x2="29" y2="39" stroke="#1d3c63" stroke-width="0.7" stroke-opacity="0.5"/>
    <line x1="13" y1="13" x2="39" y2="29" stroke="#1d3c63" stroke-width="0.7" stroke-opacity="0.5"/>
    <line x1="9"  y1="26" x2="37" y2="15" stroke="#1d3c63" stroke-width="0.7" stroke-opacity="0.5"/>
    <circle cx="24" cy="8"  r="2" fill="#8c5a0d"/>
    <circle cx="37" cy="15" r="2" fill="#8c5a0d"/>
    <circle cx="39" cy="29" r="2" fill="#8c5a0d"/>
    <circle cx="29" cy="39" r="2" fill="#8c5a0d"/>
    <circle cx="16" cy="38" r="2" fill="#8c5a0d"/>
    <circle cx="9"  cy="26" r="2" fill="#8c5a0d"/>
    <circle cx="13" cy="13" r="2" fill="#8c5a0d"/>
  </g>

  <text class="kicker" x="98" y="43">OPENRC ARCHITECTURE  ·  SECURITY REGION COMPUTATION LIBRARY</text>
  <text class="main-heading" x="98" y="69">
    Decoupling <tspan class="serif-italic c-amber">Power-Flow Physics</tspan> from <tspan class="serif-italic c-fjord">Computational Geometry</tspan> &amp; <tspan class="serif-italic c-brick">Mathematical Programming</tspan>
  </text>

  <!-- Subtle Discipline Legend Strip -->
  <g transform="translate(98, 81)">
    <circle cx="5" cy="7" r="3.5" fill="#265951"/>
    <text class="mono-sm c-slate" x="13" y="10.5">I &amp; V-B: Topology &amp; Audit</text>

    <circle cx="186" cy="7" r="3.5" fill="#8c5a0d"/>
    <text class="mono-sm c-slate" x="194" y="10.5">II: Feasibility Oracle (Physics)</text>

    <circle cx="404" cy="7" r="3.5" fill="#1d3c63"/>
    <text class="mono-sm c-slate" x="412" y="10.5">III &amp; IV: Computational Geometry</text>

    <circle cx="622" cy="7" r="3.5" fill="#7c2d2d"/>
    <text class="mono-sm c-slate" x="630" y="10.5">V-A &amp; V-C: Compression &amp; LP</text>
  </g>

  <!-- ====================================================================
       2. STAGE I: CASE & POC AXES (y = 124..288, height = 164)
       ==================================================================== -->
  <g id="stage-i-case" transform="translate(32, 124)">
    <rect class="card-base" x="0" y="0" width="836" height="164" rx="6"/>
    <!-- Top accent bar -->
    <rect x="0" y="0" width="836" height="4" rx="2" fill="#265951"/>

    <!-- Column 1: Stage Title & Mini Feeder Graph (x = 18..246) -->
    <text class="stage-num c-verdigris" x="18" y="23">STAGE I  ·  INPUT &amp; TOPOLOGY</text>
    <text class="card-title" x="18" y="43">Case + POC Axes</text>
    <text class="module-tag" x="154" y="42">OpenRC.cases</text>

    <rect class="subpanel" x="18" y="53" width="228" height="95" rx="4"/>
    <text class="mono-sm c-verdigris fw-bold" x="28" y="68">Feeder Graph &amp; Selected POCs</text>
    <!-- Feeder branches -->
    <line x1="42"  y1="100" x2="88"  y2="100" stroke="#2c3542" stroke-width="1.8"/>
    <line x1="88"  y1="100" x2="138" y2="84"  stroke="#2c3542" stroke-width="1.6"/>
    <line x1="88"  y1="100" x2="138" y2="116" stroke="#2c3542" stroke-width="1.6"/>
    <line x1="138" y1="84"  x2="188" y2="84"  stroke="#2c3542" stroke-width="1.6"/>
    <!-- Slack root -->
    <rect x="35" y="93" width="13" height="13" rx="1.5" fill="#2c3542"/>
    <text class="mono-sm c-muted" x="29" y="120">Slack</text>
    <!-- Bus nodes -->
    <circle cx="88"  cy="100" r="4" fill="#fdfcf9" stroke="#2c3542" stroke-width="1.6"/>
    <circle cx="138" cy="84"  r="5" fill="#c88a28" stroke="#2c3542" stroke-width="1.4"/>
    <circle cx="138" cy="116" r="5" fill="#c88a28" stroke="#2c3542" stroke-width="1.4"/>
    <circle cx="188" cy="84"  r="5" fill="#c88a28" stroke="#2c3542" stroke-width="1.4"/>
    <text class="mono-sm c-muted" x="28" y="139">0-based IDs</text>
    <text class="mono-sm c-slate fw-bold" x="236" y="139" text-anchor="end">Axes: Δ ∈ ℝᵈ</text>

    <!-- Vertical Divider 1 -->
    <line class="divider-rule" x1="260" y1="16" x2="260" y2="148"/>

    <!-- Column 2: Curated & Primary Test Benchmarks (x = 274..558) -->
    <text class="section-label c-verdigris" x="274" y="25">CURATED &amp; PRIMARY TEST BENCHMARKS</text>
    <text class="body-text" x="274" y="44">• <tspan class="body-bold">Curated Built-in:</tspan> <tspan class="mono-code">9bus</tspan> (3–16D core),</text>
    <text class="body-text" x="284" y="60"><tspan class="mono-code">9bus-scaled</tspan> (8D), <tspan class="mono-code">cigre-hv</tspan> (meshed OPF)</text>
    <text class="body-text" x="274" y="79">• <tspan class="body-bold">Primary Tested:</tspan> <tspan class="mono-code">pp:case33bw</tspan> (33-bus),</text>
    <text class="body-text" x="284" y="95"><tspan class="mono-code">pp:mv_oberrhein</tspan> (feeder=1), <tspan class="mono-code">pp:case9</tspan> (OPF),</text>
    <text class="body-text" x="284" y="111"><tspan class="mono-code">pp:create_cigre_network_mv</tspan> (load_scale=0.8)</text>
    <text class="body-text" x="274" y="130">• <tspan class="body-bold">Pandapower Adapter:</tspan> Any <tspan class="mono-code">pandapowerNet</tspan></text>
    <text class="body-text" x="284" y="146">or all 60 built-in grids via <tspan class="mono-code">pp:&lt;name&gt;</tspan></text>

    <!-- Vertical Divider 2 -->
    <line class="divider-rule" x1="572" y1="16" x2="572" y2="148"/>

    <!-- Column 3: Conventions & Isolation (x = 586..818) -->
    <text class="section-label c-verdigris" x="586" y="25">CONVENTIONS &amp; ISOLATION</text>
    <text class="body-text" x="586" y="46">• <tspan class="body-bold">SI / Grid Units:</tspan> <tspan class="mono-code">MW, Mvar, kV, kA</tspan></text>
    <text class="body-text" x="586" y="67">• <tspan class="body-bold">Sign Rule:</tspan> Load <tspan class="mono-code">&gt; 0</tspan>, export <tspan class="mono-code">&lt; 0</tspan></text>
    <text class="body-text" x="586" y="88">• <tspan class="body-bold">POC Axes:</tspan> <tspan class="mono-code">(bus, p/q)</tspan> e.g. <tspan class="mono-code">"247"</tspan>, <tspan class="mono-code">"1p,1q"</tspan></text>
    <text class="body-text" x="586" y="109">• <tspan class="body-bold">Root Inference:</tspan> Auto slack (<tspan class="mono-code">slack_source</tspan>)</text>

    <rect x="586" y="122" width="232" height="26" rx="4" fill="#eef5f3" stroke="#9ec2bc" stroke-width="1"/>
    <text class="mono-sm c-verdigris fw-bold" x="702" y="139" text-anchor="middle">Immutable GridCase Snapshot</text>
  </g>

  <!-- ====================================================================
       CONNECTOR I -> II (y = 288..320)
       ==================================================================== -->
  <g id="conn-i-to-ii">
    <line x1="450" y1="288" x2="450" y2="312" stroke="#2c3542" stroke-width="1.8"/>
    <polygon points="444.5,310 450,320 455.5,310" fill="#2c3542"/>
    <rect x="270" y="292" width="360" height="19" rx="9.5" fill="#f7f5f0"/>
    <text class="mono-sm c-muted" x="450" y="305.5" text-anchor="middle">
      Base nodal injections <tspan class="mono-code">(p₀, q₀)</tspan>, topology &amp; voltage/thermal limits
    </text>
  </g>

  <!-- ====================================================================
       3. STAGE II: FEASIBILITY ORACLE (y = 320..628, height = 308)
       ==================================================================== -->
  <g id="stage-ii-oracle" transform="translate(32, 320)">
    <rect class="card-oracle" x="0" y="0" width="836" height="308" rx="6"/>
    <!-- Top accent bar -->
    <rect x="0" y="0" width="836" height="4.5" rx="2" fill="#946210"/>

    <!-- Header Row -->
    <text class="stage-num c-amber" x="18" y="23">STAGE II  ·  POWER-FLOW FEASIBILITY INTERFACE</text>
    <text class="card-title" x="18" y="44">Feasibility Oracle</text>
    <text class="module-tag" x="164" y="43">OpenRC.powerflow · FeasibilityOracle</text>
    <text class="mono-sm c-amber fw-bold" x="818" y="43" text-anchor="end">orc.get_oracle(name, case)</text>

    <!-- Core Question Callout Banner (2-line balanced layout) -->
    <rect x="18" y="53" width="800" height="55" rx="4" fill="#fbf6eb" stroke="#dec393" stroke-width="1.2"/>
    <rect x="18" y="53" width="4.5" height="55" rx="1" fill="#946210"/>
    <text class="section-label c-amber" x="32" y="69">ORACLE QUERY SPECIFICATION:</text>
    <text class="body-text c-slate" x="32" y="86">
      “Given a full nodal power injection vector <tspan class="mono-code">(P, Q) = (p₀, q₀) + B·Δ</tspan> for region coordinate <tspan class="mono-code">Δ ∈ ℝᵈ</tspan>,
    </text>
    <text class="body-text c-slate" x="32" y="101">
      can the network accommodate it within all <tspan class="body-bold">voltage [v_min, v_max]</tspan> and <tspan class="body-bold">thermal I_max</tspan> limits?”
    </text>

    <!-- 2x2 Grid of Pluggable Power-Flow & OPF Models -->
    <text class="section-label" x="18" y="126">FOUR PLUGGABLE POWER-FLOW &amp; OPTIMAL POWER-FLOW MODEL FAMILIES</text>

    <!-- Model 1 (Top-Left): Linear & Conservative -->
    <g transform="translate(18, 134)">
      <rect class="subpanel" x="0" y="0" width="392" height="54" rx="4"/>
      <rect x="0" y="0" width="4" height="54" rx="1" fill="#1d3c63"/>
      <text class="body-bold" x="12" y="17">1. Linear &amp; Paired Inner Models</text>
      <text class="mono-sm c-fjord fw-bold" x="382" y="17" text-anchor="end">~25 µs · analytic</text>
      <text class="mono-code" x="12" y="33" style="font-size:8.9px;">[lindistflow ∩ lindistflow-vdrop] · [tight-voltage ∩ tight-flow]</text>
      <text class="body-sm" x="12" y="47">Standalone <tspan class="mono-sm">lindistflow</tspan> or paired 2×2 for inner approximations (no solver)</text>
    </g>

    <!-- Model 2 (Top-Right): SOCP Outer Relaxation -->
    <g transform="translate(426, 134)">
      <rect class="subpanel" x="0" y="0" width="392" height="54" rx="4"/>
      <rect x="0" y="0" width="4" height="54" rx="1" fill="#946210"/>
      <text class="body-bold" x="12" y="17">2. Conic Outer Relaxation (<tspan class="mono-code">socp</tspan>)</text>
      <text class="mono-sm c-amber fw-bold" x="382" y="17" text-anchor="end">Pyomo · MOSEK · Gurobi</text>
      <text class="mono-code" x="12" y="33" style="font-size:9.4px;">Second-Order Cone Branch-Flow Relaxation (Outer Bound)</text>
      <text class="body-sm" x="12" y="47">Fast outer-relaxation screening; certifies global infeasibility</text>
    </g>

    <!-- Model 3 (Bottom-Left): Global Non-Convex Exact -->
    <g transform="translate(18, 196)">
      <rect class="subpanel" x="0" y="0" width="392" height="54" rx="4"/>
      <rect x="0" y="0" width="4" height="54" rx="1" fill="#7c2d2d"/>
      <text class="body-bold" x="12" y="17">3. Global Non-Convex Exact (<tspan class="mono-code">exact</tspan>)</text>
      <text class="mono-sm c-brick fw-bold" x="382" y="17" text-anchor="end">Gurobi NonConvex=2</text>
      <text class="mono-code" x="12" y="33" style="font-size:9.4px;">Global Branch-Flow Adjudicator + Interval RadialSweep</text>
      <text class="body-sm" x="12" y="47">Certifies feasibility via Banach fixed-point interval enclosures</text>
    </g>

    <!-- Model 4 (Bottom-Right): AC/DC & OPF -->
    <g transform="translate(426, 196)">
      <rect class="subpanel" x="0" y="0" width="392" height="54" rx="4"/>
      <rect x="0" y="0" width="4" height="54" rx="1" fill="#265951"/>
      <text class="body-bold" x="12" y="17">4. Full AC/DC &amp; Optimal Power Flow</text>
      <text class="mono-sm c-verdigris fw-bold" x="382" y="17" text-anchor="end">pandapower · meshed OK</text>
      <text class="mono-code" x="12" y="33" style="font-size:9.4px;">acpf · dcpf (fixed dispatch)  |  acopf · dcopf (dispatchable)</text>
      <text class="body-sm" x="12" y="47">No radiality requirement; supports meshed grids &amp; generator re-dispatch</text>
    </g>

    <!-- Three-Valued Verdict Bar -->
    <line class="divider-rule" x1="18" y1="260" x2="818" y2="260"/>
    <text class="section-label" x="18" y="280">THREE-VALUED VERDICT:</text>

    <!-- Badge 1: FEASIBLE -->
    <rect x="166" y="267" width="102" height="22" rx="4" fill="#eef5f3" stroke="#265951" stroke-width="1.2"/>
    <circle cx="178" cy="278" r="3.5" fill="#265951"/>
    <text class="verdict-badge c-verdigris" x="187" y="281.5">FEASIBLE</text>

    <!-- Badge 2: INFEASIBLE -->
    <rect x="276" y="267" width="112" height="22" rx="4" fill="#faf0f0" stroke="#7c2d2d" stroke-width="1.2"/>
    <circle cx="288" cy="278" r="3.5" fill="#7c2d2d"/>
    <text class="verdict-badge c-brick" x="297" y="281.5">INFEASIBLE</text>

    <!-- Badge 3: INCONCLUSIVE -->
    <rect x="396" y="267" width="122" height="22" rx="4" fill="#fbf6eb" stroke="#946210" stroke-width="1.2"/>
    <circle cx="408" cy="278" r="3.5" fill="#946210"/>
    <text class="verdict-badge c-amber" x="417" y="281.5">INCONCLUSIVE</text>

    <!-- Verdict Explanation on Right -->
    <text class="body-sm c-slate" x="530" y="276">
      <tspan class="body-bold" style="font-size:10.4px;">Conservative rule:</tspan> <tspan class="mono-sm">INCONCLUSIVE</tspan> (non-convergence)
    </text>
    <text class="body-sm c-slate" x="530" y="290">
      stays <tspan class="body-bold" style="font-size:10.4px;">outside</tspan> during growth &amp; escalates in validation.
    </text>
  </g>

  <!-- ====================================================================
       4. CLOSED-LOOP QUERY–VERDICT PROTOCOL (y = 628..686, height = 58)
       ==================================================================== -->
  <g id="conn-ii-iii-loop">
    <!-- Left Upward Query Arrow: Builder -> Oracle -->
    <line x1="295" y1="686" x2="295" y2="637" stroke="#1d3c63" stroke-width="2"/>
    <polygon points="289.5,639 295,628 300.5,639" fill="#1d3c63"/>
    <rect x="92" y="643" width="192" height="28" rx="4" fill="#edf2f8" stroke="#a6bedb" stroke-width="1.1"/>
    <text class="mono-sm c-fjord fw-bold" x="188" y="660" text-anchor="middle">↑ Query Δ ∈ ℝᵈ (ray / seed)</text>

    <!-- Center Decoupled Pill Badge -->
    <rect x="366" y="644" width="168" height="26" rx="13" fill="#2c3542"/>
    <text class="kicker" x="450" y="660.5" text-anchor="middle" style="fill:#fdfcf9;font-size:9.2px;">DECOUPLED INTERFACE</text>

    <!-- Right Downward Verdict Arrow: Oracle -> Builder -->
    <line x1="605" y1="628" x2="605" y2="677" stroke="#946210" stroke-width="2"/>
    <polygon points="599.5,675 605,686 610.5,675" fill="#946210"/>
    <rect x="616" y="643" width="206" height="28" rx="4" fill="#fbf6eb" stroke="#dec393" stroke-width="1.1"/>
    <text class="mono-sm c-amber fw-bold" x="719" y="660" text-anchor="middle">↓ Verdict + Signed Margin</text>
  </g>

  <!-- ====================================================================
       5. ROW 3: STAGE III (BUILDER) -> STAGE IV (POLYTOPE) (y = 686..916, height = 230)
       ==================================================================== -->
  <!-- STAGE III: REGION BUILDER (Left Card, x = 32..502, width = 470) -->
  <g id="stage-iii-builder" transform="translate(32, 686)">
    <rect class="card-base" x="0" y="0" width="470" height="230" rx="6"/>
    <rect x="0" y="0" width="470" height="4" rx="2" fill="#1d3c63"/>

    <text class="stage-num c-fjord" x="18" y="23">STAGE III  ·  POLYTOPE CONSTRUCTION</text>
    <text class="card-title" x="18" y="44">Region Builder</text>
    <text class="module-tag" x="142" y="43">OpenRC.region</text>

    <rect x="304" y="28" width="148" height="21" rx="4" fill="#edf2f8" stroke="#a6bedb" stroke-width="1"/>
    <text class="mono-sm c-fjord fw-bold" x="378" y="42" text-anchor="middle" style="font-size:8.8px;">Queries Oracle Only</text>

    <line class="divider-rule" x1="18" y1="54" x2="452" y2="54"/>

    <!-- Left Subcolumn: Expansive Growth -->
    <text class="section-label c-fjord" x="18" y="72">EXPANSIVE GROWTH</text>
    <text class="body-text" x="18" y="91">• <tspan class="body-bold">Seeds:</tspan> <tspan class="mono-code">simplex</tspan>, <tspan class="mono-code">axis</tspan>, <tspan class="mono-code">random</tspan></text>
    <text class="body-text" x="18" y="110">• <tspan class="body-bold">Facet Policies:</tspan></text>
    <text class="mono-sm c-slate" x="28" y="126">volume-gain · facet-area</text>
    <text class="mono-sm c-slate" x="28" y="141">adaptive-gradient · adaptive-ema</text>
    <text class="body-text" x="18" y="160">• <tspan class="body-bold">Boundary Search:</tspan> ray bisection</text>
    <text class="body-text" x="28" y="176">or signed-margin interpolation</text>

    <!-- Vertical Divider inside Builder -->
    <line class="divider-rule" x1="236" y1="62" x2="236" y2="184"/>

    <!-- Right Subcolumn: Conservative Inner Methods -->
    <text class="section-label c-fjord" x="250" y="72">PAIRWISE INNER APPROXIMATIONS</text>
    <rect class="subpanel" x="250" y="80" width="202" height="46" rx="4"/>
    <text class="body-bold c-fjord" x="258" y="96">Method 1: <tspan class="mono-code">intersect-shift</tspan></text>
    <text class="mono-sm c-slate" x="258" y="115" style="font-size:7.6px;">(lindistflow ∩ lindistflow-vdrop) ⊖ B_loss</text>

    <rect class="subpanel" x="250" y="132" width="202" height="46" rx="4"/>
    <text class="body-bold c-fjord" x="258" y="148">Method 2: <tspan class="mono-code">tight-intersect</tspan></text>
    <text class="mono-sm c-slate" x="258" y="167" style="font-size:8.8px;">tight-voltage ∩ tight-flow (direct)</text>

    <!-- Bottom Budget Strip -->
    <line class="divider-rule" x1="18" y1="191" x2="452" y2="191"/>
    <text class="mono-sm c-muted" x="18" y="211" style="font-size:8.8px;">
      <tspan class="c-slate fw-bold">Budgets:</tspan> max_vertices · max_oracle_calls · memory_limit_mb · n_workers
    </text>
  </g>

  <!-- Horizontal Connector III -> IV (x = 502..534) -->
  <g id="conn-iii-to-iv">
    <line x1="502" y1="801" x2="525" y2="801" stroke="#1d3c63" stroke-width="2"/>
    <polygon points="523,795.5 534,801 523,806.5" fill="#1d3c63"/>
  </g>

  <!-- STAGE IV: CONVEX POLYTOPE (Right Card, x = 534..868, width = 334) -->
  <g id="stage-iv-polytope" transform="translate(534, 686)">
    <rect class="card-base" x="0" y="0" width="334" height="230" rx="6"/>
    <rect x="0" y="0" width="334" height="4" rx="2" fill="#1d3c63"/>

    <text class="stage-num c-fjord" x="16" y="23">STAGE IV  ·  SECURITY REGION</text>
    <text class="card-title" x="16" y="44">Convex Polytope</text>
    <text class="module-tag" x="152" y="43">OpenRC.geometry</text>

    <line class="divider-rule" x1="16" y1="54" x2="318" y2="54"/>

    <!-- Left Mini-Plot: 2-D Polytope Diagram (x = 16..160, y = 64..212) -->
    <rect x="16" y="64" width="144" height="148" rx="4" fill="#141e2e" stroke="#2c3542" stroke-width="1.2"/>
    <!-- Coordinate axes -->
    <line x1="16" y1="140" x2="160" y2="140" stroke="#283852" stroke-width="1"/>
    <line x1="86" y1="64"  x2="86"  y2="212" stroke="#283852" stroke-width="1"/>
    <text class="mono-sm" x="156" y="135" text-anchor="end" style="font-size:7.6px;fill:#8ea0b8;">ΔP₁</text>
    <text class="mono-sm" x="90"  y="75"  style="font-size:7.6px;fill:#8ea0b8;">ΔP₂</text>
    <text class="mono-sm" x="21"  y="76"  style="font-size:7.8px;fill:#e8be68;font-weight:700;">ℛ ⊂ 𝒞_exact</text>

    <!-- Outer Nonlinear Boundary (dashed brass, kept clear of ΔP1 label) -->
    <polygon points="86,82 128,97 132,145 106,188 50,184 29,143 40,93"
             fill="url(#envelope-hatch)" stroke="#e8be68" stroke-width="1.2" stroke-dasharray="3 2"/>
    <!-- Inner Convex Polytope -->
    <polygon points="86,90 120,104 124,143 100,179 56,175 38,139 48,101"
             fill="#244b7a" fill-opacity="0.78" stroke="#7eb0f2" stroke-width="1.8"/>
    <!-- Ray Search Vector -->
    <line x1="86" y1="140" x2="111" y2="95" stroke="#e06c6c" stroke-width="1.4" stroke-dasharray="2.5 2"/>
    <polygon points="107.5,93.5 115,88 113.5,97.5" fill="#e06c6c"/>
    <!-- Vertices -->
    <circle cx="86"  cy="90"  r="2.6" fill="#e8be68" stroke="#141e2e" stroke-width="0.9"/>
    <circle cx="120" cy="104" r="2.6" fill="#e8be68" stroke="#141e2e" stroke-width="0.9"/>
    <circle cx="124" cy="143" r="2.6" fill="#e8be68" stroke="#141e2e" stroke-width="0.9"/>
    <circle cx="100" cy="179" r="2.6" fill="#e8be68" stroke="#141e2e" stroke-width="0.9"/>
    <circle cx="56"  cy="175" r="2.6" fill="#e8be68" stroke="#141e2e" stroke-width="0.9"/>
    <circle cx="38"  cy="139" r="2.6" fill="#e8be68" stroke="#141e2e" stroke-width="0.9"/>
    <circle cx="48"  cy="101" r="2.6" fill="#e8be68" stroke="#141e2e" stroke-width="0.9"/>
    <text class="mono-sm" x="82" y="155" text-anchor="middle" style="font-size:9px;fill:#f5f3ee;font-weight:700;">A Δ ≤ b</text>
    <text class="mono-sm" x="22" y="205" style="font-size:7.8px;fill:#e8be68;">● V-rep &amp; H-rep</text>

    <!-- Right Specs: Dual Geometry & Provenance (x = 172..320) -->
    <text class="section-label c-fjord" x="172" y="75">DUAL REPRESENTATION</text>
    <text class="body-text" x="172" y="93">• <tspan class="body-bold">V-rep:</tspan> Extreme vertices</text>
    <text class="body-text" x="172" y="110">• <tspan class="body-bold">H-rep:</tspan> Facets <tspan class="mono-code">A Δ ≤ b</tspan></text>
    <text class="body-text" x="172" y="127">• <tspan class="body-bold">Set Ops:</tspan> <tspan class="mono-sm">intersect</tspan>,</text>
    <text class="body-text" x="182" y="142">slice, Pontryagin erosion</text>

    <line class="divider-rule" x1="172" y1="152" x2="318" y2="152"/>
    <text class="section-label c-fjord" x="172" y="168">PROVENANCE SIDECAR</text>
    <text class="body-text" x="172" y="185">• Bit-exact <tspan class="mono-code">float64</tspan> <tspan class="mono-code">.csv</tspan></text>
    <text class="body-text" x="172" y="201">• SHA-256 <tspan class="mono-code">.json</tspan> sidecar</text>
    <text class="mono-sm c-fjord fw-bold" x="172" y="216">Polytope.load(path, axes)</text>
  </g>

  <!-- ====================================================================
       CONNECTOR IV -> V (MANIFOLD BRANCHING TO 3 DOWNSTREAM STAGES, y = 916..950)
       ==================================================================== -->
  <g id="conn-iv-to-v">
    <!-- Vertical stem from Polytope (x = 701) down to manifold bar (y = 930) -->
    <line x1="701" y1="916" x2="701" y2="930" stroke="#2c3542" stroke-width="1.8"/>
    <!-- Horizontal distribution bar across the 3 downstream columns (x = 164..736) -->
    <line x1="164" y1="930" x2="736" y2="930" stroke="#2c3542" stroke-width="1.8"/>
    <!-- Left branch down to V-A Compression (x = 164) -->
    <line x1="164" y1="930" x2="164" y2="942" stroke="#7c2d2d" stroke-width="1.8"/>
    <polygon points="158.5,940 164,950 169.5,940" fill="#7c2d2d"/>
    <!-- Center branch down to V-B Validation (x = 450) -->
    <line x1="450" y1="930" x2="450" y2="942" stroke="#265951" stroke-width="1.8"/>
    <polygon points="444.5,940 450,950 455.5,940" fill="#265951"/>
    <!-- Right branch down to V-C Applications (x = 736) -->
    <line x1="736" y1="930" x2="736" y2="942" stroke="#7c2d2d" stroke-width="1.8"/>
    <polygon points="730.5,940 736,950 741.5,940" fill="#7c2d2d"/>
  </g>

  <!-- ====================================================================
       6. ROW 4: STAGE V — THREE DOWNSTREAM MODULES (y = 950..1066, height = 116)
       ==================================================================== -->
  <!-- STAGE V-A: COMPRESSION (x = 32..296, width = 264) -->
  <g id="stage-va-compression" transform="translate(32, 950)">
    <rect class="card-base" x="0" y="0" width="264" height="116" rx="6"/>
    <rect x="0" y="0" width="264" height="4" rx="2" fill="#7c2d2d"/>
    <text class="stage-num c-brick" x="16" y="22">STAGE V-A  ·  COMPRESSION</text>
    <text class="card-title-sm" x="16" y="41">Vertex Pruning</text>
    <text class="module-tag" x="128" y="40">OpenRC.compression</text>
    <line class="divider-rule" x1="16" y1="49" x2="248" y2="49"/>
    <text class="body-text" x="16" y="67">• <tspan class="mono-code">local-search</tspan> · <tspan class="mono-code">hybrid-vns</tspan></text>
    <text class="body-text" x="16" y="85">• <tspan class="mono-code">star-greedy</tspan> &amp; budget schedules</text>
    <text class="body-text" x="16" y="103">• Compact inner subset <tspan class="mono-code">ℛ_small ⊆ ℛ</tspan></text>
  </g>

  <!-- STAGE V-B: VALIDATION & CERTIFICATION (x = 318..582, width = 264) -->
  <g id="stage-vb-validation" transform="translate(318, 950)">
    <rect class="card-base" x="0" y="0" width="264" height="116" rx="6"/>
    <rect x="0" y="0" width="264" height="4" rx="2" fill="#265951"/>
    <text class="stage-num c-verdigris" x="16" y="22">STAGE V-B  ·  VALIDATION</text>
    <text class="card-title-sm" x="16" y="41">Audit &amp; Inner Proof</text>
    <text class="module-tag" x="150" y="40">OpenRC.validation</text>
    <line class="divider-rule" x1="16" y1="49" x2="248" y2="49"/>
    <text class="body-text" x="16" y="67">• Loss / error / unresolved vs. <tspan class="mono-code">acpf</tspan></text>
    <text class="body-text" x="16" y="85">• Escalation: <tspan class="mono-code">socp</tspan> → <tspan class="mono-code">exact</tspan> Gurobi</text>
    <text class="body-text" x="16" y="103">• Certificate: <tspan class="mono-code">verify_inner_region()</tspan></text>
  </g>

  <!-- STAGE V-C: APPLICATIONS (x = 604..868, width = 264) -->
  <g id="stage-vc-applications" transform="translate(604, 950)">
    <rect class="card-base" x="0" y="0" width="264" height="116" rx="6"/>
    <rect x="0" y="0" width="264" height="4" rx="2" fill="#7c2d2d"/>
    <text class="stage-num c-brick" x="16" y="22">STAGE V-C  ·  APPLICATIONS</text>
    <text class="card-title-sm" x="16" y="41">Downstream LPs</text>
    <text class="module-tag" x="132" y="40">OpenRC.applications</text>
    <line class="divider-rule" x1="16" y1="49" x2="248" y2="49"/>
    <text class="body-text" x="16" y="67">• Replaces power flow with <tspan class="mono-code">A Δ ≤ b</tspan></text>
    <text class="body-text" x="16" y="85">• Grid-aware capacity allocation LP</text>
    <text class="body-text" x="16" y="103">• Optimal storage sizing without PF</text>
  </g>

  <!-- ====================================================================
       7. ARCHITECTURAL SYNTHESIS & MODULAR REGISTRY FOOTER (y = 1080..1194)
       ==================================================================== -->
  <line class="inner-brass-rule" x1="18" y1="1079" x2="882" y2="1079"/>
  <line class="outer-frame" x1="18" y1="1082" x2="882" y2="1082"/>
  <rect x="19" y="1083" width="862" height="110" fill="#fdfcf9"/>

  <!-- Two-Column Architectural Notes -->
  <text class="section-label c-amber" x="34" y="1101">ARCHITECTURAL PRINCIPLE I  ·  ORACLE DECOUPLING</text>
  <text class="body-sm c-slate" x="34" y="1117">
    Builders only call <tspan class="mono-sm fw-bold">oracle.feasible_at(axes, Δ)</tspan> or <tspan class="mono-sm fw-bold">verdict_at()</tspan>.
  </text>
  <text class="body-sm c-slate" x="34" y="1132">
    Analytic <tspan class="mono-sm">lindistflow</tspan>, <tspan class="mono-sm">socp</tspan>, <tspan class="mono-sm">exact</tspan>, and <tspan class="mono-sm">acopf</tspan> share the same
  </text>
  <text class="body-sm c-slate" x="34" y="1147">
    geometry, seeding, facet-selection, and ray-search implementations.
  </text>

  <line class="divider-rule" x1="450" y1="1090" x2="450" y2="1152"/>

  <text class="section-label c-fjord" x="466" y="1101">ARCHITECTURAL PRINCIPLE II  ·  THREE-VALUED ADJUDICATION</text>
  <text class="body-sm c-slate" x="466" y="1117">
    Numerical non-convergence returns <tspan class="mono-sm fw-bold">INCONCLUSIVE</tspan> rather than a false
  </text>
  <text class="body-sm c-slate" x="466" y="1132">
    infeasibility claim. Growth conservatively excludes undecided points;
  </text>
  <text class="body-sm c-slate" x="466" y="1147">
    validation escalates them through <tspan class="mono-sm">socp</tspan> screening and <tspan class="mono-sm">exact</tspan> global solves.
  </text>

  <!-- Bottom Registry Bar -->
  <line class="divider-rule" x1="34" y1="1157" x2="866" y2="1157"/>
  <text class="mono-sm c-muted fw-bold" x="34" y="1178">REGISTRIES (demo.py list):</text>

  <g transform="translate(204, 1164)">
    <rect x="0"   y="0" width="66" height="20" rx="3" fill="#eef5f3" stroke="#9ec2bc" stroke-width="1"/>
    <text class="mono-sm c-verdigris fw-bold" x="33" y="13.5" text-anchor="middle" style="font-size:8.6px;">CASES</text>

    <rect x="74"  y="0" width="74" height="20" rx="3" fill="#fbf6eb" stroke="#dec393" stroke-width="1"/>
    <text class="mono-sm c-amber fw-bold" x="111" y="13.5" text-anchor="middle" style="font-size:8.6px;">ORACLES</text>

    <rect x="156" y="0" width="78" height="20" rx="3" fill="#edf2f8" stroke="#a6bedb" stroke-width="1"/>
    <text class="mono-sm c-fjord fw-bold" x="195" y="13.5" text-anchor="middle" style="font-size:8.6px;">BUILDERS</text>

    <rect x="242" y="0" width="64" height="20" rx="3" fill="#edf2f8" stroke="#a6bedb" stroke-width="1"/>
    <text class="mono-sm c-fjord fw-bold" x="274" y="13.5" text-anchor="middle" style="font-size:8.6px;">SEEDS</text>

    <rect x="314" y="0" width="76" height="20" rx="3" fill="#edf2f8" stroke="#a6bedb" stroke-width="1"/>
    <text class="mono-sm c-fjord fw-bold" x="352" y="13.5" text-anchor="middle" style="font-size:8.6px;">POLICIES</text>

    <rect x="398" y="0" width="96" height="20" rx="3" fill="#faf0f0" stroke="#dfb4b4" stroke-width="1"/>
    <text class="mono-sm c-brick fw-bold" x="446" y="13.5" text-anchor="middle" style="font-size:8.6px;">COMPRESSORS</text>

    <rect x="502" y="0" width="76" height="20" rx="3" fill="#eef5f3" stroke="#9ec2bc" stroke-width="1"/>
    <text class="mono-sm c-verdigris fw-bold" x="540" y="13.5" text-anchor="middle" style="font-size:8.6px;">SAMPLERS</text>

    <rect x="586" y="0" width="76" height="20" rx="3" fill="#faf0f0" stroke="#dfb4b4" stroke-width="1"/>
    <text class="mono-sm c-brick fw-bold" x="624" y="13.5" text-anchor="middle" style="font-size:8.6px;">APPLICATIONS</text>
  </g>
</svg>
"""


def write_workflow() -> None:
    (IMAGES / "workflow.svg").write_text(WORKFLOW_SVG, encoding="utf-8")


def write_region() -> None:
    from matplotlib.patches import FancyBboxPatch

    case = orc.get_case("9bus")
    axes = case.resolve_axes("247")
    with orc.get_oracle("lindistflow", case) as oracle:
        result = ExpansiveBuilder(
            policy="volume-gain", ray_kwargs={"tol": 1.0e-5}, refresh_gains=False
        ).build(case, axes, oracle, max_vertices=14)
        oracle_calls = oracle.stats.n_calls
    region = result.polytope

    # Palette matching workflow.svg
    bg_canvas = "#f7f5f0"
    bg_pane = (0.992, 0.988, 0.976, 0.92)      # #fdfcf9
    frame_outer = "#2c3542"
    frame_brass = "#c5b69e"
    grid_line = "#e2ddd2"
    facet_fill = "#2d5685"
    facet_edge = "#1d3c63"
    vertex_fill = "#8c2d2d"
    base_fill = "#c88a28"
    text_ink = "#1a212b"
    text_muted = "#5c6573"

    plt.rcParams.update({
        "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans", "Arial"],
        "font.serif": ["Palatino", "Iowan Old Style", "Georgia", "DejaVu Serif"],
        "font.monospace": ["JetBrains Mono", "SFMono-Regular", "Menlo", "DejaVu Sans Mono"],
    })

    fig = plt.figure(figsize=(8.8, 6.5), facecolor=bg_canvas)
    try:
        ax: Any = fig.add_axes((0.06, 0.08, 0.88, 0.78), projection="3d", facecolor=bg_canvas)

        # Style 3D coordinate panes and grid lines
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(bg_pane)
            axis.pane.set_edgecolor("#dcd6c8")
            axis._axinfo["grid"]["color"] = grid_line
            axis._axinfo["grid"]["linewidth"] = 0.75
            axis._axinfo["grid"]["linestyle"] = "-"
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.15

        points = region.vertices
        for simplex in region.hull.simplices:
            triangle = points[simplex]
            ax.plot_trisurf(
                triangle[:, 0], triangle[:, 1], triangle[:, 2],
                color=facet_fill, edgecolor=facet_edge, linewidth=0.65,
                alpha=0.22, shade=False,
            )

        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=46, color=vertex_fill, edgecolor="#fdfcf9", linewidth=0.95,
            depthshade=False, zorder=10,
            label=f"Extreme vertices (V-rep: {region.n_vertices})",
        )
        ax.scatter(
            [0], [0], [0],
            s=95, marker="*", color=base_fill, edgecolor=text_ink, linewidth=0.7,
            depthshade=False, zorder=12,
            label="Base operating point (0, 0, 0)",
        )

        ax.set_xlabel(f"{axes.labels[0]} (MW)", labelpad=9, color=text_ink, fontsize=10, fontweight="medium")
        ax.set_ylabel(f"{axes.labels[1]} (MW)", labelpad=9, color=text_ink, fontsize=10, fontweight="medium")
        ax.set_zlabel(f"{axes.labels[2]} (MW)", labelpad=9, color=text_ink, fontsize=10, fontweight="medium")
        ax.tick_params(colors=text_muted, labelsize=8.8, pad=2)
        ax.view_init(elev=22, azim=42)

        # Double border on figure canvas
        fig.patches.extend([
            FancyBboxPatch(
                (0.015, 0.018), 0.97, 0.964,
                boxstyle="round,pad=0,rounding_size=0.006",
                transform=fig.transFigure, fill=False,
                edgecolor=frame_outer, linewidth=1.3, zorder=20,
            ),
            FancyBboxPatch(
                (0.020, 0.024), 0.96, 0.952,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=fig.transFigure, fill=False,
                edgecolor=frame_brass, linewidth=0.8, zorder=20,
            ),
        ])

        # Header kicker & serif title matching workflow.svg
        fig.text(
            0.05, 0.935,
            "STAGE IV  ·  SECURITY REGION POLYTOPE  ·  9BUS / POC AXES 247",
            fontsize=8.2, fontfamily="monospace", fontweight="bold", color=text_muted,
        )
        fig.text(
            0.05, 0.892,
            "Deterministic 14-Point Security Region — Convex Hull & Halfspace Facets",
            fontsize=13.5, fontfamily="serif", color=text_ink,
        )

        legend = ax.legend(
            loc="upper left", bbox_to_anchor=(0.01, 0.98),
            frameon=True, facecolor="#fdfcf9", edgecolor="#c6bfb0",
            fontsize=9.2, framealpha=0.95,
        )
        for text in legend.get_texts():
            text.set_color(text_ink)

        n_facets = region.n_facets
        fig.text(
            0.05, 0.045,
            f"volume = {region.volume:.6f} MW³   ·   {region.n_vertices} hull vertices   ·   "
            f"{n_facets} halfspace facets   ·   {oracle_calls} oracle calls (lindistflow)",
            fontsize=8.6, fontfamily="monospace", color=facet_edge, fontweight="medium",
        )

        fig.savefig(IMAGES / "region_14_vertices.png", dpi=200, facecolor=bg_canvas)
    finally:
        plt.close(fig)


COMMON_SVG_STYLE = r"""
    <!-- Background coordinate grid -->
    <pattern id="bg-grid" width="36" height="36" patternUnits="userSpaceOnUse">
      <path d="M 36 0 L 0 0 0 36" fill="none" stroke="#e6e1d6" stroke-width="0.75"/>
    </pattern>

    <!-- Diagonal Hatch for Outer Nonlinear Envelope -->
    <pattern id="envelope-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <line x1="0" y1="0" x2="0" y2="7" stroke="#265951" stroke-width="1.1" stroke-opacity="0.22"/>
    </pattern>

    <!-- Card drop shadow -->
    <filter id="card-shadow" x="-2%" y="-2%" width="104%" height="106%">
      <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#1e2530" flood-opacity="0.06"/>
    </filter>

    <style>
      @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&amp;family=JetBrains+Mono:wght@400;500;600;700&amp;family=Space+Grotesk:wght@400;500;600;700&amp;display=swap');

      .canvas-bg { fill: #f7f5f0; }
      .outer-frame { fill: none; stroke: #2c3542; stroke-width: 1.5; }
      .inner-brass-rule { fill: none; stroke: #c5b69e; stroke-width: 0.9; }
      .divider-rule { stroke: #dcd6c9; stroke-width: 1; fill: none; }

      .card-base {
        fill: #fdfcf9;
        stroke: #c6bfb0;
        stroke-width: 1.2;
        filter: url(#card-shadow);
      }
      .subpanel {
        fill: #f6f4ee;
        stroke: #dcd6c8;
        stroke-width: 1;
      }
      .plot-box {
        fill: #fdfcf9;
        stroke: #dcd6c8;
        stroke-width: 1;
      }
      .pill-bg {
        fill: #fdfcf9;
        fill-opacity: 0.92;
        stroke: #dcd6c8;
        stroke-width: 0.8;
      }

      .kicker {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.8px;
        font-weight: 600;
        letter-spacing: 0.1em;
        fill: #5c6573;
      }
      .stage-num {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.6px;
        font-weight: 700;
        letter-spacing: 0.08em;
      }
      .main-heading {
        font-family: "Instrument Serif", "Palatino Linotype", "Iowan Old Style", "Baskerville", Georgia, serif;
        font-size: 18.2px;
        font-weight: 400;
        letter-spacing: 0.005em;
        fill: #1a212b;
      }
      .serif-italic {
        font-family: "Instrument Serif", "Palatino Linotype", "Iowan Old Style", "Baskerville", Georgia, serif;
        font-style: italic;
      }
      .card-title {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: -0.01em;
        fill: #1a212b;
      }
      .card-title-sm {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: -0.01em;
        fill: #1a212b;
      }
      .module-tag {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 10px;
        font-weight: 500;
        fill: #5c6573;
      }
      .body-text {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 11px;
        font-weight: 400;
        fill: #39424e;
      }
      .body-sm {
        font-family: "Space Grotesk", "Avenir Next", "Segoe UI", -apple-system, sans-serif;
        font-size: 10.4px;
        font-weight: 400;
        fill: #495361;
      }
      .mono-code {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 10px;
        font-weight: 600;
        fill: #1a212b;
      }
      .mono-sm {
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, monospace;
        font-size: 9.4px;
        font-weight: 500;
        fill: #39424e;
      }
      .axis-line { stroke: #b8b0a0; stroke-width: 1; stroke-dasharray: 2 2; }
      .grid-line { stroke: #e4dec8; stroke-width: 1; stroke-dasharray: 2 2; }

      .c-slate     { fill: #1a212b; }
      .c-muted     { fill: #5c6573; }
      .c-verdigris { fill: #265951; }
      .c-amber     { fill: #8c5a0d; }
      .c-fjord     { fill: #1d3c63; }
      .c-brick     { fill: #7c2d2d; }
      .fw-bold     { font-weight: 700; }
    </style>
"""


def write_conservative_methods() -> None:
    svg = rf"""<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 900 372" role="img" aria-label="Conservative inner approximation methods">
  <defs>
    {COMMON_SVG_STYLE}
    <marker id="arrow-erode" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7c2d2d"/>
    </marker>
    <pattern id="erosion-band" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
      <line x1="0" y1="0" x2="0" y2="6" stroke="#7c2d2d" stroke-width="1.1" stroke-opacity="0.25"/>
    </pattern>
  </defs>

  <!-- 0. BACKGROUND CANVAS & FRAME -->
  <rect class="canvas-bg" x="0" y="0" width="900" height="372" rx="6"/>
  <rect x="12" y="12" width="876" height="348" fill="url(#bg-grid)"/>
  <rect class="outer-frame" x="12" y="12" width="876" height="348" rx="4"/>
  <rect class="inner-brass-rule" x="16" y="16" width="868" height="340" rx="2"/>

  <!-- ==================== LEFT CARD: PAIR 1 (METHOD 1) ==================== -->
  <g transform="translate(28, 26)">
    <rect class="card-base" x="0" y="0" width="412" height="320" rx="6"/>
    <rect x="0" y="0" width="412" height="4" rx="2" fill="#8c5a0d"/>

    <text class="stage-num c-amber" x="18" y="23">PAIR 1  ·  METHOD 1 (DEFAULT: PATH SCOPE)</text>
    <text class="card-title" x="18" y="43">Intersect &amp; Pontryagin Shift</text>
    <text class="module-tag" x="394" y="42" text-anchor="end">method="intersect-shift"</text>

    <rect class="subpanel" x="18" y="52" width="376" height="26" rx="4"/>
    <text class="mono-sm c-slate fw-bold" x="206" y="69" text-anchor="middle">R₁ = ( C^I [lindistflow]  ∩  C^II [lindistflow-vdrop] )  ⊖  B_loss</text>

    <!-- Geometric Canvas Left -->
    <rect class="plot-box" x="18" y="85" width="376" height="168" rx="4"/>
    <line x1="30" y1="172" x2="382" y2="172" class="axis-line"/>
    <line x1="196" y1="93" x2="196" y2="245" class="axis-line"/>

    <!-- True AC Feasible Set Omega_AC -->
    <path d="M 74 164 C 78 114, 156 102, 234 106 C 304 110, 336 144, 326 190 C 316 228, 234 236, 154 230 C 96 224, 70 198, 74 164 Z"
          fill="url(#envelope-hatch)" stroke="#265951" stroke-width="1.6" stroke-dasharray="5 3"/>

    <!-- Model I: lindistflow (exceeds right/top slightly) -->
    <polygon points="88,160 144,106 272,108 344,156 314,222 146,228"
             fill="none" stroke="#1d3c63" stroke-width="1.4" stroke-dasharray="4 2"/>

    <!-- Model II: lindistflow-vdrop (shifted voltage bracket) -->
    <polygon points="72,170 128,116 250,112 320,164 298,230 130,232"
             fill="none" stroke="#8c5a0d" stroke-width="1.4" stroke-dasharray="4 2"/>

    <!-- Unshifted Intersection C_bar = C^I cap C^II (erosion collar) -->
    <polygon points="88,164 138,116 250,112 320,164 298,224 146,228"
             fill="url(#erosion-band)" stroke="#7c2d2d" stroke-width="1" stroke-dasharray="2 2"/>

    <!-- Inner Eroded Polytope R_1 -->
    <polygon points="110,166 152,128 236,126 286,164 270,210 156,214"
             fill="#265951" fill-opacity="0.22" stroke="#265951" stroke-width="2.2"/>
    <circle cx="196" cy="172" r="3.5" fill="#c88a28" stroke="#1a212b" stroke-width="1"/>
    <text x="202" y="167" class="mono-sm c-slate fw-bold">0</text>
    <text x="206" y="195" text-anchor="middle" class="mono-code c-verdigris">Certified R₁ ⊂ Ω_AC</text>

    <!-- Pontryagin Erosion Arrows (-B_loss) -->
    <line x1="242" y1="113" x2="233" y2="124" stroke="#7c2d2d" stroke-width="1.5" marker-end="url(#arrow-erode)"/>
    <line x1="316" y1="164" x2="290" y2="164" stroke="#7c2d2d" stroke-width="1.5" marker-end="url(#arrow-erode)"/>
    <line x1="288" y1="220" x2="273" y2="211" stroke="#7c2d2d" stroke-width="1.5" marker-end="url(#arrow-erode)"/>

    <!-- Corner Pill Labels -->
    <rect class="pill-bg" x="24" y="91" width="136" height="17" rx="3"/>
    <text x="30" y="103" class="mono-sm c-fjord fw-bold">--- C^I (lindistflow)</text>
    <rect class="pill-bg" x="282" y="91" width="106" height="17" rx="3"/>
    <text x="288" y="103" class="mono-sm c-verdigris fw-bold">--- True Ω_AC</text>
    <rect class="pill-bg" x="24" y="230" width="168" height="17" rx="3"/>
    <text x="30" y="242" class="mono-sm c-amber fw-bold">--- C^II (lindistflow-vdrop)</text>
    <rect class="pill-bg" x="270" y="230" width="118" height="17" rx="3"/>
    <text x="276" y="242" class="mono-sm c-brick fw-bold">→ −B_loss erosion</text>

    <!-- Bottom Explanations -->
    <text x="18" y="271" class="body-text"><tspan class="fw-bold c-slate">1. Voltage bracket:</tspan> Intersects <tspan class="mono-sm">C^I</tspan> &amp; <tspan class="mono-sm">C^II</tspan> to bracket branch voltage drop.</text>
    <text x="18" y="290" class="body-text"><tspan class="fw-bold c-slate">2. Loss erosion:</tspan> Tightens each facet by <tspan class="mono-sm">∑ max(0, a_k s_k)</tspan> (+ 2^k Q slices).</text>
    <text x="18" y="309" class="body-text"><tspan class="fw-bold c-verdigris">9bus / 247:</tspan> <tspan class="mono-code">238.08 MW³</tspan> volume · certified at <tspan class="mono-code">91.06%</tspan> max loading.</text>
  </g>

  <!-- ==================== RIGHT CARD: PAIR 2 (METHOD 2) ==================== -->
  <g transform="translate(460, 26)">
    <rect class="card-base" x="0" y="0" width="412" height="320" rx="6"/>
    <rect x="0" y="0" width="412" height="4" rx="2" fill="#1d3c63"/>

    <text class="stage-num c-fjord" x="18" y="23">PAIR 2  ·  METHOD 2 (DIRECT INTERSECTION)</text>
    <text class="card-title" x="18" y="43">Direct Tightened Intersection</text>
    <text class="module-tag" x="394" y="42" text-anchor="end">method="tight-intersect"</text>

    <rect class="subpanel" x="18" y="52" width="376" height="26" rx="4"/>
    <text class="mono-sm c-slate fw-bold" x="206" y="69" text-anchor="middle">R₂ = C^I_tight [tight-voltage]  ∩  C^II_tight [tight-flow]</text>

    <!-- Geometric Canvas Right -->
    <rect class="plot-box" x="18" y="85" width="376" height="168" rx="4"/>
    <line x1="30" y1="172" x2="382" y2="172" class="axis-line"/>
    <line x1="196" y1="93" x2="196" y2="245" class="axis-line"/>

    <!-- True AC Feasible Set Omega_AC -->
    <path d="M 74 164 C 78 114, 156 102, 234 106 C 304 110, 336 144, 326 190 C 316 228, 234 236, 154 230 C 96 224, 70 198, 74 164 Z"
          fill="url(#envelope-hatch)" stroke="#265951" stroke-width="1.6" stroke-dasharray="5 3"/>

    <!-- Model I tight: tight-voltage (tight on top/bottom voltage, slightly wider on right flow) -->
    <polygon points="96,166 150,122 262,126 334,170 268,218 144,220"
             fill="none" stroke="#1d3c63" stroke-width="1.4" stroke-dasharray="4 2"/>

    <!-- Model II tight: tight-flow (tight on left/right flow, slightly taller on top voltage) -->
    <polygon points="114,158 202,102 260,112 292,174 260,214 134,212"
             fill="none" stroke="#8c5a0d" stroke-width="1.4" stroke-dasharray="4 2"/>

    <!-- Direct Intersection R_2 (strictly inside Omega_AC) -->
    <polygon points="114,158 156,122 256,126 292,174 260,214 144,213"
             fill="#1d3c63" fill-opacity="0.20" stroke="#1d3c63" stroke-width="2.2"/>
    <circle cx="196" cy="172" r="3.5" fill="#c88a28" stroke="#1a212b" stroke-width="1"/>
    <text x="202" y="167" class="mono-sm c-slate fw-bold">0</text>
    <text x="198" y="192" text-anchor="middle" class="mono-code c-fjord">Certified R₂ (no shift)</text>

    <!-- Corner Pill Labels -->
    <rect class="pill-bg" x="24" y="91" width="168" height="17" rx="3"/>
    <text x="30" y="103" class="mono-sm c-fjord fw-bold">--- C^I_tight (tight-voltage)</text>
    <rect class="pill-bg" x="282" y="91" width="106" height="17" rx="3"/>
    <text x="288" y="103" class="mono-sm c-verdigris fw-bold">--- True Ω_AC</text>
    <rect class="pill-bg" x="24" y="230" width="156" height="17" rx="3"/>
    <text x="30" y="242" class="mono-sm c-amber fw-bold">--- C^II_tight (tight-flow)</text>
    <rect class="pill-bg" x="258" y="230" width="130" height="17" rx="3"/>
    <text x="264" y="242" class="mono-sm c-fjord fw-bold">✓ Direct C^I ∩ C^II</text>

    <!-- Bottom Explanations -->
    <text x="18" y="271" class="body-text"><tspan class="fw-bold c-slate">1. Embedded bounds:</tspan> <tspan class="mono-sm">tight-voltage</tspan> &amp; <tspan class="mono-sm">tight-flow</tspan> bound drop &amp; loss.</text>
    <text x="18" y="290" class="body-text"><tspan class="fw-bold c-slate">2. Direct intersection:</tspan> <tspan class="mono-sm">C^I_tight ∩ C^II_tight</tspan> avoids post-intersection slice builds.</text>
    <text x="18" y="309" class="body-text"><tspan class="fw-bold c-fjord">9bus / 247:</tspan> <tspan class="mono-code">95.37 MW³</tspan> volume · certified at <tspan class="mono-code">70.62%</tspan> max loading.</text>
  </g>
</svg>
"""
    (IMAGES / "conservative_methods.svg").write_text(svg, encoding="utf-8")


def write_growth_and_compression() -> None:
    svg = rf"""<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 900 372" role="img" aria-label="Seed geometry, ray growth and live-hull compression">
  <defs>
    {COMMON_SVG_STYLE}
    <marker id="arr-pine" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#265951"/>
    </marker>
    <marker id="arr-rust" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7c2d2d"/>
    </marker>
  </defs>

  <!-- 0. BACKGROUND CANVAS & FRAME -->
  <rect class="canvas-bg" x="0" y="0" width="900" height="372" rx="6"/>
  <rect x="12" y="12" width="876" height="348" fill="url(#bg-grid)"/>
  <rect class="outer-frame" x="12" y="12" width="876" height="348" rx="4"/>
  <rect class="inner-brass-rule" x="16" y="16" width="868" height="340" rx="2"/>

  <!-- ==================== CARD I: SEED GEOMETRY ==================== -->
  <g transform="translate(28, 26)">
    <rect class="card-base" x="0" y="0" width="268" height="320" rx="6"/>
    <rect x="0" y="0" width="268" height="4" rx="2" fill="#1d3c63"/>

    <text class="stage-num c-fjord" x="16" y="23">STEP I  ·  SEED GEOMETRY</text>
    <text class="card-title-sm" x="16" y="43">Axis vs. Simplex Seed</text>
    <text class="module-tag" x="252" y="42" text-anchor="end">seed="auto"</text>

    <rect class="plot-box" x="16" y="52" width="236" height="168" rx="4"/>
    <line x1="26" y1="140" x2="242" y2="140" class="axis-line"/>
    <line x1="134" y1="64" x2="134" y2="212" class="axis-line"/>

    <!-- Top Pill Legend -->
    <rect class="pill-bg" x="22" y="58" width="104" height="17" rx="3"/>
    <text x="28" y="70" class="mono-sm c-fjord fw-bold">■ axis (2d rays)</text>
    <rect class="pill-bg" x="142" y="58" width="104" height="17" rx="3"/>
    <text x="148" y="70" class="mono-sm c-brick fw-bold">▲ simplex (d+1)</text>

    <!-- True boundary -->
    <ellipse cx="134" cy="144" rx="84" ry="54" fill="url(#envelope-hatch)" stroke="#265951" stroke-width="1.3" stroke-dasharray="4 3"/>

    <!-- Axis cross-polytope -->
    <polygon points="134,90 218,144 134,198 50,144" fill="#1d3c63" fill-opacity="0.16" stroke="#1d3c63" stroke-width="1.8"/>

    <!-- One-sided simplex seed -->
    <polygon points="134,90 218,144 102,168" fill="#7c2d2d" fill-opacity="0.22" stroke="#7c2d2d" stroke-width="1.6" stroke-dasharray="3 2"/>

    <!-- Random Sobol cloud dots -->
    <circle cx="96" cy="120" r="2.5" fill="#265951"/>
    <circle cx="170" cy="118" r="2.5" fill="#265951"/>
    <circle cx="176" cy="162" r="2.5" fill="#265951"/>
    <circle cx="90" cy="160" r="2.5" fill="#265951"/>
    <circle cx="134" cy="144" r="3.5" fill="#c88a28" stroke="#1a212b" stroke-width="0.9"/>

    <rect class="subpanel" x="16" y="228" width="236" height="78" rx="4"/>
    <text x="26" y="246" class="mono-code c-slate">Axis / Simplex Volume Ratio:</text>
    <text x="26" y="264" class="mono-sm c-muted">3-D / 5-D</text>
    <text x="242" y="264" text-anchor="end" class="mono-sm c-fjord fw-bold">1.3×  /  21×</text>
    <text x="26" y="282" class="mono-sm c-muted">8-D / 16-D</text>
    <text x="242" y="282" text-anchor="end" class="mono-sm c-brick fw-bold">2,410×  /  27,900×</text>
    <text x="26" y="299" class="body-sm">Default <tspan class="mono-sm">seed="auto"</tspan> picks <tspan class="mono-sm">axis</tspan> (d ≥ 4).</text>
  </g>

  <!-- Step Chevron 1 -> 2 -->
  <circle cx="306" cy="186" r="7" fill="#fdfcf9" stroke="#c5b69e" stroke-width="1"/>
  <path d="M 304 182.5 L 308.5 186 L 304 189.5" fill="none" stroke="#2c3542" stroke-width="1.4"/>

  <!-- ==================== CARD II: RAY GROWTH & POLISH ==================== -->
  <g transform="translate(316, 26)">
    <rect class="card-base" x="0" y="0" width="268" height="320" rx="6"/>
    <rect x="0" y="0" width="268" height="4" rx="2" fill="#265951"/>

    <text class="stage-num c-verdigris" x="16" y="23">STEP II  ·  RAY &amp; RELOCATION</text>
    <text class="card-title-sm" x="16" y="43">Grow-and-Polish Engine</text>
    <text class="module-tag" x="252" y="42" text-anchor="end">ray-growth</text>

    <rect class="plot-box" x="16" y="52" width="236" height="168" rx="4"/>
    <rect class="pill-bg" x="22" y="58" width="110" height="17" rx="3"/>
    <text x="28" y="70" class="mono-sm c-brick fw-bold">↷ ∇_τ Vol polish</text>
    <rect class="pill-bg" x="138" y="58" width="108" height="17" rx="3"/>
    <text x="144" y="70" class="mono-sm c-verdigris fw-bold">● v_new (ITP ray)</text>

    <!-- Boundary curve -->
    <path d="M 34 202 C 32 122, 98 88, 180 96 C 232 102, 242 158, 222 206" fill="none" stroke="#265951" stroke-width="1.6" stroke-dasharray="4 3"/>

    <!-- Current hull -->
    <polygon points="54,196 80,130 172,136 210,200" fill="#1d3c63" fill-opacity="0.15" stroke="#1d3c63" stroke-width="1.7"/>
    <!-- New Cone Gain Delta V -->
    <polygon points="80,130 148,95 172,136" fill="#265951" fill-opacity="0.28" stroke="#265951" stroke-width="1.8"/>
    <text x="115" y="123" text-anchor="middle" class="mono-sm c-verdigris fw-bold">+ΔV</text>

    <!-- Outward ray n_f & boundary hit -->
    <line x1="126" y1="133" x2="146" y2="99" stroke="#265951" stroke-width="1.6" marker-end="url(#arr-pine)"/>
    <circle cx="148" cy="95" r="3.8" fill="#265951" stroke="#fdfcf9" stroke-width="1.1"/>

    <!-- Tangential Jacobi volume-gradient relocation -->
    <path d="M 80 130 Q 70 114 62 108" fill="none" stroke="#7c2d2d" stroke-width="1.6" marker-end="url(#arr-rust)"/>
    <path d="M 172 136 Q 190 124 202 120" fill="none" stroke="#7c2d2d" stroke-width="1.6" marker-end="url(#arr-rust)"/>
    <text x="132" y="174" text-anchor="middle" class="mono-code c-fjord">Active Hull P_k</text>

    <rect class="subpanel" x="16" y="228" width="236" height="78" rx="4"/>
    <text x="26" y="246" class="mono-code c-slate">Bandit &amp; Relocation Gains:</text>
    <text x="26" y="264" class="mono-sm c-muted">adaptive-gradient</text>
    <text x="242" y="264" text-anchor="end" class="mono-sm c-verdigris fw-bold">1.00 ray / round</text>
    <text x="26" y="282" class="mono-sm c-muted">refresh_gains=True</text>
    <text x="242" y="282" text-anchor="end" class="mono-sm c-fjord fw-bold">+107× vol (8-D)</text>
    <text x="26" y="299" class="body-sm"><tspan class="mono-sm">grow-and-polish</tspan> adds +102% to +136%.</text>
  </g>

  <!-- Step Chevron 2 -> 3 -->
  <circle cx="594" cy="186" r="7" fill="#fdfcf9" stroke="#c5b69e" stroke-width="1"/>
  <path d="M 592 182.5 L 596.5 186 L 592 189.5" fill="none" stroke="#2c3542" stroke-width="1.4"/>

  <!-- ==================== CARD III: COMPRESSION ==================== -->
  <g transform="translate(604, 26)">
    <rect class="card-base" x="0" y="0" width="268" height="320" rx="6"/>
    <rect x="0" y="0" width="268" height="4" rx="2" fill="#7c2d2d"/>

    <text class="stage-num c-brick" x="16" y="23">STEP III  ·  COMPRESSION</text>
    <text class="card-title-sm" x="16" y="43">LiveHull Pruning</text>
    <text class="module-tag" x="252" y="42" text-anchor="end">compress_live</text>

    <rect class="plot-box" x="16" y="52" width="236" height="168" rx="4"/>
    <rect class="pill-bg" x="46" y="58" width="176" height="17" rx="3"/>
    <text x="134" y="70" text-anchor="middle" class="mono-sm c-brick fw-bold">● Pruned v*_i (min ΔV_i star)</text>

    <!-- Retained compressed hull -->
    <polygon points="44,170 76,106 168,102 222,150 190,204 86,206"
             fill="#1d3c63" fill-opacity="0.18" stroke="#1d3c63" stroke-width="2"/>
    <text x="134" y="160" text-anchor="middle" class="mono-code c-fjord">Retained P_comp ⊂ P</text>

    <!-- Pruned vertex star wedge -->
    <polygon points="76,106 124,84 168,102" fill="#c88a28" fill-opacity="0.32" stroke="#7c2d2d" stroke-width="1.6" stroke-dasharray="3 2"/>
    <circle cx="124" cy="84" r="4" fill="#7c2d2d" stroke="#fdfcf9" stroke-width="1.1"/>

    <!-- Retained vertices -->
    <circle cx="44" cy="170" r="3.4" fill="#1d3c63"/>
    <circle cx="76" cy="106" r="3.4" fill="#1d3c63"/>
    <circle cx="168" cy="102" r="3.4" fill="#1d3c63"/>
    <circle cx="222" cy="150" r="3.4" fill="#1d3c63"/>
    <circle cx="190" cy="204" r="3.4" fill="#1d3c63"/>
    <circle cx="86" cy="206" r="3.4" fill="#1d3c63"/>

    <rect class="subpanel" x="16" y="228" width="236" height="78" rx="4"/>
    <text x="26" y="246" class="mono-code c-slate">In-Place Memory Governance:</text>
    <text x="26" y="264" class="mono-sm c-muted">Exact / Screened</text>
    <text x="242" y="264" text-anchor="end" class="mono-sm c-amber fw-bold">min marginal ΔV_i</text>
    <text x="26" y="282" class="mono-sm c-muted">star-screened (8-D)</text>
    <text x="242" y="282" text-anchor="end" class="mono-sm c-verdigris fw-bold">3.2×–3.9× faster</text>
    <text x="26" y="299" class="body-sm">Enforces hard <tspan class="mono-sm">budget_mb</tspan> &amp; price cap.</text>
  </g>
</svg>
"""
    (IMAGES / "growth_and_compression.svg").write_text(svg, encoding="utf-8")


def write_validation_and_certification() -> None:
    svg = rf"""<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 900 382" role="img" aria-label="Validation set measures and global inner-region certification">
  <defs>
    {COMMON_SVG_STYLE}
    <marker id="arr-slate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2c3542"/>
    </marker>
  </defs>

  <!-- 0. BACKGROUND CANVAS & FRAME -->
  <rect class="canvas-bg" x="0" y="0" width="900" height="382" rx="6"/>
  <rect x="12" y="12" width="876" height="358" fill="url(#bg-grid)"/>
  <rect class="outer-frame" x="12" y="12" width="876" height="358" rx="4"/>
  <rect class="inner-brass-rule" x="16" y="16" width="868" height="350" rx="2"/>

  <!-- ==================== LEFT CARD: SET PARTITION & COUNTEREXAMPLE ==================== -->
  <g transform="translate(28, 26)">
    <rect class="card-base" x="0" y="0" width="412" height="330" rx="6"/>
    <rect x="0" y="0" width="412" height="4" rx="2" fill="#7c2d2d"/>

    <text class="stage-num c-brick" x="18" y="23">PART I  ·  SET MEASURES &amp; COUNTEREXAMPLE</text>
    <text class="card-title" x="18" y="43">Small-Corner Refutation</text>
    <text class="module-tag" x="394" y="42" text-anchor="end">evaluate_region</text>

    <rect class="plot-box" x="18" y="52" width="376" height="182" rx="4"/>

    <!-- Top Pill Labels -->
    <rect class="pill-bg" x="24" y="58" width="172" height="17" rx="3"/>
    <text x="30" y="70" class="mono-sm c-fjord fw-bold">--- True Feasible Set D (N_D)</text>
    <rect class="pill-bg" x="204" y="58" width="184" height="17" rx="3"/>
    <text x="210" y="70" class="mono-sm c-brick fw-bold">▲ Corner R \ D (100.9% load)</text>

    <!-- True Feasible Domain N_D -->
    <path d="M 42 144 C 46 92, 130 80, 224 86 C 296 92, 324 124, 312 170 C 300 204, 210 210, 120 204 C 58 198, 38 176, 42 144 Z"
          fill="#edf2f8" fill-opacity="0.75" stroke="#1d3c63" stroke-width="1.5" stroke-dasharray="5 3"/>
    <text x="206" y="224" text-anchor="middle" class="mono-sm c-fjord fw-bold">Missed feasible space (N_D \ N_RD → Loss) · ○ MC samples</text>

    <!-- Candidate Polytope R -->
    <polygon points="88,144 140,96 268,100 358,136 278,184 138,182"
             fill="#265951" fill-opacity="0.22" stroke="#265951" stroke-width="2.1"/>
    <text x="196" y="140" text-anchor="middle" class="mono-code c-verdigris">Intersection N_RD = R ∩ D</text>

    <!-- Violating Corner (R \ D) -->
    <polygon points="312,118 358,136 314,161" fill="#7c2d2d" fill-opacity="0.45" stroke="#7c2d2d" stroke-width="1.8"/>
    <circle cx="358" cy="136" r="4.2" fill="#7c2d2d" stroke="#fdfcf9" stroke-width="1.2"/>

    <!-- Unresolved pocket N_U,R -->
    <ellipse cx="134" cy="156" rx="24" ry="12" fill="#c88a28" fill-opacity="0.25" stroke="#8c5a0d" stroke-width="1.2" stroke-dasharray="2 2"/>
    <text x="134" y="159" text-anchor="middle" class="mono-sm c-amber fw-bold">N_U,R</text>

    <!-- Sample dots inside N_RD -->
    <circle cx="164" cy="116" r="2.5" fill="#265951" stroke="#fdfcf9" stroke-width="0.6"/>
    <circle cx="210" cy="112" r="2.5" fill="#265951" stroke="#fdfcf9" stroke-width="0.6"/>
    <circle cx="248" cy="124" r="2.5" fill="#265951" stroke="#fdfcf9" stroke-width="0.6"/>
    <circle cx="232" cy="162" r="2.5" fill="#265951" stroke="#fdfcf9" stroke-width="0.6"/>
    <circle cx="180" cy="170" r="2.5" fill="#265951" stroke="#fdfcf9" stroke-width="0.6"/>
    <circle cx="282" cy="146" r="2.5" fill="#265951" stroke="#fdfcf9" stroke-width="0.6"/>

    <rect class="subpanel" x="18" y="242" width="376" height="74" rx="4"/>
    <text x="206" y="261" text-anchor="middle" class="mono-code c-slate">error = 1 − N_RD / (N_R − N_U,R)   ·   loss = 1 − N_RD / N_D</text>
    <text x="28" y="282" class="body-sm"><tspan class="fw-bold c-brick">Why certify:</tspan> On <tspan class="mono-sm">9bus/247</tspan> (<tspan class="mono-sm">feeding_branch</tspan>), 1,000 Monte Carlo draws</text>
    <text x="28" y="300" class="body-sm">report <tspan class="fw-bold c-slate">0.000% error</tspan>, yet <tspan class="mono-sm">--certify</tspan> refutes the corner at <tspan class="fw-bold c-brick">100.90%</tspan>.</text>
  </g>

  <!-- ==================== RIGHT CARD: 4-STAGE ESCALATION & CERTIFICATE ==================== -->
  <g transform="translate(460, 26)">
    <rect class="card-base" x="0" y="0" width="412" height="330" rx="6"/>
    <rect x="0" y="0" width="412" height="4" rx="2" fill="#265951"/>

    <text class="stage-num c-verdigris" x="18" y="23">PART II  ·  DETERMINISTIC ESCALATION</text>
    <text class="card-title" x="18" y="43">Four-Stage Certificate</text>
    <text class="module-tag" x="394" y="42" text-anchor="end">verify_inner_region</text>

    <!-- Stage 1 -->
    <rect class="subpanel" x="18" y="52" width="376" height="54" rx="4"/>
    <rect x="18" y="52" width="4" height="54" rx="2" fill="#1d3c63"/>
    <text x="32" y="71" class="mono-code c-fjord">1 · Physical Witness (acpf / RadialSweep)</text>
    <rect x="308" y="58" width="78" height="16" rx="3" fill="#e6f0ec" stroke="#265951" stroke-width="0.8"/>
    <text x="347" y="69.5" text-anchor="middle" class="mono-sm c-verdigris fw-bold">FEASIBLE</text>
    <text x="32" y="92" class="body-sm">Converged within limits settles point; non-convergence escalates.</text>
    <line x1="206" y1="106" x2="206" y2="118" stroke="#2c3542" stroke-width="1.3" marker-end="url(#arr-slate)"/>

    <!-- Stage 2 -->
    <rect class="subpanel" x="18" y="120" width="376" height="56" rx="4"/>
    <rect x="18" y="120" width="4" height="56" rx="2" fill="#8c5a0d"/>
    <text x="32" y="139" class="mono-code c-amber">2 · Outer SOCP + Secant Tightening (~10 ms)</text>
    <rect x="294" y="126" width="92" height="16" rx="3" fill="#f8efe0" stroke="#8c5a0d" stroke-width="0.8"/>
    <text x="340" y="137.5" text-anchor="middle" class="mono-sm c-amber fw-bold">875/885 DONE</text>
    <text x="32" y="160" class="body-sm">Outer infeasibility ⇒ <tspan class="fw-bold c-brick">INFEASIBLE</tspan>. Screens limits in 17 s vs &gt;22 min.</text>
    <line x1="206" y1="176" x2="206" y2="188" stroke="#2c3542" stroke-width="1.3" marker-end="url(#arr-slate)"/>

    <!-- Stage 3 -->
    <rect class="subpanel" x="18" y="190" width="376" height="56" rx="4"/>
    <rect x="18" y="190" width="4" height="56" rx="2" fill="#265951"/>
    <text x="32" y="209" class="mono-code c-verdigris">3 · Global Exact B&amp;B + Banach Contraction</text>
    <rect x="298" y="196" width="88" height="16" rx="3" fill="#edf2f8" stroke="#1d3c63" stroke-width="0.8"/>
    <text x="342" y="207.5" text-anchor="middle" class="mono-sm c-fjord fw-bold">NonConvex=2</text>
    <text x="32" y="230" class="body-sm">Global worst-case limits over x ∈ R + vertex contraction <tspan class="mono-sm">ξ &lt; v₀²/4</tspan>.</text>
    <line x1="206" y1="246" x2="206" y2="258" stroke="#2c3542" stroke-width="1.3" marker-end="url(#arr-slate)"/>

    <!-- Stage 4 -->
    <rect class="plot-box" x="18" y="260" width="376" height="56" rx="4" stroke="#265951" stroke-width="1.4"/>
    <rect x="18" y="260" width="4" height="56" rx="2" fill="#7c2d2d"/>
    <text x="32" y="279" class="mono-code c-slate">4 · Final Certificate &amp; Contraction Repair</text>
    <text x="32" y="297" class="mono-sm c-verdigris fw-bold">✓ CERTIFIED (scope="inner")</text>
    <text x="216" y="297" class="mono-sm c-brick fw-bold">✗ REFUTED + exact proof</text>
    <text x="32" y="311" class="mono-sm c-muted">◆ LIMIT-VIOLATED (unsafe root)</text>
    <text x="216" y="311" class="mono-sm c-fjord fw-bold">⟳ shrink_to_certified → λ·R</text>
  </g>
</svg>
"""
    (IMAGES / "validation_and_certification.svg").write_text(svg, encoding="utf-8")


def write_benchmarks_overview() -> None:
    svg = rf"""<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 900 382" role="img" aria-label="Cross-network paired volume ratios and high-dimensional live-hull memory caps">
  <defs>
    {COMMON_SVG_STYLE}
  </defs>

  <!-- 0. BACKGROUND CANVAS & FRAME -->
  <rect class="canvas-bg" x="0" y="0" width="900" height="382" rx="6"/>
  <rect x="12" y="12" width="876" height="358" fill="url(#bg-grid)"/>
  <rect class="outer-frame" x="12" y="12" width="876" height="358" rx="4"/>
  <rect class="inner-brass-rule" x="16" y="16" width="868" height="350" rx="2"/>

  <!-- ==================== LEFT CARD: 96-INSTANCE BENCHMARK ==================== -->
  <g transform="translate(28, 26)">
    <rect class="card-base" x="0" y="0" width="412" height="330" rx="6"/>
    <rect x="0" y="0" width="412" height="4" rx="2" fill="#1d3c63"/>

    <text class="stage-num c-fjord" x="18" y="23">BENCHMARK I  ·  96 PAIRED INSTANCES (95% CI)</text>
    <text class="card-title" x="18" y="43">8-D Geometric-Mean Ratios</text>
    <text class="module-tag" x="394" y="42" text-anchor="end">OpenRC.benchmark</text>

    <!-- Sub-chart A: Axis / Simplex Seed Ratio -->
    <rect class="plot-box" x="18" y="52" width="376" height="128" rx="4"/>
    <text x="28" y="69" class="mono-code c-fjord">A · Seed Ratio: axis / simplex (log₁₀ scale, 8-D &amp; 5-D)</text>
    <line x1="98" y1="76" x2="98" y2="162" stroke="#7c2d2d" stroke-width="1.3"/>
    <line x1="150" y1="76" x2="150" y2="162" class="grid-line"/>
    <line x1="202" y1="76" x2="202" y2="162" class="grid-line"/>
    <line x1="254" y1="76" x2="254" y2="162" class="grid-line"/>
    <line x1="306" y1="76" x2="306" y2="162" class="grid-line"/>

    <!-- Log10 Axis Tick Labels -->
    <text x="98" y="174" text-anchor="middle" class="mono-sm c-brick fw-bold">1×</text>
    <text x="150" y="174" text-anchor="middle" class="mono-sm c-muted">10×</text>
    <text x="202" y="174" text-anchor="middle" class="mono-sm c-muted">10²×</text>
    <text x="254" y="174" text-anchor="middle" class="mono-sm c-muted">10³×</text>
    <text x="306" y="174" text-anchor="middle" class="mono-sm c-muted">10⁴×</text>

    <text x="28" y="92" class="mono-sm c-slate">9bus</text>
    <rect x="98" y="83" width="125" height="10" rx="2" fill="#1d3c63"/>
    <text x="230" y="92" class="mono-sm c-fjord fw-bold">258× (8D) · 7.9× (5D)</text>

    <text x="28" y="113" class="mono-sm c-slate">oberrhein</text>
    <rect x="98" y="104" width="104" height="10" rx="2" fill="#1d3c63"/>
    <text x="209" y="113" class="mono-sm c-fjord fw-bold">102× (8D) · 19.3× (5D)</text>

    <text x="28" y="134" class="mono-sm c-slate">CIGRE MV</text>
    <rect x="98" y="125" width="196" height="10" rx="2" fill="#265951"/>
    <text x="300" y="134" class="mono-sm c-verdigris fw-bold">5,908× (8D)</text>

    <text x="28" y="155" class="mono-sm c-slate">case33bw</text>
    <rect x="98" y="146" width="219" height="10" rx="2" fill="#8c5a0d"/>
    <text x="323" y="155" class="mono-sm c-amber fw-bold">16,483×</text>

    <!-- Sub-chart B: Solver Strategy Gains -->
    <rect class="subpanel" x="18" y="186" width="376" height="130" rx="4"/>
    <text x="28" y="203" class="mono-code c-verdigris">B · 8-D Strategy Gains vs. Plain (1.0×)</text>
    <rect class="pill-bg" x="286" y="191" width="100" height="15" rx="3"/>
    <text x="336" y="202" text-anchor="middle" class="mono-sm"><tspan class="c-verdigris fw-bold">■ regrow</tspan>  <tspan class="c-amber fw-bold">■ polish</tspan></text>
    <line x1="98" y1="210" x2="98" y2="306" stroke="#7c2d2d" stroke-width="1.3"/>

    <text x="28" y="226" class="mono-sm c-slate">9bus</text>
    <rect x="98" y="216" width="70" height="8" rx="1.5" fill="#265951"/>
    <rect x="98" y="225" width="18" height="7" rx="1.5" fill="#8c5a0d"/>
    <text x="174" y="226" class="mono-sm c-verdigris fw-bold">2.00× regrow · 1.25× polish</text>

    <text x="28" y="250" class="mono-sm c-slate">oberrhein</text>
    <rect x="98" y="240" width="33" height="8" rx="1.5" fill="#265951"/>
    <rect x="98" y="249" width="60" height="7" rx="1.5" fill="#8c5a0d"/>
    <text x="164" y="250" class="mono-sm c-amber fw-bold">1.85× polish · 1.48× regrow</text>

    <text x="28" y="274" class="mono-sm c-slate">CIGRE MV</text>
    <rect x="98" y="264" width="34" height="8" rx="1.5" fill="#265951"/>
    <rect x="98" y="273" width="57" height="7" rx="1.5" fill="#8c5a0d"/>
    <text x="161" y="274" class="mono-sm c-amber fw-bold">1.82× polish · 1.49× regrow</text>

    <text x="28" y="298" class="mono-sm c-slate">case33bw</text>
    <rect x="98" y="288" width="15" height="8" rx="1.5" fill="#265951"/>
    <rect x="98" y="297" width="196" height="7" rx="1.5" fill="#8c5a0d"/>
    <text x="300" y="298" class="mono-sm c-amber fw-bold">3.80× polish</text>
  </g>

  <!-- ==================== RIGHT CARD: MEMORY BUDGET GOVERNANCE ==================== -->
  <g transform="translate(460, 26)">
    <rect class="card-base" x="0" y="0" width="412" height="330" rx="6"/>
    <rect x="0" y="0" width="412" height="4" rx="2" fill="#8c5a0d"/>

    <text class="stage-num c-amber" x="18" y="23">BENCHMARK II  ·  MEMORY GOVERNANCE (9BUS AC)</text>
    <text class="card-title" x="18" y="43">Hard Live-Hull Memory Caps</text>
    <text class="module-tag" x="394" y="42" text-anchor="end">MemoryAwareBuilder</text>

    <!-- 16-D Memory vs Volume Chart -->
    <rect class="plot-box" x="18" y="52" width="376" height="142" rx="4"/>
    <text x="28" y="69" class="mono-code c-slate">A · 16-D Growth under 80 MiB Cap (2¹⁶ seed facets)</text>

    <!-- 80 MiB Cap Line at x=166 -->
    <line x1="166" y1="78" x2="166" y2="186" stroke="#7c2d2d" stroke-width="1.4" stroke-dasharray="3 2"/>
    <text x="171" y="88" class="mono-sm c-brick fw-bold">80 MiB Cap</text>

    <text x="28" y="106" class="mono-sm c-slate">vertex-budget</text>
    <rect x="114" y="96" width="128" height="12" rx="2" fill="#7c2d2d" fill-opacity="0.88"/>
    <!-- Scale break mark on 433 MiB bar -->
    <line x1="212" y1="94" x2="207" y2="110" stroke="#fdfcf9" stroke-width="1.8"/>
    <line x1="217" y1="94" x2="212" y2="110" stroke="#fdfcf9" stroke-width="1.8"/>
    <text x="248" y="106" class="mono-sm c-brick fw-bold">433 MiB (5.4× cap)</text>

    <text x="28" y="130" class="mono-sm c-slate">no compress</text>
    <rect x="114" y="120" width="32" height="12" rx="2" fill="#5c6573"/>
    <text x="174" y="130" class="mono-sm c-muted">46.8 MiB · vol = 10,812 (1.0×)</text>

    <text x="28" y="154" class="mono-sm c-slate">adaptive+reloc</text>
    <rect x="114" y="144" width="32" height="12" rx="2" fill="#1d3c63"/>
    <text x="174" y="154" class="mono-sm c-fjord fw-bold">46.8 MiB · vol = 12,193 (+13%)</text>

    <text x="28" y="178" class="mono-sm c-slate">grow-&amp;-polish</text>
    <rect x="114" y="168" width="32" height="12" rx="2" fill="#265951"/>
    <text x="174" y="178" class="mono-sm c-verdigris fw-bold">46.8 MiB · vol = 16,800 (+55%)</text>

    <!-- 8-D Volume Comparison under 16 MiB Cap -->
    <rect class="subpanel" x="18" y="202" width="376" height="114" rx="4"/>
    <text x="28" y="220" class="mono-code c-slate">B · 8-D Growth under 16 MiB Cap (Peak 14.0–15.9 MiB)</text>

    <text x="28" y="246" class="mono-sm c-slate">vertex-budget</text>
    <rect x="114" y="236" width="46" height="11" rx="2" fill="#7c2d2d"/>
    <text x="166" y="246" class="mono-sm c-brick fw-bold">2.29e7 MW⁸ (−44% loss · 14.0 MiB)</text>

    <text x="28" y="272" class="mono-sm c-slate">no compress</text>
    <rect x="114" y="262" width="82" height="11" rx="2" fill="#1d3c63"/>
    <text x="202" y="272" class="mono-sm c-fjord fw-bold">4.08e7 MW⁸ (baseline · 14.5 MiB)</text>

    <text x="28" y="298" class="mono-sm c-slate">adaptive</text>
    <rect x="114" y="288" width="100" height="11" rx="2" fill="#265951"/>
    <text x="220" y="298" class="mono-sm c-verdigris fw-bold">5.00e7 MW⁸ (+22.5% · 14.7 MiB)</text>
  </g>
</svg>
"""
    (IMAGES / "benchmarks_overview.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    write_workflow()
    write_region()
    write_conservative_methods()
    write_growth_and_compression()
    write_validation_and_certification()
    write_benchmarks_overview()
    print(f"wrote {IMAGES / 'workflow.svg'}")
    print(f"wrote {IMAGES / 'region_14_vertices.png'}")
    print(f"wrote {IMAGES / 'conservative_methods.svg'}")
    print(f"wrote {IMAGES / 'growth_and_compression.svg'}")
    print(f"wrote {IMAGES / 'validation_and_certification.svg'}")
    print(f"wrote {IMAGES / 'benchmarks_overview.svg'}")


if __name__ == "__main__":
    main()
