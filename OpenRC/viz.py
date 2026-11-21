"""Plots: regions in 2-D and 3-D, growth curves, and compression trade-offs.

Matplotlib is an optional dependency imported lazily inside each plotting function.

Higher-dimensional regions can be plotted either by orthogonal projection onto
selected axes (:meth:`Polytope.subspace`, default) or by taking an exact
cross-section at ``fixed`` values of the remaining axes (:meth:`Polytope.slice`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Optional, Sequence, Union

import numpy as np

from OpenRC.cases.axes import AxisSpec
from OpenRC.geometry.polytope import Polytope


def plot_region_2d(region: Polytope, *, ax=None,
                   dims: Sequence[Union[int, AxisSpec]] = (0, 1),
                   fixed: Optional[Mapping[Union[int, AxisSpec], float]] = None,
                   color: str = "#377eb8", alpha: float = 0.25,
                   show_vertices: bool = True, label: Optional[str] = None):
    """Draw a 2-D region (or 2-D projection / cross-section) as a filled polygon."""
    import matplotlib.pyplot as plt

    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    if (isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, Real))
            or not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be a finite number in [0, 1]")
    if not isinstance(show_vertices, (bool, np.bool_)):
        raise ValueError("show_vertices must be a boolean")
    if isinstance(dims, (str, bytes, bytearray)):
        raise TypeError("dims must be a sequence of 2 axis selectors, not a string")
    dims_list = list(dims)
    if len(dims_list) != 2:
        raise ValueError(f"need exactly 2 dimensions to plot, got {len(dims_list)}")
    base = region.slice(fixed) if fixed is not None else region
    is_default_ints = (
        base.dim == 2
        and dims_list == [0, 1]
        and all(not isinstance(d, (bool, np.bool_)) and isinstance(d, Integral) for d in dims_list)
    )
    body = base if is_default_ints else base.subspace(dims_list)
    if body.dim != 2:
        raise ValueError(f"need exactly 2 dimensions to plot, got {body.dim}")

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))

    verts = body.hull_vertices
    if body.hull is not None and len(verts) >= 3:
        centre = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - centre[1], verts[:, 0] - centre[0])
        ordered = verts[np.argsort(angles)]
        closed = np.vstack([ordered, ordered[0]])
        ax.fill(ordered[:, 0], ordered[:, 1], color=color, alpha=float(alpha),
                label=label if not show_vertices else None)
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.4)
    if show_vertices:
        ax.scatter(verts[:, 0], verts[:, 1], s=14, color=color, zorder=3, label=label)

    labels = (body.axes.pretty_labels if body.axes is not None
              else [f"x{i}" for i in range(2)])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    lower, upper = body.bounding_box(0.1)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.grid(alpha=0.3)
    return ax


def plot_region_3d(region: Polytope, *, ax=None,
                   dims: Sequence[Union[int, AxisSpec]] = (0, 1, 2),
                   fixed: Optional[Mapping[Union[int, AxisSpec], float]] = None,
                   color: str = "#377eb8", alpha: float = 0.25,
                   show_vertices: bool = True, label: Optional[str] = None):
    """Draw a (projection or slice of a) region as a translucent 3-D hull."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if not isinstance(region, Polytope):
        raise TypeError("region must be a Polytope")
    if (isinstance(alpha, (bool, np.bool_))
            or not isinstance(alpha, (int, float, Real))
            or not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be a finite number in [0, 1]")
    if not isinstance(show_vertices, (bool, np.bool_)):
        raise ValueError("show_vertices must be a boolean")
    if isinstance(dims, (str, bytes, bytearray)):
        raise TypeError("dims must be a sequence of 3 axis selectors, not a string")
    dims_list = list(dims)
    if len(dims_list) != 3:
        raise ValueError(f"need exactly 3 dimensions to plot, got {len(dims_list)}")
    base = region.slice(fixed) if fixed is not None else region
    is_default_ints = (
        base.dim == 3
        and dims_list == [0, 1, 2]
        and all(not isinstance(d, (bool, np.bool_)) and isinstance(d, Integral) for d in dims_list)
    )
    body = base if is_default_ints else base.subspace(dims_list)
    if body.dim != 3:
        raise ValueError(f"need exactly 3 dimensions to plot, got {body.dim}")

    if ax is None:
        figure = plt.figure(figsize=(6, 6))
        ax = figure.add_subplot(111, projection="3d")

    points = body.vertices
    verts = body.hull_vertices
    hull = body.hull
    if hull is not None:
        surface = Poly3DCollection(
            [points[simplex] for simplex in hull.simplices],
            label=label if not show_vertices else None,
        )
        surface.set_color(color)
        surface.set_alpha(float(alpha))
        surface.set_edgecolor((0, 0, 0, 0.15))
        ax.add_collection3d(surface)
    if show_vertices:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=8, color=color,
                   depthshade=False, label=label)

    labels = (body.axes.pretty_labels if body.axes is not None
              else [f"x{i}" for i in range(3)])
    ax.set_xlabel(labels[0], labelpad=10)
    ax.set_ylabel(labels[1], labelpad=10)
    ax.set_zlabel(labels[2], labelpad=10)
    lower, upper = body.bounding_box(0.1)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    return ax


def plot_growth(results: dict, *, ax=None, x: str = "n_vertices", y: str = "volume"):
    """Growth curves for several builds, keyed by label.

    Pass ``x="oracle_calls"`` to plot hypervolume against cumulative power-flow
    evaluations instead of vertex count.
    """
    import matplotlib.pyplot as plt

    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping of label -> result")
    if not isinstance(x, str) or not x.strip() or not isinstance(y, str) or not y.strip():
        raise ValueError("x and y must be non-empty column names")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, result in results.items():
        frame = result.to_frame() if hasattr(result, "to_frame") else result
        if len(frame) == 0:
            continue
        ax.plot(frame[x], frame[y], label=label, linewidth=1.8)
        plotted = True
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    if plotted:
        ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_compression(results: dict, *, ax=None):
    """Retained hypervolume against vertex count, one curve per compressor."""
    import matplotlib.pyplot as plt

    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping of label -> result")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, result in results.items():
        frame = result.to_frame() if hasattr(result, "to_frame") else result
        if len(frame) == 0:
            continue
        ax.plot(frame["n_vertices"], 100.0 - frame["volume_loss_pct"],
                marker="o", markersize=3, label=label)
        plotted = True
    ax.set_xlabel("vertices remaining")
    ax.set_ylabel("hypervolume retained (%)")
    ax.invert_xaxis()
    if plotted:
        ax.legend()
    ax.grid(alpha=0.3)
    return ax
