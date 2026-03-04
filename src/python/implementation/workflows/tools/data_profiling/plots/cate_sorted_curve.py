from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from python.implementation.workflows.tools.data_profiling.plots.utils import fig_to_png_bytes
from python.implementation.workflows.tools.data_profiling.plots.model import CohortCate, GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import align_finite_triplet


def plot_cate_sorted_curve(
    cohorts: Sequence[CohortCate],
    *,
    key: str = "cate_sorted_curve",
    title: str = "Sorted CATE curve (heterogeneity shape)",
    dpi: int = 160,
    max_points_per_cohort: int = 5000,
) -> GraphImage:
    """
    Heterogeneity “shape” plot:
      - sort cate values and plot vs percentile index
      - works for 1+ cohorts
      - if intervals exist, shades interval band (sorted by cate, same order applied)

    If n is huge, downsample deterministically to max_points_per_cohort.
    """
    fig: Figure = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)

    any_plotted = False

    for i, c in enumerate(cohorts):
        cate_f, lo_f, hi_f = align_finite_triplet(c.cate, c.lower, c.upper)
        if cate_f.size == 0:
            continue

        # downsample deterministically if huge
        if cate_f.size > max_points_per_cohort:
            idx = np.linspace(0, cate_f.size - 1, num=max_points_per_cohort).astype(int)
            cate_f = cate_f[idx]
            if lo_f is not None and hi_f is not None:
                lo_f = lo_f[idx]
                hi_f = hi_f[idx]

        order = np.argsort(cate_f)
        cate_s = cate_f[order]
        x = np.linspace(0.0, 100.0, num=cate_s.size)

        ax.plot(x, cate_s, linewidth=1, label=c.group_key)
        any_plotted = True

        if lo_f is not None and hi_f is not None and lo_f.size == cate_f.size:
            lo_s = lo_f[order]
            hi_s = hi_f[order]
            ax.fill_between(x, lo_s, hi_s, alpha=0.15)

    if not any_plotted:
        ax.text(0.5, 0.5, "No finite CATE values to plot.", ha="center", va="center")
        ax.set_axis_off()
        return GraphImage(key=key, title=title, mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Percentile (sorted individuals)")
    ax.set_ylabel("CATE")
    ax.set_title(title)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="best", frameon=True)

    return GraphImage(key=key, title=title, mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))