from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from python.implementation.workflows.tools.data_profiling.plots.utils import fig_to_png_bytes
from python.implementation.workflows.tools.data_profiling.plots.model import CohortCate, GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import align_finite_triplet, bootstrap_ci_mean


def plot_cate_forest_mean_ci(
    cohorts: Sequence[CohortCate],
    *,
    key: str = "cate_forest_mean_ci",
    title: str = "Mean CATE by cohort (bootstrap CI)",
    dpi: int = 160,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> GraphImage:
    """
    Forest plot:
      - point = mean(CATE) per cohort
      - CI = bootstrap CI of the mean (robust and model-agnostic)

    This is the best “comparison” plot for multiple cohorts.
    """
    rows: List[Tuple[str, float, float, float, int]] = []
    for i, c in enumerate(cohorts):
        cate_f, _, _ = align_finite_triplet(c.cate, None, None)
        if cate_f.size == 0:
            continue
        mu, lo, hi = bootstrap_ci_mean(cate_f, n_boot=n_boot, alpha=alpha, seed=1000 + i)
        rows.append((c.group_key, mu, lo, hi, int(cate_f.size)))

    fig: Figure = plt.figure(figsize=(9, max(3.0, 0.55 * max(1, len(rows)) + 1.5)))
    ax = fig.add_subplot(111)

    if not rows:
        ax.text(0.5, 0.5, "No finite CATE values to plot.", ha="center", va="center")
        ax.set_axis_off()
        return GraphImage(key=key, title=title, mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))

    # Sort by mean for readability
    rows.sort(key=lambda r: r[1])
    names = [f"{nm} (n={n})" for nm, _, _, _, n in rows]
    means = np.array([m for _, m, _, _, _ in rows], dtype=float)
    lows  = np.array([l for _, _, l, _, _ in rows], dtype=float)
    highs = np.array([h for _, _, _, h, _ in rows], dtype=float)

    y = np.arange(len(rows), dtype=float)

    ax.axvline(0.0, linewidth=1)
    ax.errorbar(
        means,
        y,
        xerr=[means - lows, highs - means],
        fmt="o",
        capsize=3,
        linestyle="None",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean CATE")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5)

    return GraphImage(key=key, title=title, mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))