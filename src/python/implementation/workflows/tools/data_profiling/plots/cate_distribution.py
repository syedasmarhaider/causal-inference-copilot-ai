from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from python.implementation.workflows.tools.data_profiling.plots.model import CohortCate, GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import (
    align_finite_triplet,
    fig_to_png_bytes,
)


def plot_cate_distribution(
    cohorts: Sequence[CohortCate],
    *,
    key: str = "cate_distribution",
    title: str = "CATE distribution",
    dpi: int = 160,
) -> GraphImage:
    """
    Distribution plot that behaves well for:
      - single cohort: histogram (n>30) else strip
      - multi cohort: stacked strip (small n) + ridge histogram (large n)

    Intervals are not required.
    """
    cleaned: list[tuple[str, np.ndarray]] = []
    for c in cohorts:
        cate_f, _, _ = align_finite_triplet(c.cate, None, None)
        if cate_f.size > 0:
            cleaned.append((c.group_key, cate_f))

    fig: Figure = plt.figure(figsize=(9, max(3.0, 1.2 * max(1, len(cleaned)))))
    ax = fig.add_subplot(111)

    if not cleaned:
        ax.text(0.5, 0.5, "No finite CATE values to plot.", ha="center", va="center")
        ax.set_axis_off()
        return GraphImage(key=key, title=title, mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))

    # Single cohort
    if len(cleaned) == 1:
        name, x = cleaned[0]
        n = int(x.size)
        ax.axvline(0.0, linewidth=1)

        if n <= 30:
            rng = np.random.default_rng(0)
            y = rng.normal(0.0, 0.03, size=n)
            ax.plot(x, y, marker="o", linestyle="None")
            ax.set_yticks([])
            ax.set_title(f"{title} (n={n}) — {name}")
        else:
            ax.hist(x, bins="auto", density=False)
            ax.set_title(f"{title} (n={n}) — {name}")
            ax.set_ylabel("Count")

        ax.set_xlabel("CATE")
        return GraphImage(key=key, title=title, mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))

    # Multiple cohorts
    ax.axvline(0.0, linewidth=1)
    y_base = np.arange(len(cleaned), dtype=float)
    rng = np.random.default_rng(0)

    for i, (_name, x) in enumerate(cleaned):
        n = int(x.size)
        if n <= 30:
            y = np.full(n, y_base[i]) + rng.normal(0.0, 0.08, size=n)
            ax.plot(x, y, marker="o", linestyle="None")
        else:
            counts, edges = np.histogram(x, bins="auto")
            if counts.max() > 0:
                counts = counts / counts.max()
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, y_base[i] + 0.35 * counts, linewidth=1)

    ax.set_yticks(y_base)
    ax.set_yticklabels([n for n, _ in cleaned])
    ax.set_xlabel("CATE")
    ax.set_title(f"{title} by cohort (dots for small n, ridges for large n)")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5)

    return GraphImage(key=key, title=title, mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))
