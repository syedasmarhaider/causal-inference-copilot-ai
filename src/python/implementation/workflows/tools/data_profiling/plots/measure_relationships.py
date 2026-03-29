from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import (
    fig_to_png_bytes,
    select_numeric_columns,
)

KEY = "numeric_correlation_heatmap"


def generate_measure_relationships_graph(
    df: pd.DataFrame,
    *,
    max_cols: int = 18,
    method: Literal["spearman", "pearson"] = "spearman",
    dpi: int = 160,
) -> GraphImage:
    """
    Clinician framing:
      - “Measures that move together”
      - selects numeric-ish columns (including numbers stored as text)
      - if too few, returns a clean explanatory image (not a technical error)
    """
    cols = select_numeric_columns(df, max_cols=max_cols)

    fig, ax = plt.subplots(figsize=(11, 8))

    if len(cols) < 3:
        ax.axis("off")
        ax.set_title("Measures that move together")
        ax.text(
            0.5,
            0.55,
            "Not enough numeric clinical measures to show relationships.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.text(
            0.5,
            0.42,
            "This can happen when numbers are stored as text or many fields are not recorded.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        fig.tight_layout()
        return GraphImage(
            key=KEY,
            title="Measure relationships",
            mime="image/png",
            content=fig_to_png_bytes(fig, dpi=dpi),
        )

    work = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = work.corr(method=method, min_periods=10).fillna(0.0).to_numpy(dtype=float)

    im = ax.imshow(corr, vmin=-1.0, vmax=1.0)
    ax.set_title("Measures that move together")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return GraphImage(
        key=KEY,
        title="Measure relationships",
        mime="image/png",
        content=fig_to_png_bytes(fig, dpi=dpi),
    )