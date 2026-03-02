from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import fig_to_png_bytes, fmt_pct, safe_missing_rate


KEY = "missingness_by_column"


def generate_data_completeness_graph(
    df: pd.DataFrame,
    *,
    top_k: int = 20,
    warn_line: float = 0.10,
    bad_line: float = 0.30,
    dpi: int = 160,
) -> GraphImage:
    """
    Clinician-first missingness plot:
      - horizontal bars
      - “Not recorded (%)”
      - only worst offenders (top_k)
      - simple thresholds: minor / serious
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    n_rows = int(df.shape[0])
    rows: List[Tuple[str, float]] = [(str(c), safe_missing_rate(df[c])) for c in df.columns]
    rows.sort(key=lambda x: x[1], reverse=True)
    shown = rows[:top_k]

    names = [r[0] for r in shown][::-1]
    vals = [r[1] for r in shown][::-1]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(names) + 2)))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Not recorded (%)")
    ax.set_title("Data Missingness (what’s not recorded)")

    for i, v in enumerate(vals):
        ax.text(min(v + 0.01, 0.98), i, fmt_pct(v), va="center", fontsize=9)

    n_serious = sum(1 for _, m in rows if m >= bad_line)
    ax.text(
        0.0,
        -0.10,
        f"{n_serious} variables have ≥{int(bad_line*100)}% not recorded (n={n_rows} patients).",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.tight_layout()
    return GraphImage(
        key=KEY,
        title="Data completeness",
        mime="image/png",
        content=fig_to_png_bytes(fig, dpi=dpi),
    )