from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import (
    build_binary_treatment_from_protocol,
    fig_to_png_bytes,
    protocol_WX_columns,
)


def _split_label(label: str) -> tuple[str, str]:
    # label is produced by build_binary_treatment_from_protocol as: "control vs treated"
    parts = [p.strip() for p in label.split(" vs ")]
    if len(parts) == 2:
        return parts[0], parts[1]
    return "Control", "Treated"


def _safe_cols_present(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def generate_causal_missingness_by_group_graph(
    df: pd.DataFrame,
    protocol: Any,
    *,
    top_k: int = 12,
    include_effect_modifiers: bool = True,
    key: str = "causal_missingness_by_group",
) -> GraphImage:
    """
    Graph (4): Differential missingness by treatment group.
    Clinically interpretable: highlights documentation/measurement differences between groups.

    - Uses protocol to build a binary treatment indicator (and drops rows not in the contrast).
    - Checks missingness across baseline features (W and optionally X).
    """
    d, t_bin, label, dropped = build_binary_treatment_from_protocol(df, protocol)
    control_name, treated_name = _split_label(label)

    W, X, feats = protocol_WX_columns(protocol, include_effect_modifiers=include_effect_modifiers)
    cols = _safe_cols_present(d, feats)
    if not cols:
        raise ValueError("No baseline columns (W/X) found in df to compute missingness by group.")

    t_mask = (t_bin == 1)
    c_mask = (t_bin == 0)

    # Missingness per group
    miss_t = d.loc[t_mask, cols].isna().mean()
    miss_c = d.loc[c_mask, cols].isna().mean()
    gap = (miss_t - miss_c).abs()

    # Sort by clinically relevant signal: big between-group gaps first, then high overall missingness
    overall = d[cols].isna().mean()
    order = (
        pd.DataFrame({"gap": gap, "overall": overall})
        .sort_values(["gap", "overall"], ascending=[False, False])
        .index.tolist()
    )

    show = order[: max(1, min(int(top_k), len(order)))]
    miss_t_s = miss_t.reindex(show)
    miss_c_s = miss_c.reindex(show)

    y = np.arange(len(show))

    fig = plt.figure(figsize=(10.5, max(4.2, 0.35 * len(show) + 2.2)))
    ax = fig.add_subplot(111)

    # Use horizontal bars (not scatter) for cleaner comparison.
    bar_h = 0.34
    ax.barh(
        y - bar_h / 2,
        miss_c_s.to_numpy(dtype=float),
        height=bar_h,
        label=f"Control: {control_name}",
        alpha=0.85,
    )
    ax.barh(
        y + bar_h / 2,
        miss_t_s.to_numpy(dtype=float),
        height=bar_h,
        label=f"Treated: {treated_name}",
        alpha=0.85,
    )

    # Draw a thin connector between the two group values so gap remains obvious.
    for i in range(len(show)):
        x0 = float(miss_c_s.iat[i])
        x1 = float(miss_t_s.iat[i])
        ax.plot([x0, x1], [i, i], linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(show)
    ax.invert_yaxis()

    xmax = float(max(miss_t_s.max(), miss_c_s.max(), 0.05))
    ax.set_xlim(0.0, min(1.0, xmax + 0.05))

    ax.set_xlabel("Not recorded at baseline (fraction missing)")
    ax.set_title("Baseline data completeness differs by group (potential bias risk)")
    ax.grid(axis="x", alpha=0.2)

    # Small, clinical-friendly reference guides
    ax.axvline(0.1, linewidth=1)
    ax.axvline(0.2, linewidth=1)

    ax.legend(loc="lower right")

    kept_n = len(d)
    fig.text(
        0.01,
        0.01,
        f"Interpretation: Large gaps suggest different measurement/documentation workflows across groups.\n"
        f"Rows kept for contrast: {kept_n}. Dropped (not in contrast / missing treatment): {dropped}. "
        f"Top {len(show)} variables shown (sorted by gap).",
        fontsize=9,
    )

    return GraphImage(
        key=key,
        title="Differential missingness by group",
        mime="image/png",
        content=fig_to_png_bytes(fig),
    )
