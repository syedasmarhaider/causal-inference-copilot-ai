from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import (
    fig_to_png_bytes,
    fmt_pct,
    fmt_k,
    build_binary_treatment_from_protocol,
    protocol_WX_columns,
    fit_treatment_likelihood_scores,
)


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]


def generate_comparability_overlap_histogram_graph(
    df: pd.DataFrame,
    protocol: Any,
    *,
    include_effect_modifiers: bool = True,
    categorical_contrast: str | None = None,
    bins: int = 20,
    dpi: int = 160,
    strict: bool = True,
) -> GraphImage:
    """
    Clinician-friendly overlap histogram (NOT mirror).

    Shows distributions of:
      "Likelihood of receiving treatment (based on baseline profile)"
    for Treatment vs Control on the same axis.

    Assumes df is already filtered by exclusions upstream.
    """
    # 1) Build binary treatment indicator
    d, t, treat_label, dropped_t = build_binary_treatment_from_protocol(
        df,
        protocol,
        categorical_contrast=categorical_contrast,
    )

    n_treated = int((t == 1).sum())
    n_control = int((t == 0).sum())

    if (n_treated == 0) or (n_control == 0):
        if strict:
            raise ValueError(f"One group is empty after treatment parsing. treated={n_treated}, control={n_control}")
        fig, ax = plt.subplots(figsize=(11, 5.0))
        ax.axis("off")
        ax.set_title("Are there similar patients in both groups?")
        ax.text(0.5, 0.55, "Only one group is present in the current data.", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.text(0.5, 0.40, f"treated={n_treated}, control={n_control}", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        fig.tight_layout()
        return GraphImage(key="comparability_overlap", title="Group similarity (overlap)", mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))

    # 2) Baseline features (W + optional X)
    _W, _X, feature_cols = protocol_WX_columns(protocol, include_effect_modifiers=include_effect_modifiers)

    missing = _require_columns(d, feature_cols)
    if missing:
        if strict:
            raise ValueError(f"Missing baseline columns needed to compare patients: {missing}")
        feature_cols = [c for c in feature_cols if c in d.columns]

    if len(feature_cols) == 0:
        if strict:
            raise ValueError("No baseline features available (W/X empty or missing).")
        fig, ax = plt.subplots(figsize=(11, 5.0))
        ax.axis("off")
        ax.set_title("Are there similar patients in both groups?")
        ax.text(0.5, 0.55, "No baseline features available to compare patients.", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig.tight_layout()
        return GraphImage(key="comparability_overlap", title="Group similarity (overlap)", mime="image/png", content=fig_to_png_bytes(fig, dpi=dpi))

    # 3) Score per patient: likelihood of receiving treatment given baseline profile
    scores = fit_treatment_likelihood_scores(d, t, feature_cols)

    s_t = scores[t == 1]
    s_c = scores[t == 0]

    # 4) Comparable zone = overlapping score range
    lo = float(max(np.min(s_t), np.min(s_c)))
    hi = float(min(np.max(s_t), np.max(s_c)))

    outside_t = float(((s_t < lo) | (s_t > hi)).mean()) if len(s_t) else 0.0
    outside_c = float(((s_c < lo) | (s_c > hi)).mean()) if len(s_c) else 0.0
    inside_all = float(((scores >= lo) & (scores <= hi)).mean()) if (lo < hi) else 0.0

    # 5) Plot: overlapping histograms as “share within each group”
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_title(f"Are there similar patients in both groups? ({treat_label})")

    edges = np.linspace(0.0, 1.0, bins + 1)
    w_t = np.ones(len(s_t), dtype=float) / max(1, len(s_t))
    w_c = np.ones(len(s_c), dtype=float) / max(1, len(s_c))

    ax.hist(s_c, bins=edges, weights=w_c, alpha=0.55, label="Control group")
    ax.hist(s_t, bins=edges, weights=w_t, alpha=0.55, label="Treatment group")

    # Shade comparable zone + boundaries (minimal)
    if lo < hi:
        ax.axvspan(lo, hi, alpha=0.10, color="gray")
        ax.axvline(lo, linestyle="--", linewidth=1)
        ax.axvline(hi, linestyle="--", linewidth=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Likelihood of receiving the treatment (based on baseline profile)")
    ax.set_ylabel("Share of patients (within each group)")
    ax.legend(loc="upper right", frameon=True)

    ax.text(
        0.0,
        -0.20,
        f"Comparable zone covers {fmt_pct(inside_all)} of patients. "
        f"Outside zone: treatment {fmt_pct(outside_t)}, control {fmt_pct(outside_c)}.",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
    )
    if dropped_t > 0:
        ax.text(
            0.0,
            -0.30,
            f"{fmt_k(dropped_t)} rows ignored due to missing/unknown treatment values.",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )

    fig.tight_layout()
    return GraphImage(
        key="comparability_overlap",
        title="Group similarity (overlap)",
        mime="image/png",
        content=fig_to_png_bytes(fig, dpi=dpi),
    )