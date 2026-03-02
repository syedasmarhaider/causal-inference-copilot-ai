from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python.domain.repo.data_repo import ImageMime
from python.implementation.workflows.tools.data_profiling.plots.model  import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils  import (
    fig_to_png_bytes,
    coerce_numeric_ratio,
    protocol_WX_columns,
    build_binary_treatment_from_protocol,
    fit_treatment_likelihood_scores,
)


def _split_label(label: str) -> Tuple[str, str]:
    parts = [p.strip() for p in label.split(" vs ")]
    if len(parts) == 2:
        return parts[0], parts[1]
    return "Control", "Treated"


def _safe_cols_present(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _is_numericish(s: pd.Series, *, thr: float = 0.9) -> bool:
    return bool(pd.api.types.is_numeric_dtype(s) or (coerce_numeric_ratio(s) >= thr))


def _abs_corr_with_t(x: pd.Series, t: np.ndarray) -> float:
    z = pd.to_numeric(x, errors="coerce")
    m = z.notna()
    if int(m.sum()) < 50:
        return 0.0
    a = z[m].to_numpy(dtype=float)
    b = t[m.to_numpy()].astype(float)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(c):
        return 0.0
    return abs(c)


def _cramers_v(x: pd.Series, t: np.ndarray) -> float:
    # Simple Cramér's V (no external deps); good enough for ranking.
    a = x.astype("object").fillna("MISSING")
    tab = pd.crosstab(a, pd.Series(t, name="t"))
    if tab.size == 0:
        return 0.0
    obs = tab.to_numpy(dtype=float)
    n = float(obs.sum())
    if n <= 0:
        return 0.0

    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    exp = (row_sums @ col_sums) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / exp)

    r, k = obs.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return 0.0
    v = math.sqrt(max(chi2 / n, 0.0) / denom)
    return float(v) if math.isfinite(v) else 0.0


def _pick_key_confounder(d: pd.DataFrame, t_bin: np.ndarray, candidates: Sequence[str]) -> str:
    # Prefer numeric-ish confounders by absolute correlation; fallback to categorical by Cramér's V.
    best_col = None
    best_score = -1.0

    # Pass 1: numeric-ish
    for c in candidates:
        s = d[c]
        if _is_numericish(s):
            score = _abs_corr_with_t(s, t_bin)
            if score > best_score:
                best_score = score
                best_col = c

    if best_col is not None and best_score > 0:
        return best_col

    # Pass 2: categorical-ish
    for c in candidates:
        s = d[c]
        score = _cramers_v(s, t_bin)
        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None:
        raise ValueError("Could not select a key confounder (no usable baseline columns).")
    return best_col


def generate_propensity_vs_key_confounder_graph(
    df: pd.DataFrame,
    protocol: Any,
    *,
    key_confounder: Optional[str] = None,
    include_effect_modifiers_in_propensity: bool = True,
    key: str = "causal_propensity_vs_key_confounder",
) -> GraphImage:
    """
    Graph (5): Propensity vs key confounder.

    - Fits propensity-like scores: P(T=1 | baseline profile) using your shared scorer.
    - Picks one clinically informative confounder automatically (unless provided).
    - Adds a decile trend line for interpretability.
    """
    d, t_bin, label, dropped = build_binary_treatment_from_protocol(df, protocol)
    control_name, treated_name = _split_label(label)

    W, X, feats = protocol_WX_columns(protocol, include_effect_modifiers=include_effect_modifiers_in_propensity)
    feats_present = _safe_cols_present(d, feats)
    if not feats_present:
        raise ValueError("No baseline columns (W/X) found in df to fit propensity scores.")

    scores = fit_treatment_likelihood_scores(d, t_bin, feats_present)

    # Choose confounder from W (confounders), not X by default
    W_present = _safe_cols_present(d, W)
    if key_confounder is None:
        if not W_present:
            # fallback: if no W, pick from available features
            W_present = feats_present
        key_confounder = _pick_key_confounder(d, t_bin, W_present)

    if key_confounder not in d.columns:
        raise ValueError(f"key_confounder '{key_confounder}' missing from df")

    x = d[key_confounder]

    fig = plt.figure(figsize=(10.5, 5.6))
    ax = fig.add_subplot(111)

    if _is_numericish(x):
        xv = pd.to_numeric(x, errors="coerce")
        m = xv.notna()
        ax.scatter(xv[m].to_numpy(dtype=float), scores[m.to_numpy()], s=10, alpha=0.25)

        # Decile trend line (more interpretable than a raw cloud)
        try:
            q = pd.qcut(xv[m], q=10, duplicates="drop")
            tmp = pd.DataFrame({"x": xv[m].to_numpy(dtype=float), "p": scores[m.to_numpy()], "bin": q})
            grp = tmp.groupby("bin", observed=True).agg(x_mean=("x", "mean"), p_mean=("p", "mean")).sort_values("x_mean")
            ax.plot(grp["x_mean"].to_numpy(), grp["p_mean"].to_numpy(), marker="o", linewidth=2)
        except Exception:
            # if qcut fails (too few unique), skip trend line
            pass

        ax.set_xlabel(f"{key_confounder} (baseline)")
        ax.set_ylabel("Likelihood of being treated (based on baseline profile)")
        ax.set_title("Treatment assignment pressure vs a key baseline driver")
    else:
        # Categorical confounder: show propensity distributions by category (top categories)
        s = x.astype("object").fillna("MISSING")
        vc = s.value_counts(dropna=False)
        top = vc.index.astype(str).tolist()[:8]
        s2 = s.astype(str).where(s.astype(str).isin(top), other="Other")

        tmp = pd.DataFrame({"cat": s2, "p": scores})
        cats = tmp["cat"].value_counts().index.tolist()

        data = [tmp.loc[tmp["cat"] == c, "p"].to_numpy(dtype=float) for c in cats]
        ax.boxplot(data, labels=cats, showfliers=False)
        ax.set_xlabel(f"{key_confounder} (baseline categories)")
        ax.set_ylabel("Likelihood of being treated (based on baseline profile)")
        ax.set_title("Treatment assignment differs by baseline category")

    fig.text(
        0.01,
        0.01,
        f"Interpretation: If scores approach 0 or 1 across wide confounder ranges, positivity is weak there.\n"
        f"Control: {control_name}. Treated: {treated_name}. Rows kept: {len(d)}. Dropped: {dropped}.",
        fontsize=9,
    )

    return GraphImage(
        key=key,
        title="Propensity vs key confounder",
        mime="image/png",
        content=fig_to_png_bytes(fig),
    )