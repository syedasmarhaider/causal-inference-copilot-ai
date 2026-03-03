from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from  python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.utils import (
    coerce_numeric_ratio,
    fig_to_png_bytes,
    protocol_WX_columns,
    build_binary_treatment_from_protocol,
    fit_treatment_likelihood_scores,
)

# ----------------------------
# helpers
# ----------------------------

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


def _rank_confounders(d: pd.DataFrame, t_bin: np.ndarray, candidates: Sequence[str]) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for c in candidates:
        s = d[c]
        if _is_numericish(s):
            score = _abs_corr_with_t(s, t_bin)
        else:
            score = _cramers_v(s, t_bin)
        if math.isfinite(score) and score > 0:
            scored.append((c, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _plot_propensity_vs_confounder(
    *,
    d: pd.DataFrame,
    scores: np.ndarray,
    confounder: str,
    control_name: str,
    treated_name: str,
) -> plt.Figure:
    x = d[confounder]
    fig = plt.figure(figsize=(10.5, 5.6))
    ax = fig.add_subplot(111)

    if _is_numericish(x):
        xv = pd.to_numeric(x, errors="coerce")
        m = xv.notna()

        # Density-friendly option: hexbin (avoids “blue cloud” overplotting)
        # If you prefer scatter, swap this with ax.scatter(..., alpha=0.2)
        ax.hexbin(xv[m].to_numpy(dtype=float), scores[m.to_numpy()], gridsize=45, mincnt=1)

        # Decile trend line (clinically readable)
        try:
            q = pd.qcut(xv[m], q=10, duplicates="drop")
            tmp = pd.DataFrame({"x": xv[m].to_numpy(dtype=float), "p": scores[m.to_numpy()], "bin": q})
            grp = tmp.groupby("bin", observed=True).agg(x_mean=("x", "mean"), p_mean=("p", "mean")).sort_values("x_mean")
            ax.plot(grp["x_mean"].to_numpy(), grp["p_mean"].to_numpy(), marker="o", linewidth=2)
        except Exception:
            pass

        ax.set_xlabel(f"{confounder} (baseline)")
        ax.set_ylabel("Likelihood of being treated (based on baseline profile)")
        ax.set_title("Treatment assignment pressure vs baseline driver")

    else:
        s = x.astype("object").fillna("MISSING")
        vc = s.value_counts(dropna=False)
        top = vc.index.astype(str).tolist()[:8]
        s2 = s.astype(str).where(s.astype(str).isin(top), other="Other")

        tmp = pd.DataFrame({"cat": s2, "p": scores})
        cats = tmp["cat"].value_counts().index.tolist()
        data = [tmp.loc[tmp["cat"] == c, "p"].to_numpy(dtype=float) for c in cats]

        ax.boxplot(data, labels=cats, showfliers=False)
        ax.set_xlabel(f"{confounder} (baseline categories)")
        ax.set_ylabel("Likelihood of being treated (based on baseline profile)")
        ax.set_title("Treatment assignment pressure differs by baseline category")

    fig.text(
        0.01,
        0.01,
        f"Control: {control_name}. Treated: {treated_name}. "
        f"Interpretation: near-0/near-1 propensities across ranges indicate weak positivity there.",
        fontsize=9,
    )
    return fig


# ----------------------------
# public API
# ----------------------------

def generate_propensity_vs_top_confounders_graphs(
    df: pd.DataFrame,
    protocol: Any,
    *,
    top_k: int = 4,
    confounders: Optional[Sequence[str]] = None,
    include_effect_modifiers_in_propensity: bool = True,
    key_prefix: str = "causal_propensity_vs",
) -> List[GraphImage]:
    """
    Graph (5) — but for multiple baseline drivers.
    Returns one GraphImage per confounder.

    - If confounders is None: auto-picks top_k baseline covariates (W) most associated with treatment.
    - Fits propensity scores once, reuses them for all plots.
    """
    d, t_bin, label, dropped = build_binary_treatment_from_protocol(df, protocol)
    control_name, treated_name = _split_label(label)

    W, X, feats = protocol_WX_columns(protocol, include_effect_modifiers=include_effect_modifiers_in_propensity)
    feats_present = _safe_cols_present(d, feats)
    if not feats_present:
        raise ValueError("No baseline columns (W/X) found in df to fit propensity scores.")

    scores = fit_treatment_likelihood_scores(d, t_bin, feats_present)

    W_present = _safe_cols_present(d, W)
    if confounders is None:
        # Auto-pick from W (confounders)
        ranked = _rank_confounders(d, t_bin, W_present or feats_present)
        picked = [c for c, _ in ranked[: max(1, int(top_k))]]
    else:
        picked = [c for c in confounders if c in d.columns]

    if not picked:
        raise ValueError("No confounders available to plot (check protocol W columns).")

    out: List[GraphImage] = []
    for c in picked:
        fig = _plot_propensity_vs_confounder(
            d=d,
            scores=scores,
            confounder=c,
            control_name=control_name,
            treated_name=treated_name,
        )
        out.append(
            GraphImage(
                key=f"{key_prefix}__{c}",
                title=f"Treatment assignment pressure vs {c}",
                mime="image/png",
                content= fig_to_png_bytes(fig),
            )
        )

    return out