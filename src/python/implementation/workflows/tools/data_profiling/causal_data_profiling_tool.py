from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Headless-safe matplotlib
import matplotlib

from python.domain.workflows.tool import Tool
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_bool_dtype, is_numeric_dtype



class CausalDataProfilingTool(Tool):
    NAME: ClassVar[str] = "CAUSAL_DATA_PROFILING"
    
    def get_tool_name(self) -> str:
        return self.NAME
    
    def get_tool_info(self) -> str:
        return "Tool for generating causal data profiling graphs, such as propensity score overlap, covariate balance (Love) plots, and weight distribution plots. These graphs help diagnose potential issues with causal inference analyses, such as lack of common support or extreme weights."

    def generate_causal_graphs(
        self,
        df: pd.DataFrame,
        protocol: Any,
        *,
        compute_quantiles: bool = True,
        strict: bool = True,
    ) -> List[CausalGraphImage]:
        """
        Generates a set of causal data profiling graphs based on the input DataFrame and protocol.
        - df: The dataset to profile, after exclusions and preprocessing.
        - protocol: The causal protocol containing treatment, covariate, and outcome specifications.
        - compute_quantiles: Whether to compute quantiles for numeric covariates (may be expensive).
        - strict: If True, raises errors for common issues (e.g., missing columns, invalid treatment values). If False, tries to proceed with best effort and may produce less accurate graphs.
        
        Returns a list of CausalGraphImage objects containing the generated graphs.
        """
        love_plot = generate_love_plot_graphs(
            df=df,
            protocol=protocol,
        )
        
        overlap_plots = generate_overlap_graphs(
            df=df,
            protocol=protocol,
        )
        
        weight_dist_plots = generate_weight_stability_graphs(
            df=df,
            protocol=protocol,
        )
    
        return love_plot + overlap_plots + weight_dist_plots





# =============================================================================
# Types (reuse your existing ProtocolSpec/TreatmentSpec models at call sites)
# =============================================================================

CausalImageMime = Literal["image/png", "image/jpeg", "image/webp"]

@dataclass(frozen=True)
class CausalGraphImage:
    key: str
    title: str
    mime: CausalImageMime
    content: bytes


# =============================================================================
# Errors (structured and parseable)
# =============================================================================

@dataclass(frozen=True)
class CausalDiagErrorDetails:
    reason: str
    hint: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None

class CausalDiagnosticsError(RuntimeError):
    def __init__(self, details: CausalDiagErrorDetails):
        self.details = details
        msg = details.reason
        if details.hint:
            msg = f"{msg} Hint: {details.hint}"
        super().__init__(msg)


# =============================================================================
# Shared cohort + preprocessing helpers
# =============================================================================

def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise CausalDiagnosticsError(
            CausalDiagErrorDetails(
                reason="Dataset missing required columns.",
                hint="Ensure protocol columns exist in df.",
                evidence={"missing_columns": missing},
            )
        )

def _apply_exclusions(df: pd.DataFrame, protocol: Any) -> pd.DataFrame:
    """
    Minimal exclusions implementation placeholder:
    - Your project likely already has a deterministic DataProcessingTool.apply_exclusion_rules.
    - Here we simply return df unchanged if no exclusions or tool not wired.
    """
    # If you have the real tool, call it here.
    return df

def _parse_binary_treatment(series: pd.Series, treated: str, control: str, *, strict: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t (0/1) and mask_kept (rows kept)
    """
    s = series.astype("object")
    is_missing = s.isna()
    valid = (~is_missing) & (s.isin([treated, control]))
    if strict:
        bad = (~is_missing) & (~s.isin([treated, control]))
        if bool(bad.any()):
            bad_vals = sorted(set(map(str, s[bad].unique().tolist())))
            raise CausalDiagnosticsError(
                CausalDiagErrorDetails(
                    reason="Treatment column contains values outside {treated, control}.",
                    hint="Fix protocol.treated/control or clean dataset values.",
                    evidence={"bad_values_sample": bad_vals[:20], "treated": treated, "control": control},
                )
            )

    kept = valid.to_numpy(dtype=bool)
    t = np.where(s.to_numpy(dtype=object) == treated, 1, 0).astype(int)
    return t, kept

def _build_covariate_frame(df: pd.DataFrame, covariates: Sequence[str]) -> pd.DataFrame:
    """
    Adds explicit missingness indicators: <col>__missing (0/1).
    Keeps original covariate names untouched for interpretability in plots.
    """
    W = df.loc[:, list(covariates)].copy()

    # Normalize booleans to {0,1} for numeric handling + SMD
    for c in W.columns:
        if is_bool_dtype(W[c]):
            W[c] = W[c].astype("float")  # may become NaN for missing, OK
    for c in covariates:
        W[f"{c}__missing"] = df[c].isna().astype(int)

    return W

def _split_numeric_categorical(W: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    cat_cols: List[str] = []
    for c in W.columns:
        # missing indicators are numeric by construction
        if c.endswith("__missing"):
            numeric_cols.append(c)
            continue
        if is_numeric_dtype(W[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)
    return numeric_cols, cat_cols

def _fit_propensity_binary(
    W: pd.DataFrame,
    t: np.ndarray,
    *,
    max_iter: int = 2000,
    C: float = 1.0,
    eps: float = 1e-3,
    random_state: int = 0,
) -> np.ndarray:
    """
    Fits a regularized logistic propensity model and returns clipped e(W).
    Deterministic pipeline (no stochastic learner).
    """
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if len(W) != len(t):
        raise ValueError("W and t length mismatch")

    n1 = int((t == 1).sum())
    n0 = int((t == 0).sum())
    if n1 == 0 or n0 == 0:
        raise CausalDiagnosticsError(
            CausalDiagErrorDetails(
                reason="Cannot fit propensity model: one treatment arm has zero rows.",
                evidence={"n_treated": n1, "n_control": n0},
            )
        )

    numeric_cols, cat_cols = _split_numeric_categorical(W)

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(W, t)

    e = pipe.predict_proba(W)[:, 1].astype(float)
    e = np.clip(e, eps, 1.0 - eps)
    return e

def _stabilized_ipw(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    p1 = float((t == 1).mean())
    p0 = 1.0 - p1
    w = np.where(t == 1, p1 / e, p0 / (1.0 - e)).astype(float)
    return w

def _ess(weights: np.ndarray) -> float:
    s1 = float(weights.sum())
    s2 = float((weights ** 2).sum())
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2

def _fig_to_png_bytes(fig: Figure, *, dpi: int = 160) -> bytes:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)


# =============================================================================
# SMD / balance helpers (clinically interpretable aggregation)
# =============================================================================

def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    sw = float(w.sum())
    if sw <= 0:
        return float("nan")
    return float((x * w).sum() / sw)

def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mu = _weighted_mean(x, w)
    if not math.isfinite(mu):
        return float("nan")
    sw = float(w.sum())
    if sw <= 0:
        return float("nan")
    return float((w * (x - mu) ** 2).sum() / sw)

def _smd_numeric(x: np.ndarray, t: np.ndarray, *, w: Optional[np.ndarray] = None) -> float:
    """
    Standardized mean difference for numeric x.
    If w is provided, uses weighted mean/var within groups.
    """
    mask1 = (t == 1)
    mask0 = (t == 0)
    x1 = x[mask1]
    x0 = x[mask0]
    if w is None:
        m1 = float(np.nanmean(x1)) if len(x1) else float("nan")
        m0 = float(np.nanmean(x0)) if len(x0) else float("nan")
        v1 = float(np.nanvar(x1)) if len(x1) else float("nan")
        v0 = float(np.nanvar(x0)) if len(x0) else float("nan")
    else:
        w1 = w[mask1]
        w0 = w[mask0]
        m1 = _weighted_mean(x1, w1)
        m0 = _weighted_mean(x0, w0)
        v1 = _weighted_var(x1, w1)
        v0 = _weighted_var(x0, w0)

    denom = math.sqrt(max(1e-12, 0.5 * (v1 + v0))) if (math.isfinite(v1) and math.isfinite(v0)) else float("nan")
    if not math.isfinite(denom) or denom <= 0:
        return 0.0
    return float((m1 - m0) / denom)

def _smd_binary(p1: float, p0: float) -> float:
    denom = math.sqrt(max(1e-12, 0.5 * (p1 * (1 - p1) + p0 * (1 - p0))))
    return float((p1 - p0) / denom) if denom > 0 else 0.0

def _balance_table_max_smd_per_covariate(
    df: pd.DataFrame,
    covariates: Sequence[str],
    t: np.ndarray,
    *,
    weights: Optional[np.ndarray],
) -> List[Tuple[str, float]]:
    """
    Returns list of (covariate_name, max_abs_smd_for_that_covariate).
    - Numeric: direct SMD on imputed numeric values.
    - Categorical: compute SMD per level, take max abs.
    - Also includes <cov>__missing as its own variable (binary SMD).
    """
    rows: List[Tuple[str, float]] = []

    # Work on imputed copies for numeric/categorical interpretability
    for cov in covariates:
        s = df[cov]
        miss = s.isna().to_numpy(dtype=bool)

        # Missing indicator SMD (binary)
        xmiss = miss.astype(float)
        if weights is None:
            p1 = float(xmiss[t == 1].mean()) if (t == 1).any() else 0.0
            p0 = float(xmiss[t == 0].mean()) if (t == 0).any() else 0.0
        else:
            w = weights
            w1 = w[t == 1]; w0 = w[t == 0]
            p1 = float((xmiss[t == 1] * w1).sum() / max(1e-12, w1.sum()))
            p0 = float((xmiss[t == 0] * w0).sum() / max(1e-12, w0.sum()))
        rows.append((f"{cov}__missing", abs(_smd_binary(p1, p0))))

        # Covariate SMD
        if is_bool_dtype(s) or is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            med = float(np.nanmedian(x.to_numpy(dtype=float))) if np.isfinite(np.nanmedian(x.to_numpy(dtype=float))) else 0.0
            x_imp = x.fillna(med).to_numpy(dtype=float)
            smd = _smd_numeric(x_imp, t, w=weights)
            rows.append((cov, abs(smd)))
        else:
            # categorical: impute missing as "MISSING"
            x = s.astype("object").fillna("MISSING").astype(str).to_numpy(dtype=object)

            # levels observed (cap for stability)
            uniq = pd.unique(x)
            # If huge, cap to most frequent and lump the rest into OTHER
            if len(uniq) > 50:
                vc = pd.Series(x).value_counts()
                keep = set(vc.index[:49].astype(str).tolist())
                x = np.array([v if v in keep else "OTHER" for v in x], dtype=object)
                uniq = pd.unique(x)

            max_abs = 0.0
            for lvl in uniq:
                z = (x == lvl).astype(float)
                if weights is None:
                    p1 = float(z[t == 1].mean()) if (t == 1).any() else 0.0
                    p0 = float(z[t == 0].mean()) if (t == 0).any() else 0.0
                else:
                    w = weights
                    w1 = w[t == 1]; w0 = w[t == 0]
                    p1 = float((z[t == 1] * w1).sum() / max(1e-12, w1.sum()))
                    p0 = float((z[t == 0] * w0).sum() / max(1e-12, w0.sum()))
                max_abs = max(max_abs, abs(_smd_binary(p1, p0)))

            rows.append((cov, float(max_abs)))

    return rows


# =============================================================================
# Plot builders
# =============================================================================

def _plot_overlap_propensity(
    e: np.ndarray,
    t: np.ndarray,
    *,
    title: str,
    bins: int = 20,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("Propensity score e(W)")
    ax.set_ylabel("Density")

    e1 = e[t == 1]
    e0 = e[t == 0]

    # Common support interval
    lo = float(max(e1.min(initial=1.0), e0.min(initial=1.0)))
    hi = float(min(e1.max(initial=0.0), e0.max(initial=0.0)))
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))

    ax.hist(e0, bins=bins, range=(0.0, 1.0), density=True, alpha=0.6, label="Control")
    ax.hist(e1, bins=bins, range=(0.0, 1.0), density=True, alpha=0.6, label="Treated")

    ax.axvline(lo, linestyle="--")
    ax.axvline(hi, linestyle="--")
    ax.legend(loc="best")

    # Outside support percentages
    out1 = float(((e1 < lo) | (e1 > hi)).mean()) if len(e1) else 0.0
    out0 = float(((e0 < lo) | (e0 > hi)).mean()) if len(e0) else 0.0

    ax.text(
        0.02,
        0.98,
        f"Common support: [{lo:.3f}, {hi:.3f}]\nOutside support: treated {out1*100:.1f}%, control {out0*100:.1f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    return fig

def _plot_love(
    smd_pre: List[Tuple[str, float]],
    smd_post: List[Tuple[str, float]],
    *,
    title: str,
    max_items: int = 35,
    threshold: float = 0.1,
) -> Figure:
    # Merge into dicts
    pre = dict(smd_pre)
    post = dict(smd_post)

    keys = sorted(set(pre.keys()) | set(post.keys()))
    rows = [(k, float(pre.get(k, 0.0)), float(post.get(k, 0.0))) for k in keys]

    # Sort by post imbalance desc
    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    shown = rows[:max_items]

    names = [r[0] for r in shown][::-1]
    x_pre = [r[1] for r in shown][::-1]
    x_post = [r[2] for r in shown][::-1]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(names) + 2)))
    ax.set_title(title)
    ax.set_xlabel("Absolute standardized mean difference |SMD|")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)

    ax.scatter(x_pre, y, label="Pre-adjustment", alpha=0.8)
    ax.scatter(x_post, y, label="Post-IPW", alpha=0.8)

    ax.axvline(threshold, linestyle="--")
    ax.text(threshold, len(names) - 0.5, f"  threshold {threshold:.2f}", va="center", fontsize=9)

    ax.set_xlim(left=0.0)
    ax.legend(loc="best", fontsize=9)

    if len(rows) > max_items:
        ax.text(
            0.0,
            -0.10,
            f"Showing top {max_items} covariates by post-IPW imbalance out of {len(rows)}.",
            transform=ax.transAxes,
            fontsize=9,
        )

    fig.tight_layout()
    return fig

def _plot_weight_stability(
    w: np.ndarray,
    t: np.ndarray,
    *,
    title: str,
    bins: int = 30,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("Stabilized IPW weight")
    ax.set_ylabel("Count")

    w1 = w[t == 1]
    w0 = w[t == 0]

    # For readability, clip x-axis at p99 of combined weights
    w_all = np.concatenate([w0, w1]) if len(w0) and len(w1) else w
    x_max = float(np.quantile(w_all, 0.99)) if len(w_all) else 1.0
    x_max = max(1e-6, x_max)

    ax.hist(np.clip(w0, 0, x_max), bins=bins, alpha=0.6, label="Control")
    ax.hist(np.clip(w1, 0, x_max), bins=bins, alpha=0.6, label="Treated")
    ax.legend(loc="best")

    ess1 = _ess(w1) if len(w1) else 0.0
    ess0 = _ess(w0) if len(w0) else 0.0

    ax.text(
        0.02,
        0.98,
        f"Display clipped at p99={x_max:.2f}\n"
        f"ESS treated={ess1:.1f} / n={len(w1)}; ESS control={ess0:.1f} / n={len(w0)}\n"
        f"p99 weight (raw)={float(np.quantile(w_all, 0.99)):.2f}  max weight (raw)={float(np.max(w_all)):.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    return fig


# =============================================================================
# THREE helper functions (what you asked for)
# =============================================================================

def generate_overlap_graphs(
    df: pd.DataFrame,
    protocol: Any,
    *,
    strict: bool = True,
    min_n_per_arm: int = 30,
    dpi: int = 160,
) -> List[CausalGraphImage]:
    """
    Graph 1: Overlap / common support.
    - Binary treatment: 1 plot.
    - Categorical treatment: baseline vs each non-baseline level (1 plot per contrast).
    """
    if min_n_per_arm < 5:
        raise ValueError("min_n_per_arm too small")

    df0 = _apply_exclusions(df, protocol)

    covariates: List[str] = list(protocol.covariates)
    t_spec = protocol.treatment_spec
    t_col = str(t_spec.column)

    _require_columns(df0, [t_col, *covariates])

    W = _build_covariate_frame(df0, covariates)

    out: List[CausalGraphImage] = []

    if t_spec.kind == "binary":
        t_raw = df0[t_col]
        t, kept = _parse_binary_treatment(t_raw, treated=str(t_spec.treated), control=str(t_spec.control), strict=strict)
        Wk = W.loc[kept].reset_index(drop=True)
        tk = t[kept]

        if int((tk == 1).sum()) < min_n_per_arm or int((tk == 0).sum()) < min_n_per_arm:
            raise CausalDiagnosticsError(
                CausalDiagErrorDetails(
                    reason="Insufficient cohort size per arm for overlap diagnostics.",
                    evidence={"n_treated": int((tk == 1).sum()), "n_control": int((tk == 0).sum())},
                )
            )

        e = _fit_propensity_binary(Wk, tk)
        fig = _plot_overlap_propensity(e, tk, title=f"Overlap: {t_col} ({t_spec.control} vs {t_spec.treated})")
        out.append(
            CausalGraphImage(
                key="overlap_propensity_binary",
                title="Overlap / common support",
                mime="image/png",
                content=_fig_to_png_bytes(fig, dpi=dpi),
            )
        )
        return out

    # categorical -> baseline contrasts
    levels = list(t_spec.levels)
    baseline = str(levels[0])
    for lvl in levels[1:]:
        lvl = str(lvl)
        mask = df0[t_col].isin([baseline, lvl]) & (~df0[t_col].isna())
        dfk = df0.loc[mask].copy()
        if dfk.empty:
            continue

        Wk = _build_covariate_frame(dfk, covariates)
        tk = (dfk[t_col].astype("object") == lvl).to_numpy(dtype=int)

        if int((tk == 1).sum()) < min_n_per_arm or int((tk == 0).sum()) < min_n_per_arm:
            if strict:
                raise CausalDiagnosticsError(
                    CausalDiagErrorDetails(
                        reason="Insufficient cohort size per arm for a categorical contrast.",
                        hint="Consider merging rare treatment levels or changing baseline.",
                        evidence={"baseline": baseline, "level": lvl, "n_level": int((tk == 1).sum()), "n_baseline": int((tk == 0).sum())},
                    )
                )
            continue

        e = _fit_propensity_binary(Wk, tk)
        fig = _plot_overlap_propensity(e, tk, title=f"Overlap: {t_col} ({baseline} vs {lvl})")
        out.append(
            CausalGraphImage(
                key=f"overlap_propensity_{baseline}_vs_{lvl}",
                title=f"Overlap / common support ({baseline} vs {lvl})",
                mime="image/png",
                content=_fig_to_png_bytes(fig, dpi=dpi),
            )
        )

    if strict and not out:
        raise CausalDiagnosticsError(
            CausalDiagErrorDetails(
                reason="No categorical contrasts could be plotted.",
                hint="Check treatment levels and data availability.",
                evidence={"levels": levels},
            )
        )
    return out


def generate_love_plot_graphs(
    df: pd.DataFrame,
    protocol: Any,
    *,
    strict: bool = True,
    min_n_per_arm: int = 30,
    dpi: int = 160,
    max_items: int = 35,
    threshold: float = 0.1,
) -> List[CausalGraphImage]:
    """
    Graph 2: Love plot (covariate balance pre vs post-IPW).
    - Uses stabilized IPW from propensity model.
    - Categorical treatment -> baseline vs each non-baseline level.
    """
    df0 = _apply_exclusions(df, protocol)

    covariates: List[str] = list(protocol.covariates)
    t_spec = protocol.treatment_spec
    t_col = str(t_spec.column)

    _require_columns(df0, [t_col, *covariates])

    out: List[CausalGraphImage] = []

    def _one(dfk: pd.DataFrame, *, title: str, key: str) -> CausalGraphImage:
        Wk = _build_covariate_frame(dfk, covariates)
        # Binary group coding must already be in dfk as tk
        return CausalGraphImage(key=key, title=title, mime="image/png", content=b"")

    if t_spec.kind == "binary":
        t, kept = _parse_binary_treatment(df0[t_col], treated=str(t_spec.treated), control=str(t_spec.control), strict=strict)
        dfk = df0.loc[kept].reset_index(drop=True)
        tk = t[kept]
        if int((tk == 1).sum()) < min_n_per_arm or int((tk == 0).sum()) < min_n_per_arm:
            raise CausalDiagnosticsError(
                CausalDiagErrorDetails(
                    reason="Insufficient cohort size per arm for balance diagnostics.",
                    evidence={"n_treated": int((tk == 1).sum()), "n_control": int((tk == 0).sum())},
                )
            )

        Wk = _build_covariate_frame(dfk, covariates)
        e = _fit_propensity_binary(Wk, tk)
        w = _stabilized_ipw(tk, e)

        smd_pre = _balance_table_max_smd_per_covariate(dfk, covariates, tk, weights=None)
        smd_post = _balance_table_max_smd_per_covariate(dfk, covariates, tk, weights=w)

        fig = _plot_love(
            smd_pre, smd_post,
            title=f"Covariate balance (Love plot): {t_col} ({t_spec.control} vs {t_spec.treated})",
            max_items=max_items,
            threshold=threshold,
        )
        out.append(
            CausalGraphImage(
                key="love_plot_binary",
                title="Covariate balance (Love plot)",
                mime="image/png",
                content=_fig_to_png_bytes(fig, dpi=dpi),
            )
        )
        return out

    # categorical contrasts
    levels = list(t_spec.levels)
    baseline = str(levels[0])
    for lvl in levels[1:]:
        lvl = str(lvl)
        mask = df0[t_col].isin([baseline, lvl]) & (~df0[t_col].isna())
        dfk = df0.loc[mask].reset_index(drop=True)
        if dfk.empty:
            continue
        tk = (dfk[t_col].astype("object") == lvl).to_numpy(dtype=int)

        if int((tk == 1).sum()) < min_n_per_arm or int((tk == 0).sum()) < min_n_per_arm:
            if strict:
                raise CausalDiagnosticsError(
                    CausalDiagErrorDetails(
                        reason="Insufficient cohort size per arm for a categorical contrast (balance).",
                        evidence={"baseline": baseline, "level": lvl, "n_level": int((tk == 1).sum()), "n_baseline": int((tk == 0).sum())},
                    )
                )
            continue

        Wk = _build_covariate_frame(dfk, covariates)
        e = _fit_propensity_binary(Wk, tk)
        w = _stabilized_ipw(tk, e)

        smd_pre = _balance_table_max_smd_per_covariate(dfk, covariates, tk, weights=None)
        smd_post = _balance_table_max_smd_per_covariate(dfk, covariates, tk, weights=w)

        fig = _plot_love(
            smd_pre, smd_post,
            title=f"Covariate balance (Love plot): {t_col} ({baseline} vs {lvl})",
            max_items=max_items,
            threshold=threshold,
        )
        out.append(
            CausalGraphImage(
                key=f"love_plot_{baseline}_vs_{lvl}",
                title=f"Covariate balance (Love plot) ({baseline} vs {lvl})",
                mime="image/png",
                content=_fig_to_png_bytes(fig, dpi=dpi),
            )
        )

    if strict and not out:
        raise CausalDiagnosticsError(
            CausalDiagErrorDetails(
                reason="No categorical contrasts could be plotted (balance).",
                evidence={"levels": levels},
            )
        )
    return out


def generate_weight_stability_graphs(
    df: pd.DataFrame,
    protocol: Any,
    *,
    strict: bool = True,
    min_n_per_arm: int = 30,
    dpi: int = 160,
) -> List[CausalGraphImage]:
    """
    Graph 3: Weight stability (stabilized IPW histogram + ESS).
    - Uses stabilized IPW from propensity model.
    - Categorical treatment -> baseline vs each non-baseline level.
    """
    df0 = _apply_exclusions(df, protocol)

    covariates: List[str] = list(protocol.covariates)
    t_spec = protocol.treatment_spec
    t_col = str(t_spec.column)

    _require_columns(df0, [t_col, *covariates])

    out: List[CausalGraphImage] = []

    if t_spec.kind == "binary":
        t, kept = _parse_binary_treatment(df0[t_col], treated=str(t_spec.treated), control=str(t_spec.control), strict=strict)
        dfk = df0.loc[kept].reset_index(drop=True)
        tk = t[kept]

        if int((tk == 1).sum()) < min_n_per_arm or int((tk == 0).sum()) < min_n_per_arm:
            raise CausalDiagnosticsError(
                CausalDiagErrorDetails(
                    reason="Insufficient cohort size per arm for weight diagnostics.",
                    evidence={"n_treated": int((tk == 1).sum()), "n_control": int((tk == 0).sum())},
                )
            )

        Wk = _build_covariate_frame(dfk, covariates)
        e = _fit_propensity_binary(Wk, tk)
        w = _stabilized_ipw(tk, e)

        fig = _plot_weight_stability(
            w, tk,
            title=f"Weight stability (stabilized IPW): {t_col} ({t_spec.control} vs {t_spec.treated})",
        )
        out.append(
            CausalGraphImage(
                key="weight_stability_binary",
                title="Weight stability (IPW)",
                mime="image/png",
                content=_fig_to_png_bytes(fig, dpi=dpi),
            )
        )
        return out

    # categorical contrasts
    levels = list(t_spec.levels)
    baseline = str(levels[0])
    for lvl in levels[1:]:
        lvl = str(lvl)
        mask = df0[t_col].isin([baseline, lvl]) & (~df0[t_col].isna())
        dfk = df0.loc[mask].reset_index(drop=True)
        if dfk.empty:
            continue
        tk = (dfk[t_col].astype("object") == lvl).to_numpy(dtype=int)

        if int((tk == 1).sum()) < min_n_per_arm or int((tk == 0).sum()) < min_n_per_arm:
            if strict:
                raise CausalDiagnosticsError(
                    CausalDiagErrorDetails(
                        reason="Insufficient cohort size per arm for a categorical contrast (weights).",
                        evidence={"baseline": baseline, "level": lvl, "n_level": int((tk == 1).sum()), "n_baseline": int((tk == 0).sum())},
                    )
                )
            continue

        Wk = _build_covariate_frame(dfk, covariates)
        e = _fit_propensity_binary(Wk, tk)
        w = _stabilized_ipw(tk, e)

        fig = _plot_weight_stability(
            w, tk,
            title=f"Weight stability (stabilized IPW): {t_col} ({baseline} vs {lvl})",
        )
        out.append(
            CausalGraphImage(
                key=f"weight_stability_{baseline}_vs_{lvl}",
                title=f"Weight stability (IPW) ({baseline} vs {lvl})",
                mime="image/png",
                content=_fig_to_png_bytes(fig, dpi=dpi),
            )
        )

    if strict and not out:
        raise CausalDiagnosticsError(
            CausalDiagErrorDetails(
                reason="No categorical contrasts could be plotted (weights).",
                evidence={"levels": levels},
            )
        )
    return out