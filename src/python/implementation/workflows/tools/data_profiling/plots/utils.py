from __future__ import annotations

import io
import math
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def fig_to_png_bytes(fig: Figure, *, dpi: int = 160) -> bytes:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight") # pyright: ignore[reportUnknownMemberType]
        return buf.getvalue()
    finally:
        plt.close(fig)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def fmt_k(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n/1000:.1f}k"
    return f"{int(round(n/1000)):d}k"


def safe_nunique(s: pd.Series) -> Optional[int]:
    try:
        return int(s.nunique(dropna=True))
    except Exception:
        return None


def safe_missing_rate(s: pd.Series) -> float:
    try:
        n = int(s.isna().sum())
        d = len(s)
        return float(n / d) if d > 0 else 0.0
    except Exception:
        return 0.0


def coerce_numeric_ratio(s: pd.Series, *, sample_size: int = 2000) -> float:
    x = s.dropna()
    if x.empty:
        return 0.0
    if len(x) > sample_size:
        x = x.sample(n=sample_size, random_state=0)
    num = pd.to_numeric(x, errors="coerce")
    return float(num.notna().mean())


def select_numeric_columns(
    df: pd.DataFrame,
    *,
    max_cols: int,
    min_non_missing_rate: float = 0.7,
    numeric_like_threshold: float = 0.9,
) -> List[str]:
    """
    Picks numeric-ish columns in a clinical-friendly way:
      - true numeric dtypes OR mostly numeric-like strings
      - filters out highly missing columns
      - filters out constant columns
      - ranks by low missingness, then high variance
    """
    scores: List[Tuple[str, float, float]] = []  # (col, missing_rate, variance)

    for c in df.columns:
        s = df[c]
        miss = safe_missing_rate(s)
        if (1.0 - miss) < min_non_missing_rate:
            continue

        is_numericish = pd.api.types.is_numeric_dtype(s) or (coerce_numeric_ratio(s) >= numeric_like_threshold)
        if not is_numericish:
            continue

        x = pd.to_numeric(s, errors="coerce")
        if not x.notna().any():
            continue

        var = float(x.var(skipna=True).item()) # pyright: ignore[reportUnknownMemberType, reportArgumentType, reportAttributeAccessIssue]
        if not math.isfinite(var) or var <= 0:
            continue

        scores.append((str(c), miss, var))

    scores.sort(key=lambda t: (t[1], -t[2]))
    return [c for c, _, _ in scores[:max_cols]]


# -----------------------------------------------------------------------------
# Protocol role extraction (T / Y / W / X)
# -----------------------------------------------------------------------------

def protocol_treatment_info(protocol: Any) -> Tuple[str, str, Optional[str], Optional[str], Optional[List[str]]]:
    """
    Returns:
      (kind, treatment_col, treated, control, levels)

    - kind: "binary" or "categorical"
    - treated/control are set for binary only
    - levels are set for categorical only
    """
    t_spec = getattr(protocol, "treatment_spec", None)
    if t_spec is None:
        raise ValueError("protocol.treatment_spec is missing")

    kind = str(getattr(t_spec, "kind"))
    col = str(getattr(t_spec, "column"))

    if kind == "binary":
        treated = str(getattr(t_spec, "treated"))
        control = str(getattr(t_spec, "control"))
        return kind, col, treated, control, None

    if kind == "categorical":
        levels_raw = getattr(t_spec, "levels", None)
        if not levels_raw or len(levels_raw) < 2:
            raise ValueError("categorical treatment requires >=2 levels")
        levels = [str(x) for x in levels_raw]
        return kind, col, None, None, levels

    raise ValueError(f"Unknown treatment kind: {kind}")


def protocol_outcome_column(protocol: Any) -> str:
    y_spec = getattr(protocol, "outcome_spec", None)
    if y_spec is None:
        raise ValueError("protocol.outcome_spec is missing")
    return str(getattr(y_spec, "column"))


def protocol_covariates(protocol: Any) -> List[str]:
    return [str(x) for x in (getattr(protocol, "covariates", None) or [])] # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


def protocol_effect_modifiers(protocol: Any) -> List[str]:
    return [str(x) for x in (getattr(protocol, "effect_modifiers", None) or [])] # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


def protocol_WX_columns(protocol: Any, *, include_effect_modifiers: bool = True) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (W_cols, X_cols, feature_cols_for_scoring).
    For scoring we usually want W + (optionally) X.
    """
    W = protocol_covariates(protocol)
    X = protocol_effect_modifiers(protocol)
    feats = list(W)
    if include_effect_modifiers:
        for c in X:
            if c not in feats:
                feats.append(c)
    return W, X, feats


# -----------------------------------------------------------------------------
# Treatment vector creation (binary) from protocol
# -----------------------------------------------------------------------------

def build_binary_treatment_from_protocol(
    df: pd.DataFrame,
    protocol: Any,
    *,
    categorical_contrast: Optional[str] = None,
    categorical_strategy: Literal["most_common", "first_available"] = "most_common",
) -> Tuple[pd.DataFrame, np.ndarray, str, int]:
    """
    Converts treatment into a binary indicator for the comparability map.

    Returns:
      (df_kept, t_bin, label, dropped_rows)

    - For binary treatment: uses treated/control exactly.
    - For categorical treatment: baseline = levels[0], compare baseline vs one other level.
      Default other level is the most common non-baseline level present in data.
    """
    kind, t_col, treated, control, levels = protocol_treatment_info(protocol)
    if t_col not in df.columns:
        raise ValueError(f"Treatment column '{t_col}' missing from df")

    s = df[t_col].astype("object")

    if kind == "binary":
        assert treated is not None and control is not None
        keep = (~s.isna()) & (s.isin([treated, control]))
        dropped = int((~keep).sum())
        d = df.loc[keep].copy()
        t_bin = (d[t_col].astype("object") == treated).to_numpy(dtype=int)
        label = f"{control} vs {treated}"
        return d, t_bin, label, dropped

    # categorical
    assert levels is not None and len(levels) >= 2
    baseline = str(levels[0])
    candidates = [str(x) for x in levels[1:]]

    if categorical_contrast is not None:
        other = str(categorical_contrast)
        if other not in candidates:
            raise ValueError(f"categorical_contrast '{other}' must be one of {candidates}")
    else:
        valid = s[(~s.isna()) & (s.isin([baseline, *candidates]))]
        if valid.empty:
            raise ValueError("No rows match categorical treatment levels in data.")

        vc = valid.value_counts()
        if categorical_strategy == "most_common":
            # choose most common non-baseline
            other = None
            for lvl in vc.index.astype(str).tolist():
                if lvl != baseline and lvl in candidates:
                    other = lvl
                    break
            if other is None:
                other = candidates[0]
        else:
            # first available candidate that exists in data, else candidates[0]
            other = candidates[0]
            for lvl in candidates:
                if (valid.astype(str) == lvl).any():
                    other = lvl
                    break

    keep = (~s.isna()) & (s.isin([baseline, other]))
    dropped = int((~keep).sum())
    d = df.loc[keep].copy()
    t_bin = (d[t_col].astype("object") == other).to_numpy(dtype=int)
    label = f"{baseline} vs {other}"
    return d, t_bin, label, dropped


# -----------------------------------------------------------------------------
# Shared scoring: "likelihood of receiving treatment given baseline profile"
# -----------------------------------------------------------------------------

def _split_numeric_categorical_for_score(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    numeric_like_threshold: float,
) -> Tuple[List[str], List[str]]:
    num: List[str] = []
    cat: List[str] = []
    for c in cols:
        s = df[c]
        is_num = pd.api.types.is_numeric_dtype(s) or (coerce_numeric_ratio(s) >= numeric_like_threshold)
        if is_num:
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def fit_treatment_likelihood_scores(
    df: pd.DataFrame,
    t: np.ndarray,
    feature_cols: Sequence[str],
    *,
    add_missing_indicators: bool = True,
    numeric_like_threshold: float = 0.9,
    clip_eps: float = 1e-3,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 0,
) -> np.ndarray:
    """
    Returns one score per row:
      "likelihood of receiving the treatment given baseline profile"

    Notes:
      - deterministic
      - handles numeric stored as text
      - includes missingness indicators by default (important in clinical data)
    """
    X = df.loc[:, list(feature_cols)].copy()

    if add_missing_indicators:
        for c in feature_cols:
            X[f"{c}__missing"] = df[c].isna().astype(int)

    base_cols = list(feature_cols)
    num_cols, cat_cols = _split_numeric_categorical_for_score(df, base_cols, numeric_like_threshold=numeric_like_threshold)
    if add_missing_indicators:
        num_cols = num_cols + [f"{c}__missing" for c in feature_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", ohe),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X, t) # pyright: ignore[reportUnknownMemberType]

    scores = pipe.predict_proba(X)[:, 1].astype(float) # pyright: ignore[reportUnknownMemberType]
    scores = np.clip(scores, clip_eps, 1.0 - clip_eps)
    return scores