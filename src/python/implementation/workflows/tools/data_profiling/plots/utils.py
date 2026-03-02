from __future__ import annotations

import io
import math
from typing import List, Optional, Tuple

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