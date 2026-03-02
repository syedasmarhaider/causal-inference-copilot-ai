from __future__ import annotations

import json
import math
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from python.domain.workflows.tool import Tool

import io
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from python.domain.workflows.tool import Tool

ImageMime = Literal["image/png", "image/jpeg", "image/webp"]

InferredKind = Literal["NUMERIC", "DATETIME", "BOOLEAN", "CATEGORICAL", "OTHER"]

# =============================================================================
# Errors (structured + parseable)
# =============================================================================


class ColumnProfileErrorDetailsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: Optional[str] = None
    reason: str
    hint: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None


class DatasetProfilingError(RuntimeError):
    def __init__(self, details: ColumnProfileErrorDetailsModel):
        self.details = details
        msg = details.reason
        if details.column:
            msg = f"Column '{details.column}': {msg}"
        if details.hint:
            msg = f"{msg} Hint: {details.hint}"
        super().__init__(msg)


# =============================================================================
# Pydantic output schema (discriminated union)
# =============================================================================




class NumericSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None  # {"0.05": 1.2, ...}


class DatetimeSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[str] = None  # isoformat-ish
    max: Optional[str] = None


class BooleanSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: Dict[str, int]  # keys are stringified values


class CategoryCountModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    count: int


class CategoricalSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_categories: List[CategoryCountModel]
    other_count: int


class OtherSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distinct_values_sample: List[str]


class ColumnProfileCommonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    dtype: Optional[str] = None
    n_rows: int
    n_missing: int
    missing_rate: float
    distinct_count: Optional[int] = None
    note: Optional[str] = None  # only used in non-strict mode fallbacks


class NumericColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["NUMERIC"]
    summary: NumericSummaryModel


class DatetimeColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["DATETIME"]
    summary: DatetimeSummaryModel


class BooleanColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["BOOLEAN"]
    summary: BooleanSummaryModel


class CategoricalColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["CATEGORICAL"]
    summary: CategoricalSummaryModel


class OtherColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["OTHER"]
    summary: OtherSummaryModel


ColumnProfileModel = Union[
    NumericColumnProfileModel,
    DatetimeColumnProfileModel,
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    OtherColumnProfileModel,
]

# Discriminator annotation (pydantic v2)
DiscriminatedColumnProfile = Annotated[ColumnProfileModel, Field(discriminator="inferred_kind")]


class DatasetSummaryModel(BaseModel):
    """
    Deterministic order: profiles follow df.columns order.
    """
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    profiles: List[DiscriminatedColumnProfile] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]


# =============================================================================
# Public output contract
# =============================================================================

@dataclass(frozen=True)
class GraphImage:
    key: Literal[
        "missingness_by_column",
        "distinctness_vs_missingness",
        "numeric_correlation_heatmap",
    ]
    title: str
    mime: ImageMime
    content: bytes
# =============================================================================
# Public API (STATE-style profiling tool; no constructor; pure funcs)
# =============================================================================

class DatasetProfilingTool(Tool):
    NAME: ClassVar[str] = "DATA_PROFILING"
    """
    "State tool" API:
      - extract_dataset_summary(df, ...) -> DatasetSummaryModel
      - dataset_summary_to_json(summary, ...) -> strict JSON (no NaN/Inf)
      - dataset_summary_from_json(payload) -> DatasetSummaryModel
    """
    

    def get_tool_name(self) -> str:
        return self.NAME
    
    def get_tool_info(self) -> str:
        return "Tool for profiling datasets for summary statistics and data quality insights."
    
    
    def generate_basic_stats_graphs(
        self,
        df: pd.DataFrame,
        *,
        # graph sizing / caps
        max_columns_missingness: int = 60,
        max_numeric_for_corr: int = 25,
        max_rows_for_corr: int = 50_000,
        corr_method: Literal["pearson", "spearman"] = "pearson",
        dpi: int = 150,
    ) -> List[GraphImage]:
        _validate_df(df)
        _validate_int("max_columns_missingness", max_columns_missingness, min_value=1)
        _validate_int("max_numeric_for_corr", max_numeric_for_corr, min_value=2)
        _validate_int("max_rows_for_corr", max_rows_for_corr, min_value=100)
        _validate_int("dpi", dpi, min_value=72)

        metrics = _compute_column_metrics(df)

        out: List[GraphImage] = []

        # 1) Missingness bar chart
        fig1 = _plot_missingness_bar(metrics, max_columns=max_columns_missingness)
        out.append(
            GraphImage(
                key="missingness_by_column",
                title="Missingness by column",
                mime="image/png",
                content=_fig_to_png_bytes(fig1, dpi=dpi),
            )
        )

        # 2) Distinctness vs missingness scatter
        fig2 = _plot_distinctness_vs_missingness(metrics)
        out.append(
            GraphImage(
                key="distinctness_vs_missingness",
                title="Distinctness vs missingness",
                mime="image/png",
                content=_fig_to_png_bytes(fig2, dpi=dpi),
            )
        )

        # 3) Correlation heatmap for selected numeric columns
        fig3 = _plot_numeric_correlation_heatmap(
            df,
            metrics,
            max_numeric=max_numeric_for_corr,
            max_rows=max_rows_for_corr,
            method=corr_method,
        )
        out.append(
            GraphImage(
                key="numeric_correlation_heatmap",
                title="Numeric correlation heatmap",
                mime="image/png",
                content=_fig_to_png_bytes(fig3, dpi=dpi),
            )
        )

        return out





    def extract_dataset_summary(
        self,
        df: pd.DataFrame,
        *,
        max_categories: int = 50,
        sample_distinct: int = 50,
        compute_quantiles: bool = True,
        strict: bool = True,
    ) -> DatasetSummaryModel:
        _validate_params(max_categories=max_categories, sample_distinct=sample_distinct)

        cols = _get_columns(df, strict=strict)
        n_rows = _safe_n_rows(df, strict=strict)
        dtypes = getattr(df, "dtypes", None)

        profiles: List[DiscriminatedColumnProfile] = []

        for col_key in cols:
            name = str(col_key).strip()
            if not name:
                if strict:
                    raise DatasetProfilingError(
                        ColumnProfileErrorDetailsModel(
                            column=None,
                            reason="Dataset contains an empty column name.",
                            hint="Rename the column to a non-empty string.",
                            evidence={"raw_column": repr(col_key)},
                        )
                    )
                continue

            try:
                s = _get_series(df, col_key, name, strict=strict)
                dtype_str = _dtype_to_str(dtypes, col_key)
                kind = _infer_kind(s)

                n_missing, missing_rate = _missingness(s, n_rows=n_rows)
                distinct = _distinct_count(s)

                base = ColumnProfileCommonModel(
                    name=name,
                    dtype=dtype_str,
                    n_rows=n_rows,
                    n_missing=n_missing,
                    missing_rate=missing_rate,
                    distinct_count=distinct,
                )

                if kind == "NUMERIC":
                    profiles.append(
                        NumericColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="NUMERIC",
                            summary=_numeric_summary(s, compute_quantiles=compute_quantiles),
                        )
                    )
                elif kind == "DATETIME":
                    profiles.append(
                        DatetimeColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="DATETIME",
                            summary=_datetime_summary(s),
                        )
                    )
                elif kind == "BOOLEAN":
                    profiles.append(
                        BooleanColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="BOOLEAN",
                            summary=_boolean_summary(s),
                        )
                    )
                elif kind == "CATEGORICAL":
                    profiles.append(
                        CategoricalColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="CATEGORICAL",
                            summary=_categorical_summary(s, max_categories=max_categories),
                        )
                    )
                else:
                    profiles.append(
                        OtherColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="OTHER",
                            summary=_other_summary(s, sample_distinct=sample_distinct),
                        )
                    )

            except DatasetProfilingError:
                if strict:
                    raise
                # Non-strict fallback: still a VALID profile
                profiles.append(
                    OtherColumnProfileModel(
                        name=name,
                        dtype=_dtype_to_str(dtypes, col_key),
                        n_rows=n_rows,
                        n_missing=0,
                        missing_rate=0.0,
                        distinct_count=None,
                        inferred_kind="OTHER",
                        summary=OtherSummaryModel(distinct_values_sample=[]),
                        note="Profiling failed for this column in non-strict mode.",
                    )
                )

        if strict and not profiles:
            raise DatasetProfilingError(
                ColumnProfileErrorDetailsModel(
                    column=None,
                    reason="No columns could be profiled.",
                    hint="Verify the dataset is tabular and contains at least one named column.",
                )
            )

        return DatasetSummaryModel(n_rows=n_rows, profiles=profiles)

    def dataset_summary_to_json(self,
        summary: DatasetSummaryModel,
        *,
        indent: int | None = None,
    ) -> str:
        return summary.model_dump_json(indent=indent)

    def dataset_summary_from_json(self, payload: str) -> DatasetSummaryModel:
        data = json.loads(payload)
        return DatasetSummaryModel.model_validate(data)


# =============================================================================
# Internals (logic preserved from your reference)
# =============================================================================

def _validate_params(*, max_categories: int, sample_distinct: int) -> None:
    if max_categories <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="max_categories must be > 0.",
                evidence={"max_categories": max_categories},
            )
        )
    if sample_distinct <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="sample_distinct must be > 0.",
                evidence={"sample_distinct": sample_distinct},
            )
        )


def _get_columns(df: Any, *, strict: bool) -> List[Any]:
    raw_cols = getattr(df, "columns", None)
    if raw_cols is None:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Dataset object has no 'columns' attribute; not a DataFrame-like table.",
                hint="Use a pandas DataFrame (recommended) or provide an object exposing .columns and df[col].",
                evidence={"df_type": type(df).__name__},
            )
        )
    try:
        cols = list(raw_cols)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Could not iterate dataset columns.",
                hint="Ensure df.columns is iterable.",
                evidence={"df_type": type(df).__name__, "error": repr(e)},
            )
        )
    if strict and not cols:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Dataset has zero columns.",
                hint="Provide at least one column.",
            )
        )
    return cols


def _safe_n_rows(df: Any, *, strict: bool) -> int:
    try:
        shape = getattr(df, "shape", None)
        if shape is not None and len(shape) >= 1:
            n = int(shape[0])
            if n < 0:
                raise ValueError("negative row count")
            return n
    except Exception as e:
        if strict:
            raise DatasetProfilingError(
                ColumnProfileErrorDetailsModel(
                    column=None,
                    reason="Could not read df.shape[0].",
                    hint="Ensure df.shape is valid (pandas DataFrame recommended).",
                    evidence={"df_type": type(df).__name__, "error": repr(e)},
                )
            )
    try:
        n = int(len(df))
        if n < 0:
            raise ValueError("negative row count")
        return n
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Could not determine dataset row count.",
                hint="Ensure df implements __len__ or provides df.shape.",
                evidence={"df_type": type(df).__name__, "error": repr(e)},
            )
        )


def _get_series(df: Any, col_key: Any, col_name: str, *, strict: bool) -> Any:
    try:
        return df[col_key]
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=col_name,
                reason="Could not access column via df[col].",
                hint="Verify the column exists and df supports __getitem__ (pandas DataFrame recommended).",
                evidence={"col_key": repr(col_key), "df_type": type(df).__name__, "error": repr(e)},
            )
        )


def _dtype_to_str(dtypes: Any, col_key: Any) -> Optional[str]:
    try:
        if dtypes is None:
            return None
        return str(dtypes[col_key])
    except Exception:
        return None


def _missingness(series: Any, *, n_rows: int) -> Tuple[int, float]:
    try:
        if hasattr(series, "isna"):
            n_missing = int(series.isna().sum())
        elif hasattr(series, "isnull"):
            n_missing = int(series.isnull().sum())
        else:
            vals = list(series)
            n_missing = sum(1 for x in vals if x is None)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Could not compute missingness.",
                hint="Ensure the column supports isna()/isnull() or is iterable.",
                evidence={"error": repr(e)},
            )
        )

    rate = (n_missing / n_rows) if n_rows > 0 else 0.0
    return n_missing, float(rate)


def _distinct_count(series: Any) -> Optional[int]:
    try:
        if hasattr(series, "nunique"):
            return int(series.nunique(dropna=True))
    except Exception:
        return None
    return None


def _as_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


def _as_datetime_str(v: Any) -> Optional[str]:
    try:
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            return str(v.isoformat())
        return str(v)
    except Exception:
        return None


def _numeric_summary(series: Any, *, compute_quantiles: bool) -> NumericSummaryModel:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series

        if hasattr(s, "astype"):
            try:
                s = s.astype(float)
            except Exception:
                # Preserve your old behavior: don't crash; emit empty numeric stats.
                return NumericSummaryModel()

        mn = _as_float_or_none(getattr(s, "min", lambda: None)())
        mx = _as_float_or_none(getattr(s, "max", lambda: None)())
        mean = _as_float_or_none(getattr(s, "mean", lambda: None)())
        std = _as_float_or_none(getattr(s, "std", lambda: None)())

        quantiles: Optional[Dict[str, float]] = None
        if compute_quantiles and hasattr(s, "quantile"):
            qs = [0.05, 0.25, 0.5, 0.75, 0.95]
            qvals = s.quantile(qs)
            items = list(qvals.items()) if hasattr(qvals, "items") else []

            out: Dict[str, float] = {}
            for k, v in items:
                fv = _as_float_or_none(v)
                if fv is not None:
                    out[str(k)] = fv
            quantiles = out or None

        return NumericSummaryModel(min=mn, max=mx, mean=mean, std=std, quantiles=quantiles)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Numeric summary failed.",
                hint="Verify the column is numeric or coercible to float.",
                evidence={"error": repr(e)},
            )
        )


def _datetime_summary(series: Any) -> DatetimeSummaryModel:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series
        mn = getattr(s, "min", lambda: None)()
        mx = getattr(s, "max", lambda: None)()
        return DatetimeSummaryModel(min=_as_datetime_str(mn), max=_as_datetime_str(mx))
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Datetime summary failed.",
                hint="Verify the column is datetime-like.",
                evidence={"error": repr(e)},
            )
        )


def _boolean_summary(series: Any) -> BooleanSummaryModel:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = list(vc.items()) if hasattr(vc, "items") else []
            counts: Dict[str, int] = {}
            for k, v in items:
                counts[str(k)] = int(v)
            return BooleanSummaryModel(counts=counts)

        vals = list(series)
        counts = {
            "True": sum(1 for x in vals if x is True),
            "False": sum(1 for x in vals if x is False),
        }
        return BooleanSummaryModel(counts=counts)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Boolean summary failed.",
                hint="Verify the column is boolean-like or has value_counts().",
                evidence={"error": repr(e)},
            )
        )


def _categorical_summary(series: Any, *, max_categories: int) -> CategoricalSummaryModel:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = list(vc.items()) if hasattr(vc, "items") else []
        else:
            vals = [x for x in list(series) if x is not None]
            freq: Dict[str, int] = {}
            for x in vals:
                sx = str(x)
                freq[sx] = freq.get(sx, 0) + 1
            items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)

        top_items = items[:max_categories]
        other_count = sum(int(v) for _, v in items[max_categories:]) if len(items) > max_categories else 0

        top = [CategoryCountModel(value=str(k), count=int(v)) for k, v in top_items]
        return CategoricalSummaryModel(top_categories=top, other_count=int(other_count))
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Categorical summary failed.",
                hint="Verify the column is categorical/string-like or iterable.",
                evidence={"error": repr(e)},
            )
        )


def _other_summary(series: Any, *, sample_distinct: int) -> OtherSummaryModel:
    try:
        s = series.dropna() if hasattr(series, "dropna") else [x for x in list(series) if x is not None]

        if hasattr(s, "unique"):
            uniq: List[Any] = list(s.unique()) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
            distinct = [str(x) for x in uniq[:sample_distinct]]
        else:
            seen: List[str] = []
            for x in list(s):
                sx = str(x)
                if sx not in seen:
                    seen.append(sx)
                if len(seen) >= sample_distinct:
                    break
            distinct = seen

        return OtherSummaryModel(distinct_values_sample=distinct)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Other-type summary failed.",
                hint="Verify the column is iterable or provides .unique().",
                evidence={"error": repr(e)},
            )
        )   




# =============================================================================
# Internals
# =============================================================================

@dataclass(frozen=True)
class _ColMetric:
    name: str
    dtype_str: str
    inferred_kind: InferredKind
    n_rows: int
    n_missing: int
    missing_rate: float
    distinct_count: Optional[int]
    variance: Optional[float]  # only for numeric-ish columns


def _validate_df(df: Any) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if df.shape[1] == 0:
        raise ValueError("df has zero columns")
    # note: zero rows is allowed; graphs will be mostly empty but valid


def _validate_int(name: str, v: int, *, min_value: int) -> None:
    if not isinstance(v, int):
        raise TypeError(f"{name} must be int, got {type(v).__name__}")
    if v < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {v}")


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


def _compute_column_metrics(df: pd.DataFrame) -> List[_ColMetric]:
    n_rows = int(df.shape[0])
    dtypes = df.dtypes

    metrics: List[_ColMetric] = []
    for col in df.columns:
        name = str(col).strip()
        if not name:
            name = str(col)  # keep something stable

        dtype_str = str(dtypes[col])
        s = df[col]
        kind = _infer_kind(s)

        s = df[col]
        # missingness
        try:
            n_missing = int(s.isna().sum())
        except Exception:
            # very defensive fallback
            vals = list(s)
            n_missing = sum(1 for x in vals if x is None)
        missing_rate = float(n_missing / n_rows) if n_rows > 0 else 0.0

        # distinct count (dropna)
        distinct_count: Optional[int]
        try:
            distinct_count = int(s.nunique(dropna=True))
        except Exception:
            distinct_count = None

        # variance (only for numeric columns; used for correlation selection ranking)
        variance: Optional[float] = None
        if kind == "NUMERIC":
            try:
                variance = _safe_float(pd.to_numeric(s, errors="coerce").var(skipna=True))
            except Exception:
                variance = None

        metrics.append(
            _ColMetric(
                name=name,
                dtype_str=dtype_str,
                inferred_kind=kind,
                n_rows=n_rows,
                n_missing=n_missing,
                missing_rate=missing_rate,
                distinct_count=distinct_count,
                variance=variance,
            )
        )
    return metrics


def _fig_to_png_bytes(fig: Figure, *, dpi: int) -> bytes:
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Plot 1: Missingness bar chart
# -----------------------------------------------------------------------------

def _plot_missingness_bar(metrics: Sequence[_ColMetric], *, max_columns: int) -> Figure:
    # sort by missing desc; show top K
    ordered = sorted(metrics, key=lambda m: m.missing_rate, reverse=True)
    shown = ordered[:max_columns]

    names = [m.name for m in shown]
    rates = [m.missing_rate for m in shown]

    fig, ax = plt.subplots(figsize=(max(8, min(16, 0.25 * len(shown) + 6)), 6))
    ax.bar(range(len(shown)), rates)
    ax.set_title("Missingness by column (top columns)")
    ax.set_ylabel("Missing rate")
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)

    if len(metrics) > max_columns:
        ax.text(
            0.0,
            -0.18,
            f"Showing top {max_columns} by missingness out of {len(metrics)} columns.",
            transform=ax.transAxes,
            fontsize=9,
        )

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Plot 2: Distinctness vs Missingness scatter (log-x)
# -----------------------------------------------------------------------------

_KIND_MARKERS: Dict[InferredKind, str] = {
    "NUMERIC": "o",
    "DATETIME": "s",
    "BOOLEAN": "^",
    "CATEGORICAL": "D",
    "OTHER": "x",
}

def _plot_distinctness_vs_missingness(metrics: Sequence[_ColMetric]) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Distinctness vs missingness (log distinct count)")
    ax.set_xlabel("Distinct count (log scale)")
    ax.set_ylabel("Missing rate")
    ax.set_ylim(0.0, 1.0)

    # group by kind; let matplotlib auto-assign colors via cycle
    for kind in ["NUMERIC", "DATETIME", "BOOLEAN", "CATEGORICAL", "OTHER"]:
        pts = [m for m in metrics if m.inferred_kind == kind]
        if not pts:
            continue

        xs: List[float] = []
        ys: List[float] = []
        for m in pts:
            # if distinct_count missing, skip point
            if m.distinct_count is None:
                continue
            # avoid log(0)
            xs.append(float(max(1, m.distinct_count)))
            ys.append(float(m.missing_rate))

        if xs:
            ax.scatter(xs, ys, marker=_KIND_MARKERS[kind], alpha=0.75, label=kind)

    ax.set_xscale("log")
    ax.legend(title="Inferred kind", loc="best", fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Plot 3: Numeric correlation heatmap (selected columns)
# -----------------------------------------------------------------------------

def _select_numeric_for_corr(
    df: pd.DataFrame,
    metrics: Sequence[_ColMetric],
    *,
    max_numeric: int,
) -> List[str]:
    # select NUMERIC columns; rank by low missingness then high variance (fallback to 0)
    numeric = [m for m in metrics if m.inferred_kind == "NUMERIC"]
    if not numeric:
        return []

    def key(m: _ColMetric) -> Tuple[float, float]:
        var = m.variance if m.variance is not None else 0.0
        return (m.missing_rate, -var)

    ordered = sorted(numeric, key=key)
    cols = [m.name for m in ordered]

    # ensure these names exist as df columns (handles whitespace normalization mismatch)
    df_cols = set(map(str, df.columns))
    present = [c for c in cols if c in df_cols]
    return present[:max_numeric]


def _plot_numeric_correlation_heatmap(
    df: pd.DataFrame,
    metrics: Sequence[_ColMetric],
    *,
    max_numeric: int,
    max_rows: int,
    method: Literal["pearson", "spearman"],
) -> Figure:
    cols = _select_numeric_for_corr(df, metrics, max_numeric=max_numeric)

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(cols) < 2:
        ax.axis("off")
        ax.set_title("Numeric correlation heatmap")
        ax.text(
            0.5,
            0.5,
            "Not enough numeric columns to compute correlations.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    work = df[cols]
    # cap rows for speed on huge datasets
    if len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=0)

    # coerce to numeric and compute correlation
    work_num = work.apply(pd.to_numeric, errors="coerce")
    corr = work_num.corr(method=method, min_periods=10)

    # fill NaNs for display (e.g., constant columns); keep it stable
    corr_disp = corr.fillna(0.0).to_numpy(dtype=float)

    im = ax.imshow(corr_disp, vmin=-1.0, vmax=1.0)
    ax.set_title(f"Numeric correlation heatmap ({method}), n={len(cols)} cols")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig        





def _infer_kind(
    series: pd.Series,
    *,
    numeric_coerce_threshold: float = 0.90,
    datetime_coerce_threshold: float = 0.90,
    sample_size: int = 2000,
) -> InferredKind:
    # 1) Fast path: real dtypes
    if is_datetime64_any_dtype(series):
        return "DATETIME"
    if is_bool_dtype(series):
        return "BOOLEAN"
    if is_numeric_dtype(series):
        return "NUMERIC"

    # 2) Sample non-missing for heuristics (deterministic)
    s = series.dropna()
    if s.empty:
        return "OTHER"

    if len(s) > sample_size:
        s = s.sample(n=sample_size, random_state=0)

    # 3) Detect complex objects -> OTHER
    if s.map(lambda x: isinstance(x, (dict, list, set, tuple))).any():
        return "OTHER"

    # 4) Boolean-like strings
    low = s.astype(str).str.strip().str.lower()
    bool_vocab = {"true", "false", "0", "1", "yes", "no", "y", "n"}
    if float(low.isin(bool_vocab).mean()) >= 0.95:
        return "BOOLEAN"

    # 5) Numeric-like strings
    num = pd.to_numeric(s, errors="coerce")
    if float(num.notna().mean()) >= numeric_coerce_threshold:
        return "NUMERIC"

    # 6) Datetime-like strings
    dt = pd.to_datetime(s, errors="coerce")
    if float(dt.notna().mean()) >= datetime_coerce_threshold:
        return "DATETIME"

    return "CATEGORICAL"