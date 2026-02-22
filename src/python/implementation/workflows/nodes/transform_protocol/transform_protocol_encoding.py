from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from python.implementation.workflows.nodes.transform_protocol.transform_protcol_plan import (
    BinaryMapAsUnknown,
    BinaryMapErrorIfNA,
    BinaryMapIdxParams,
    BinaryMapImputeConstant,
    BinaryMapImputeToken,
    BinaryMapParams,
    DateTimeToEpochParams,
    IdxAsUnknown,
    IdxErrorIfNA,
    IdxImputeIndex,
    IdxImputeMode,
    Log1pParams,
    MinMaxParams,
    NumericAddMissingIndicator,
    NumericErrorIfNA,
    NumericImputeMean,
    NumericImputeMedian,
    NumericKeepNA,
    OneHotDummyNA,
    OneHotImputeMode,
    OneHotImputeToken,
    OneHotParams,
    OrdinalAsUnknown,
    OrdinalErrorIfNA,
    OrdinalImputeMode,
    OrdinalImputeToken,
    OrdinalMapIdxParams,
    OrdinalMapParams,
    StandardizeParams,
    ToNumericParams,
)
from python.implementation.workflows.utils.validation import (
    ValidationIssueModel,
    ValidationSeverity,
)


@dataclass(frozen=True)
class TransformPlanApplicationError(RuntimeError):
    message: str
    evidence: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.message} | evidence={self.evidence}"


def _issue(
    message: str,
    severity: ValidationSeverity,
    evidence: dict[str, Any] | None = None,
    fix_hint: str | None = None,
) -> ValidationIssueModel:
    return ValidationIssueModel(
        severity=severity,
        message=message,
        evidence=evidence or {},
        fix_hint=fix_hint,
    )


def _strip_only(x: Any) -> Any:
    if isinstance(x, str):
        return x.strip()
    return x


def apply_one_hot_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: OneHotParams,
    prefix_sep: str = "__",
) -> ValidationIssueModel | None:
    """
    Applies one-hot encoding IN-PLACE to a single column, per OneHotParams.

    Returns:
      - None                         => applied successfully, no warnings
      - ValidationIssueModel("WARN") => applied successfully, but warning emitted
      - ValidationIssueModel("FAIL") => not applied, df not mutated   (DATA issue)

    Raises:
      - TransformPlanApplicationError => PLAN/LLM issue (invalid plan / incompatible spec)
    """

    # -------------------------
    # PLAN/LLM validations
    # -------------------------
    if column not in df_to_change.columns:
        raise TransformPlanApplicationError(
            f"one_hot: plan references unknown column {column!r}.",
            evidence={
                "column": column,
                "n_columns": int(len(df_to_change.columns)),
                "columns_sample": list(df_to_change.columns[:50]),
            },
        )

    loc = df_to_change.columns.get_loc(column)
    if not isinstance(loc, int):
        raise TransformPlanApplicationError(
            f"one_hot: column name {column!r} is not unique; cannot apply deterministically.",
            evidence={"column": column, "get_loc_type": str(type(loc))},
        )

    miss = params.missingness
    if not isinstance(miss, (OneHotImputeToken, OneHotImputeMode, OneHotDummyNA)):
        # We explicitly do NOT support dropping rows here.
        raise TransformPlanApplicationError(
            "one_hot: unsupported missingness strategy for one_hot (row dropping is disallowed in this stage).",
            evidence={
                "column": column,
                "missingness_type": miss.__class__.__name__,
                "allowed": ["OneHotDummyNA", "OneHotImputeToken", "OneHotImputeMode"],
            },
        )

    # -------------------------
    # DATA-driven failures/warnings (no mutation yet)
    # -------------------------
    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column]
    s = s0.map(_strip_only)

    n_missing = int(pd.isna(s).sum())

    if isinstance(miss, OneHotImputeToken):
        s = s.fillna(miss.token) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, OneHotImputeMode):
        non_na = s.dropna()
        if non_na.empty:
            return _issue(
                f"one_hot: impute_mode cannot run because {column!r} has no non-missing values.",
                severity="FAIL",
                evidence={"column": column, "n_rows": int(len(s0)), "n_missing": n_missing},
                fix_hint="Use dummy_na or impute_token.",
            )
        vc = non_na.value_counts(dropna=True)
        max_count = int(vc.max())
        top = vc[vc == max_count].index.tolist()
        mode_value = sorted(top, key=lambda x: str(x))[0]
        s = s.fillna(mode_value) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, OneHotDummyNA): # pyright: ignore[reportUnnecessaryIsInstance]
        pass  # dummy_na=True will represent NA explicitly

    # max_categories (DATA issue)
    if params.max_categories is not None:
        n_unique = int(s.dropna().nunique())
        if n_unique > int(params.max_categories):
            return _issue(
                f"one_hot: too many categories in {column!r} (n_unique={n_unique}, max={int(params.max_categories)}).",
                severity="FAIL",
                evidence={"column": column, "n_unique_non_na": n_unique, "max_categories": int(params.max_categories)},
                fix_hint="Increase max_categories or choose ordinal_map/binary_map.",
            )

    # Encode (still no mutation)
    dummy_na_flag = isinstance(miss, OneHotDummyNA)

    dummies = pd.get_dummies(
        s,
        prefix=column,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na_flag,
        drop_first=False,
    )

    if dummies.shape[1] == 0:
        return _issue(
            f"one_hot: no dummy columns produced for {column!r}.",
            severity="FAIL",
            evidence={"column": column, "n_unique_non_na": int(s.dropna().nunique()), "n_missing": n_missing},
            fix_hint="Column may be all-missing or constant; adjust missingness/encoding.",
        )

    dummies = dummies.astype("int8")

    # drop_first warning
    if params.drop_first and dummies.shape[1] <= 1:
        return_issue = _issue(
            f"one_hot: cannot drop first dummy for {column!r}; only {int(dummies.shape[1])} dummy column(s) produced.",
            severity="WARN",
            evidence={"column": column, "n_dummy_cols": int(dummies.shape[1])},
            fix_hint="Set drop_first=False or verify the column has >1 category.",
        )

    # drop_first deterministic drop (if possible)
    if params.drop_first and dummies.shape[1] > 1:
        cols_sorted = sorted(dummies.columns)
        to_drop = cols_sorted[0]

        if dummy_na_flag:
            def is_na_dummy(colname: str) -> bool:
                if not colname.startswith(f"{column}{prefix_sep}"):
                    return False
                cat = colname.split(prefix_sep, 1)[1].lower()
                return cat in {"nan", "<na>"}

            non_na = [c for c in cols_sorted if not is_na_dummy(c)]
            to_drop = non_na[0] if non_na else cols_sorted[0]

        dummies = dummies.drop(columns=[to_drop])

    # -------------------------
    # COMMIT (mutate only now)
    # -------------------------
    dummies = dummies.reindex(df_to_change.index)

    df_to_change.drop(columns=[column], inplace=True)
    for i, new_col in enumerate(dummies.columns):
        df_to_change.insert(loc + i, new_col, dummies[new_col]) # pyright: ignore[reportUnknownMemberType]

    return return_issue


def _ensure_column_exists_unique(df: pd.DataFrame, *, column: str, encoder: str) -> int:
    if column not in df.columns:
        raise TransformPlanApplicationError(
            f"{encoder}: plan references unknown column {column!r}.",
            evidence={
                "column": column,
                "n_columns": int(len(df.columns)),
                "columns_sample": list(df.columns[:50]),
            },
        )
    loc = df.columns.get_loc(column)
    if not isinstance(loc, int):
        raise TransformPlanApplicationError(
            f"{encoder}: column name {column!r} is not unique; cannot apply deterministically.",
            evidence={"column": column, "get_loc_type": str(type(loc))},
        )
    return loc


def _apply_numeric_output_missingness(
    *,
    out: pd.Series,
    output_missingness: Any,
    base_column: str,
) -> tuple[pd.Series, Optional[pd.Series], Optional[str], Optional[ValidationIssueModel]]:
    """
    Applies NumericMissingnessSpec to an output numeric series.
    Returns: (out2, indicator_series_or_none, indicator_colname_or_none, warn_issue_or_none)
    PLAN faults raise. DATA faults return FAIL issue (caller should propagate).
    """
    warn_issue: Optional[ValidationIssueModel] = None
    indicator: Optional[pd.Series] = None
    indicator_name: Optional[str] = None

    if isinstance(output_missingness, NumericKeepNA):
        return out, None, None, None

    if isinstance(output_missingness, NumericAddMissingIndicator):
        suffix = output_missingness.suffix
        indicator_name = f"{base_column}{suffix}"
        indicator = out.isna().astype("int8")
        return out, indicator, indicator_name, None

    if isinstance(output_missingness, NumericErrorIfNA):
        n_na = int(out.isna().sum())
        if n_na > 0:
            return (
                out,
                None,
                None,
                _issue(
                    f"numeric_output_missingness: error_if_na but output contains missing values for {base_column!r}.",
                    severity="FAIL",
                    evidence={"column": base_column, "n_missing": n_na},
                    fix_hint="Use keep_na / add_missing_indicator / impute_mean / impute_median.",
                ),
            )
        return out, None, None, None

    if isinstance(output_missingness, (NumericImputeMean, NumericImputeMedian)):
        non_na = out.dropna()
        if non_na.empty:
            return (
                out,
                None,
                None,
                _issue(
                    f"numeric_output_missingness: cannot impute because output is all-missing for {base_column!r}.",
                    severity="FAIL",
                    evidence={"column": base_column, "n_rows": int(len(out))},
                    fix_hint="Use add_missing_indicator or keep_na, or fix upstream mapping/unknown_value.",
                ),
            )

        fill_value = float(non_na.mean()) if isinstance(output_missingness, NumericImputeMean) else float(non_na.median())
        out2 = out.fillna(fill_value) # pyright: ignore[reportUnknownMemberType]

        # Optional WARN: imputation occurred
        if int(out.isna().sum()) > 0:
            warn_issue = _issue(
                f"numeric_output_missingness: imputed missing values in {base_column!r}.",
                severity="WARN",
                evidence={"column": base_column, "fill_value": fill_value, "n_imputed": int(out.isna().sum())},
            )

        return out2, None, None, warn_issue

    raise TransformPlanApplicationError(
        "numeric_output_missingness: unsupported output_missingness type (plan validation failure).",
        evidence={"column": base_column, "type": output_missingness.__class__.__name__},
    )


def apply_binary_map_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: BinaryMapParams,
) -> ValidationIssueModel | None:
    """
    Applies BinaryMapParams IN-PLACE to a single column.

    DATA issues => return ValidationIssueModel("FAIL") and do not mutate.
    PLAN issues => raise TransformPlanApplicationError.
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="binary_map")

    # PLAN checks (fast)
    if not params.mapping:
        raise TransformPlanApplicationError("binary_map: mapping must be non-empty.", evidence={"column": column})

    miss = params.missingness
    out_miss = params.output_missingness

    if isinstance(miss, BinaryMapAsUnknown) and not params.allow_unknown:
        raise TransformPlanApplicationError(
            "binary_map: missingness='as_unknown' requires allow_unknown=True.",
            evidence={"column": column},
        )

    if isinstance(miss, BinaryMapImputeToken):
        if miss.token not in params.mapping:
            raise TransformPlanApplicationError(
                "binary_map: impute_token token must exist in mapping (LLM plan bug).",
                evidence={"column": column, "token": miss.token, "mapping_keys_sample": list(params.mapping.keys())[:25]},
            )

    # Stage 1: strip-only
    s0 = df_to_change[column]
    s = s0.map(_strip_only)

    is_na = pd.isna(s)
    n_missing = int(is_na.sum())

    # Stage 2: input missingness handling (NO row drops)
    missing_output_override: Optional[float] = None

    if isinstance(miss, BinaryMapErrorIfNA):
        if n_missing > 0:
            return _issue(
                f"binary_map: missingness='error_if_na' but {column!r} contains missing values.",
                severity="FAIL",
                evidence={"column": column, "n_missing": n_missing},
                fix_hint="Use impute_token / impute_constant / as_unknown (with allow_unknown=True).",
            )

    elif isinstance(miss, BinaryMapImputeToken):
        s = s.fillna(miss.token) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, BinaryMapImputeConstant):
        missing_output_override = float(miss.value)

    elif isinstance(miss, BinaryMapAsUnknown): # pyright: ignore[reportUnnecessaryIsInstance]
        # handled at output stage (NA treated as unknown)
        pass

    else:
        raise TransformPlanApplicationError(
            "binary_map: unsupported missingness spec type (plan validation failure).",
            evidence={"column": column, "missingness_type": miss.__class__.__name__},
        )

    # Stage 3: map values → numeric
    out = pd.Series(index=s.index, dtype="float64")

    # Map known categories
    # Only map strings; anything else is "unknown category"
    def map_one(v: Any) -> tuple[bool, float]:
        # returns (known, value)
        if isinstance(v, str) and v in params.mapping:
            return True, float(params.mapping[v])
        return False, float("nan")

    known_mask = pd.Series(False, index=s.index)
    unknown_mask = pd.Series(False, index=s.index)

    for idx, v in s.items():
        if pd.isna(v):
            continue
        known, val = map_one(v)
        if known:
            out.at[idx] = val
            known_mask.at[idx] = True
        else:
            unknown_mask.at[idx] = True

    # Handle unknown categories
    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0 and not params.allow_unknown:
        sample = s.loc[unknown_mask].dropna().astype(object).head(25).tolist()
        return _issue(
            f"binary_map: encountered unknown categories in {column!r} (allow_unknown=False).",
            severity="FAIL",
            evidence={"column": column, "n_unknown": n_unknown, "unknown_sample": sample},
            fix_hint="Set allow_unknown=True (and optionally unknown_value) or extend mapping.",
        )

    unknown_value = float(params.unknown_value) if params.unknown_value is not None else float("nan")
    if n_unknown > 0:
        out.loc[unknown_mask] = unknown_value

    # Handle missing (NA) after missingness strategy
    if missing_output_override is not None:
        out.loc[pd.isna(s)] = missing_output_override
    elif isinstance(miss, BinaryMapAsUnknown):
        # treat NA as unknown
        out.loc[pd.isna(s)] = unknown_value
    else:
        # leave as NaN
        pass

    # Stage 4: output missingness (may add indicator, may impute, may fail)
    out2, ind, ind_name, out_miss_issue = _apply_numeric_output_missingness(
        out=out,
        output_missingness=out_miss,
        base_column=column,
    )
    if out_miss_issue is not None and out_miss_issue.severity == "FAIL":
        return out_miss_issue
    warn_issue = out_miss_issue if out_miss_issue is not None and out_miss_issue.severity == "WARN" else None

    # PLAN: indicator column name collision check (before commit)
    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "binary_map: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # -------------------------
    # COMMIT (mutate only now)
    # -------------------------
    df_to_change.drop(columns=[column], inplace=True)

    df_to_change.insert(loc, column, out2.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return warn_issue


def apply_binary_map_idx_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: BinaryMapIdxParams,
) -> ValidationIssueModel | None:
    """
    Applies BinaryMapIdxParams IN-PLACE to a single column (expects category indices).

    DATA issues => return ValidationIssueModel("FAIL") and do not mutate.
    PLAN issues => raise TransformPlanApplicationError.
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="binary_map_idx")

    # PLAN checks: disjoint sets (defensive)
    pos, neg, drp = set(params.pos), set(params.neg), set(params.drop)
    inter = (pos & neg) | (pos & drp) | (neg & drp)
    if inter:
        raise TransformPlanApplicationError(
            "binary_map_idx: pos/neg/drop must be disjoint.",
            evidence={"column": column, "overlap": sorted(inter)},
        )

    miss = params.missingness
    out_miss = params.output_missingness

    # Missingness requires allow_unknown if as_unknown
    if isinstance(miss, IdxAsUnknown) and not params.allow_unknown:
        raise TransformPlanApplicationError(
            "binary_map_idx: missingness='as_unknown' requires allow_unknown=True.",
            evidence={"column": column},
        )

    if isinstance(miss, IdxImputeIndex) and miss.index in drp:
        raise TransformPlanApplicationError(
            "binary_map_idx: impute_index must not be in drop list.",
            evidence={"column": column, "impute_index": int(miss.index)},
        )

    # Stage 1: strip-only for strings (mostly irrelevant for idx columns, but harmless)
    s0 = df_to_change[column]
    s = s0.map(_strip_only)

    is_na = pd.isna(s)
    n_missing = int(is_na.sum())

    # Stage 2: missingness on indices (NO row drops)
    if isinstance(miss, IdxErrorIfNA):
        if n_missing > 0:
            return _issue(
                f"binary_map_idx: missingness='error_if_na' but {column!r} contains missing values.",
                severity="FAIL",
                evidence={"column": column, "n_missing": n_missing},
                fix_hint="Use impute_index / impute_mode / as_unknown (with allow_unknown=True).",
            )

    elif isinstance(miss, IdxImputeIndex):
        s = s.fillna(int(miss.index)) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, IdxImputeMode):
        non_na = s.dropna()
        if non_na.empty:
            return _issue(
                f"binary_map_idx: impute_mode cannot run because {column!r} has no non-missing values.",
                severity="FAIL",
                evidence={"column": column, "n_rows": int(len(s0)), "n_missing": n_missing},
                fix_hint="Use impute_index or as_unknown (allow_unknown=True).",
            )
        vc = non_na.value_counts(dropna=True)
        max_count = int(vc.max())
        top = vc[vc == max_count].index.tolist()
        mode_value = sorted(top, key=lambda x: int(x) if isinstance(x, (int, bool)) else str(x))[0]
        s = s.fillna(mode_value) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, IdxAsUnknown): # pyright: ignore[reportUnnecessaryIsInstance]
        pass  # NA treated as unknown at output stage

    else:
        raise TransformPlanApplicationError(
            "binary_map_idx: unsupported missingness spec type (plan validation failure).",
            evidence={"column": column, "missingness_type": miss.__class__.__name__},
        )

    # Stage 3: map indices → 0/1/NA
    out = pd.Series(index=s.index, dtype="float64")

    unknown_mask = pd.Series(False, index=s.index)

    for idx, v in s.items():
        if pd.isna(v):
            continue

        # treat numpy ints like ints; reject strings etc as unknown
        if isinstance(v, (int, bool)):
            iv = int(v)
        else:
            unknown_mask.at[idx] = True
            continue

        if iv in pos:
            out.at[idx] = 1.0
        elif iv in neg:
            out.at[idx] = 0.0
        elif iv in drp:
            out.at[idx] = float("nan")
        else:
            unknown_mask.at[idx] = True

    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0 and not params.allow_unknown:
        sample = s.loc[unknown_mask].dropna().astype(object).head(25).tolist()
        return _issue(
            f"binary_map_idx: encountered unknown indices in {column!r} (allow_unknown=False).",
            severity="FAIL",
            evidence={"column": column, "n_unknown": n_unknown, "unknown_sample": sample},
            fix_hint="Set allow_unknown=True (and optionally unknown_value) or adjust pos/neg/drop lists.",
        )

    unknown_value = float(params.unknown_value) if params.unknown_value is not None else float("nan")
    if n_unknown > 0:
        out.loc[unknown_mask] = unknown_value

    # Handle NA as unknown if configured
    if isinstance(miss, IdxAsUnknown):
        out.loc[pd.isna(s)] = unknown_value
    # else leave NaN

    # Stage 4: output missingness
    out2, ind, ind_name, out_miss_issue = _apply_numeric_output_missingness(
        out=out,
        output_missingness=out_miss,
        base_column=column,
    )
    if out_miss_issue is not None and out_miss_issue.severity == "FAIL":
        return out_miss_issue
    warn_issue = out_miss_issue if out_miss_issue is not None and out_miss_issue.severity == "WARN" else None

    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "binary_map_idx: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # -------------------------
    # COMMIT
    # -------------------------
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, out2.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return warn_issue

def apply_ordinal_map_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: OrdinalMapParams,
) -> ValidationIssueModel | None:
    """
    Applies OrdinalMapParams IN-PLACE to a single column (string categories).

    DATA issues => return ValidationIssueModel("FAIL") and do not mutate.
    PLAN/LLM issues => raise TransformPlanApplicationError.

    Row-dropping is NOT supported in this stage.
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="ordinal_map")

    # --- PLAN checks (defensive) ---
    if len(params.order) != len(set(params.order)):
        raise TransformPlanApplicationError(
            "ordinal_map: params.order must not contain duplicates.",
            evidence={"column": column, "order_len": len(params.order)},
        )

    miss = params.missingness

    if isinstance(miss, OrdinalAsUnknown) and not params.allow_unknown:
        raise TransformPlanApplicationError(
            "ordinal_map: missingness='as_unknown' requires allow_unknown=True.",
            evidence={"column": column},
        )

    if not isinstance(miss, (OrdinalAsUnknown, OrdinalErrorIfNA, OrdinalImputeMode, OrdinalImputeToken)):
        raise TransformPlanApplicationError(
            "ordinal_map: unsupported missingness type (plan validation failure).",
            evidence={"column": column, "missingness_type": miss.__class__.__name__},
        )

    # --- DATA path (no mutation yet) ---
    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column]
    s = s0.map(_strip_only)

    is_na = pd.isna(s)
    n_missing = int(is_na.sum())

    effective_order = list(params.order)

    # Missingness handling (no row drops)
    if isinstance(miss, OrdinalErrorIfNA):
        if n_missing > 0:
            return _issue(
                f"ordinal_map: missingness='error_if_na' but {column!r} contains missing values.",
                severity="FAIL",
                evidence={"column": column, "n_missing": n_missing},
                fix_hint="Use as_unknown (with allow_unknown=True), impute_token, or impute_mode.",
            )

    elif isinstance(miss, OrdinalImputeMode):
        non_na = s.dropna()
        if non_na.empty:
            return _issue(
                f"ordinal_map: impute_mode cannot run because {column!r} has no non-missing values.",
                severity="FAIL",
                evidence={"column": column, "n_rows": int(len(s0)), "n_missing": n_missing},
                fix_hint="Use impute_token or dummy_na-like strategy (not applicable here), or allow_unknown.",
            )
        vc = non_na.value_counts(dropna=True)
        max_count = int(vc.max())
        top = vc[vc == max_count].index.tolist()
        mode_value = sorted(top, key=lambda x: str(x))[0]
        s = s.fillna(mode_value) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, OrdinalImputeToken):
        token = miss.token
        if token not in effective_order:
            if miss.position == "prepend":
                effective_order = [token] + effective_order
            else:
                effective_order = effective_order + [token]
        s = s.fillna(token) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, OrdinalAsUnknown): # pyright: ignore[reportUnnecessaryIsInstance]
        # NA treated as unknown later
        pass

    # Build mapping dict: category -> code
    # codes are numeric; use float64 so NaN is representable
    mapping: dict[str, float] = {cat: float(params.start + i) for i, cat in enumerate(effective_order)}

    out = pd.Series(index=s.index, dtype="float64")

    unknown_mask = pd.Series(False, index=s.index)

    for idx, v in s.items():
        if pd.isna(v):
            continue

        if isinstance(v, str) and v in mapping:
            out.at[idx] = mapping[v]
        else:
            unknown_mask.at[idx] = True

    # Unknown categories
    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0 and not params.allow_unknown:
        sample = s.loc[unknown_mask].dropna().astype(object).head(25).tolist()
        return _issue(
            f"ordinal_map: encountered unknown categories in {column!r} (allow_unknown=False).",
            severity="FAIL",
            evidence={"column": column, "n_unknown": n_unknown, "unknown_sample": sample},
            fix_hint="Set allow_unknown=True (and optionally unknown_value) or extend params.order.",
        )

    unknown_value = float(params.unknown_value) if params.unknown_value is not None else float("nan")
    if n_unknown > 0:
        out.loc[unknown_mask] = unknown_value
        return_issue = return_issue or _issue(
            f"ordinal_map: unknown categories encountered in {column!r}; encoded using unknown_value.",
            severity="WARN",
            evidence={"column": column, "n_unknown": n_unknown, "unknown_value": params.unknown_value},
            fix_hint="If this is unexpected, extend params.order or disable allow_unknown to fail fast.",
        )

    # Missing values: if as_unknown, treat NA as unknown_value; else leave NaN
    if isinstance(miss, OrdinalAsUnknown) and n_missing > 0:
        out.loc[is_na] = unknown_value
        return_issue = return_issue or _issue(
            f"ordinal_map: missing values in {column!r} treated as unknown.",
            severity="WARN",
            evidence={"column": column, "n_missing": n_missing, "unknown_value": params.unknown_value},
            fix_hint="If missingness should be explicit, consider imputing a token into the order.",
        )

    # Output missingness (NumericMissingnessSpec)
    out2, ind, ind_name, out_miss_issue = _apply_numeric_output_missingness(
        out=out,
        output_missingness=params.output_missingness,
        base_column=column,
    )
    if out_miss_issue is not None and out_miss_issue.severity == "FAIL":
        return out_miss_issue
    if out_miss_issue is not None and out_miss_issue.severity == "WARN":
        return_issue = return_issue or out_miss_issue

    # PLAN: indicator name collision
    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "ordinal_map: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # --- COMMIT ---
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, out2.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return return_issue

def apply_ordinal_map_idx_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: OrdinalMapIdxParams,
) -> ValidationIssueModel | None:
    """
    Applies OrdinalMapIdxParams IN-PLACE to a single column (category indices).

    DATA issues => return ValidationIssueModel("FAIL") and do not mutate.
    PLAN/LLM issues => raise TransformPlanApplicationError.

    Row-dropping is NOT supported in this stage.
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="ordinal_map_idx")

    # --- PLAN checks (defensive) ---
    if len(params.order) != len(set(params.order)):
        raise TransformPlanApplicationError(
            "ordinal_map_idx: params.order must not contain duplicates.",
            evidence={"column": column, "order_len": len(params.order)},
        )

    inter = set(params.order) & set(params.drop)
    if inter:
        raise TransformPlanApplicationError(
            "ordinal_map_idx: params.order and params.drop must be disjoint.",
            evidence={"column": column, "overlap": sorted(inter)},
        )

    miss = params.missingness

    # Missingness requires allow_unknown if as_unknown
    if isinstance(miss, IdxAsUnknown) and not params.allow_unknown:
        raise TransformPlanApplicationError(
            "ordinal_map_idx: missingness='as_unknown' requires allow_unknown=True.",
            evidence={"column": column},
        )

    if isinstance(miss, IdxImputeIndex) and int(miss.index) in set(params.drop):
        raise TransformPlanApplicationError(
            "ordinal_map_idx: impute_index must not be in drop list.",
            evidence={"column": column, "impute_index": int(miss.index)},
        )

    if not isinstance(miss, (IdxAsUnknown, IdxErrorIfNA, IdxImputeIndex, IdxImputeMode)): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TransformPlanApplicationError(
            "ordinal_map_idx: unsupported missingness type (plan validation failure).",
            evidence={"column": column, "missingness_type": miss.__class__.__name__},
        )

    # --- DATA path (no mutation yet) ---
    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column]
    s = s0.map(_strip_only)

    is_na = pd.isna(s)
    n_missing = int(is_na.sum())

    # Missingness handling (no row drops)
    if isinstance(miss, IdxErrorIfNA):
        if n_missing > 0:
            return _issue(
                f"ordinal_map_idx: missingness='error_if_na' but {column!r} contains missing values.",
                severity="FAIL",
                evidence={"column": column, "n_missing": n_missing},
                fix_hint="Use impute_index / impute_mode / as_unknown (with allow_unknown=True).",
            )

    elif isinstance(miss, IdxImputeIndex):
        s = s.fillna(int(miss.index)) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, IdxImputeMode):
        non_na = s.dropna()
        if non_na.empty:
            return _issue(
                f"ordinal_map_idx: impute_mode cannot run because {column!r} has no non-missing values.",
                severity="FAIL",
                evidence={"column": column, "n_rows": int(len(s0)), "n_missing": n_missing},
                fix_hint="Use impute_index or as_unknown (allow_unknown=True).",
            )
        vc = non_na.value_counts(dropna=True)
        max_count = int(vc.max())
        top = vc[vc == max_count].index.tolist()
        mode_value = sorted(top, key=lambda x: int(x) if isinstance(x, (int, bool)) else str(x))[0]
        s = s.fillna(mode_value) # pyright: ignore[reportUnknownMemberType]

    elif isinstance(miss, IdxAsUnknown): # pyright: ignore[reportUnnecessaryIsInstance]
        pass  # NA treated as unknown later

    # Build mapping dict: idx -> code
    order_pos: dict[int, float] = {int(cat): float(params.start + i) for i, cat in enumerate(params.order)}
    drop_set = {int(x) for x in params.drop}

    out = pd.Series(index=s.index, dtype="float64")
    unknown_mask = pd.Series(False, index=s.index)

    for idx, v in s.items():
        if pd.isna(v):
            continue

        if isinstance(v, (int, bool)):
            iv = int(v)
        else:
            unknown_mask.at[idx] = True
            continue

        if iv in drop_set:
            out.at[idx] = float("nan")
        elif iv in order_pos:
            out.at[idx] = order_pos[iv]
        else:
            unknown_mask.at[idx] = True

    # Unknown indices
    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0 and not params.allow_unknown:
        sample = s.loc[unknown_mask].dropna().astype(object).head(25).tolist()
        return _issue(
            f"ordinal_map_idx: encountered unknown indices in {column!r} (allow_unknown=False).",
            severity="FAIL",
            evidence={"column": column, "n_unknown": n_unknown, "unknown_sample": sample},
            fix_hint="Set allow_unknown=True (and optionally unknown_value) or extend params.order / adjust params.drop.",
        )

    unknown_value = float(params.unknown_value) if params.unknown_value is not None else float("nan")
    if n_unknown > 0:
        out.loc[unknown_mask] = unknown_value
        return_issue = return_issue or _issue(
            f"ordinal_map_idx: unknown indices encountered in {column!r}; encoded using unknown_value.",
            severity="WARN",
            evidence={"column": column, "n_unknown": n_unknown, "unknown_value": params.unknown_value},
            fix_hint="If this is unexpected, extend params.order or disable allow_unknown to fail fast.",
        )

    # Missing values: if as_unknown, treat NA as unknown_value; else leave NaN
    if isinstance(miss, IdxAsUnknown) and n_missing > 0:
        out.loc[is_na] = unknown_value
        return_issue = return_issue or _issue(
            f"ordinal_map_idx: missing values in {column!r} treated as unknown.",
            severity="WARN",
            evidence={"column": column, "n_missing": n_missing, "unknown_value": params.unknown_value},
            fix_hint="If missingness should map to a specific index, use impute_index.",
        )

    # Output missingness
    out2, ind, ind_name, out_miss_issue = _apply_numeric_output_missingness(
        out=out,
        output_missingness=params.output_missingness,
        base_column=column,
    )
    if out_miss_issue is not None and out_miss_issue.severity == "FAIL":
        return out_miss_issue
    if out_miss_issue is not None and out_miss_issue.severity == "WARN":
        return_issue = return_issue or out_miss_issue

    # PLAN: indicator name collision
    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "ordinal_map_idx: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # --- COMMIT ---
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, out2.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return return_issue

def apply_to_numeric_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: ToNumericParams,
) -> ValidationIssueModel | None:
    """
    Convert a single column to numeric IN-PLACE.

    PLAN/LLM faults => raise TransformPlanApplicationError
    DATA faults     => return ValidationIssueModel("FAIL") and do not mutate
    WARN            => return ValidationIssueModel("WARN") and apply
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="to_numeric")

    # PLAN: missingness must be supported (no row drops)
    if not isinstance(
        params.missingness,
        (NumericKeepNA, NumericAddMissingIndicator, NumericErrorIfNA, NumericImputeMean, NumericImputeMedian),
    ): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TransformPlanApplicationError(
            "to_numeric: unsupported missingness strategy (row dropping disallowed / unknown type).",
            evidence={"column": column, "missingness_type": params.missingness.__class__.__name__},
        )

    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column].map(_strip_only)

    # Detect coercion failures deterministically
    coerced = pd.to_numeric(s0, errors="coerce")
    invalid_mask = (~pd.isna(s0)) & pd.isna(coerced)
    n_invalid = int(invalid_mask.sum())

    if params.errors == "raise":
        if n_invalid > 0:
            sample = s0.loc[invalid_mask].astype(object).head(25).tolist()
            return _issue(
                f"to_numeric: errors='raise' but non-numeric values found in {column!r}.",
                severity="FAIL",
                evidence={"column": column, "n_invalid": n_invalid, "invalid_sample": sample},
                fix_hint="Use errors='coerce' or clean upstream / pick a different encoding.",
            )
        numeric = pd.to_numeric(s0, errors="raise")  # safe now
    elif params.errors == "coerce":
        numeric = coerced
        if n_invalid > 0:
            sample = s0.loc[invalid_mask].astype(object).head(25).tolist()
            return_issue = _issue(
                f"to_numeric: coerced non-numeric values to NA in {column!r}.",
                severity="WARN",
                evidence={"column": column, "n_coerced_to_na": n_invalid, "sample": sample},
                fix_hint="If this is unexpected, use errors='raise' or clean upstream.",
            )
    else:
        raise TransformPlanApplicationError(
            "to_numeric: invalid params.errors (plan validation failure).",
            evidence={"column": column, "errors": str(params.errors)},
        )

    numeric = numeric.astype("float64")

    # Apply numeric missingness (no mutation yet)
    out2, ind, ind_name, miss_issue = _apply_numeric_output_missingness(
        out=numeric,
        output_missingness=params.missingness,
        base_column=column,
    )
    if miss_issue is not None and miss_issue.severity == "FAIL":
        return miss_issue
    if miss_issue is not None and miss_issue.severity == "WARN":
        return_issue = return_issue or miss_issue

    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "to_numeric: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # COMMIT
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, out2.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]
    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return return_issue


def apply_log1p_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: Log1pParams,
) -> ValidationIssueModel | None:
    """
    Apply log1p transform to a single column IN-PLACE.

    PLAN/LLM faults => raise TransformPlanApplicationError
    DATA faults     => return ValidationIssueModel("FAIL") and do not mutate
    WARN            => return ValidationIssueModel("WARN") and apply
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="log1p")

    if not isinstance(
        params.missingness,
        (NumericKeepNA, NumericAddMissingIndicator, NumericErrorIfNA, NumericImputeMean, NumericImputeMedian),
    ): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TransformPlanApplicationError(
            "log1p: unsupported missingness strategy (row dropping disallowed / unknown type).",
            evidence={"column": column, "missingness_type": params.missingness.__class__.__name__},
        )

    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column].map(_strip_only)
    numeric = pd.to_numeric(s0, errors="coerce").astype("float64")

    invalid_mask = (~pd.isna(s0)) & pd.isna(numeric)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        sample = s0.loc[invalid_mask].astype(object).head(25).tolist()
        return _issue(
            f"log1p: non-numeric values found in {column!r}.",
            severity="FAIL",
            evidence={"column": column, "n_invalid": n_invalid, "invalid_sample": sample},
            fix_hint="Run to_numeric first or choose a different encoding.",
        )

    # Apply numeric missingness BEFORE transform (no mutation yet)
    num2, ind, ind_name, miss_issue = _apply_numeric_output_missingness(
        out=numeric,
        output_missingness=params.missingness,
        base_column=column,
    )
    if miss_issue is not None and miss_issue.severity == "FAIL":
        return miss_issue
    if miss_issue is not None and miss_issue.severity == "WARN":
        return_issue = return_issue or miss_issue

    # Domain checks (DATA)
    vals = num2.dropna()
    if not vals.empty:
        if (vals <= -1.0).any():
            bad = vals[vals <= -1.0].head(25).tolist()
            return _issue(
                f"log1p: values <= -1 found in {column!r} (log1p undefined).",
                severity="FAIL",
                evidence={"column": column, "n_bad": int((vals <= -1.0).sum()), "bad_sample": bad},
                fix_hint="Clip/filter upstream or avoid log1p for this column.",
            )
        if not params.allow_negative and (vals < 0.0).any():
            bad = vals[vals < 0.0].head(25).tolist()
            return _issue(
                f"log1p: negative values found in {column!r} but allow_negative=False.",
                severity="FAIL",
                evidence={"column": column, "n_negative": int((vals < 0.0).sum()), "negative_sample": bad},
                fix_hint="Set allow_negative=True or avoid log1p.",
            )

    transformed = pd.Series(np.log1p(num2), index=num2.index)

    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "log1p: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # COMMIT
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, transformed.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]
    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return return_issue


def apply_standardize_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: StandardizeParams,
) -> ValidationIssueModel | None:
    """
    Standardize a single column IN-PLACE: (x - mean) / std.

    PLAN/LLM faults => raise TransformPlanApplicationError
    DATA faults     => return ValidationIssueModel("FAIL") and do not mutate
    WARN            => return ValidationIssueModel("WARN") and apply
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="standardize")

    if params.ddof < 0:
        raise TransformPlanApplicationError(
            "standardize: ddof must be >= 0 (plan validation failure).",
            evidence={"column": column, "ddof": int(params.ddof)},
        )
    if params.eps <= 0:
        raise TransformPlanApplicationError(
            "standardize: eps must be > 0 (plan validation failure).",
            evidence={"column": column, "eps": float(params.eps)},
        )

    if not isinstance(
        params.missingness,
        (NumericKeepNA, NumericAddMissingIndicator, NumericErrorIfNA, NumericImputeMean, NumericImputeMedian),
    ): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TransformPlanApplicationError(
            "standardize: unsupported missingness strategy (row dropping disallowed / unknown type).",
            evidence={"column": column, "missingness_type": params.missingness.__class__.__name__},
        )

    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column].map(_strip_only)
    numeric = pd.to_numeric(s0, errors="coerce").astype("float64")

    invalid_mask = (~pd.isna(s0)) & pd.isna(numeric)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        sample = s0.loc[invalid_mask].astype(object).head(25).tolist()
        return _issue(
            f"standardize: non-numeric values found in {column!r}.",
            severity="FAIL",
            evidence={"column": column, "n_invalid": n_invalid, "invalid_sample": sample},
            fix_hint="Run to_numeric first or choose a different encoding.",
        )

    # Apply numeric missingness BEFORE transform (no mutation yet)
    num2, ind, ind_name, miss_issue = _apply_numeric_output_missingness(
        out=numeric,
        output_missingness=params.missingness,
        base_column=column,
    )
    if miss_issue is not None and miss_issue.severity == "FAIL":
        return miss_issue
    if miss_issue is not None and miss_issue.severity == "WARN":
        return_issue = return_issue or miss_issue

    vals = num2.dropna()
    if vals.empty:
        return _issue(
            f"standardize: cannot standardize {column!r} because all values are missing after preprocessing.",
            severity="FAIL",
            evidence={"column": column, "n_rows": int(len(num2))},
            fix_hint="Use add_missing_indicator/keep_na or fix upstream missingness.",
        )

    mean = float(vals.mean())
    std = float(vals.std(ddof=int(params.ddof)))

    if std < float(params.eps):
        # near-constant column => standardize to 0 for non-missing values (apply, but WARN)
        out = num2.copy()
        out.loc[~out.isna()] = 0.0
        return_issue = return_issue or _issue(
            f"standardize: near-constant column {column!r} (std<{params.eps}); set non-missing values to 0.",
            severity="WARN",
            evidence={"column": column, "mean": mean, "std": std, "eps": float(params.eps)},
            fix_hint="Consider dropping the column upstream if it's uninformative.",
        )
    else:
        out = (num2 - mean) / std

    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "standardize: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # COMMIT
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, out.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]
    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return return_issue


def apply_minmax_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: MinMaxParams,
) -> ValidationIssueModel | None:
    """
    MinMax scale a single column IN-PLACE: (x - min) / (max - min).

    PLAN/LLM faults => raise TransformPlanApplicationError
    DATA faults     => return ValidationIssueModel("FAIL") and do not mutate
    WARN            => return ValidationIssueModel("WARN") and apply
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="minmax")

    if params.eps <= 0:
        raise TransformPlanApplicationError(
            "minmax: eps must be > 0 (plan validation failure).",
            evidence={"column": column, "eps": float(params.eps)},
        )

    if not isinstance(
        params.missingness,
        (NumericKeepNA, NumericAddMissingIndicator, NumericErrorIfNA, NumericImputeMean, NumericImputeMedian),
    ): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TransformPlanApplicationError(
            "minmax: unsupported missingness strategy (row dropping disallowed / unknown type).",
            evidence={"column": column, "missingness_type": params.missingness.__class__.__name__},
        )

    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column].map(_strip_only)
    numeric = pd.to_numeric(s0, errors="coerce").astype("float64")

    invalid_mask = (~pd.isna(s0)) & pd.isna(numeric)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        sample = s0.loc[invalid_mask].astype(object).head(25).tolist()
        return _issue(
            f"minmax: non-numeric values found in {column!r}.",
            severity="FAIL",
            evidence={"column": column, "n_invalid": n_invalid, "invalid_sample": sample},
            fix_hint="Run to_numeric first or choose a different encoding.",
        )

    # Apply numeric missingness BEFORE transform (no mutation yet)
    num2, ind, ind_name, miss_issue = _apply_numeric_output_missingness(
        out=numeric,
        output_missingness=params.missingness,
        base_column=column,
    )
    if miss_issue is not None and miss_issue.severity == "FAIL":
        return miss_issue
    if miss_issue is not None and miss_issue.severity == "WARN":
        return_issue = return_issue or miss_issue

    vals = num2.dropna()
    if vals.empty:
        return _issue(
            f"minmax: cannot scale {column!r} because all values are missing after preprocessing.",
            severity="FAIL",
            evidence={"column": column, "n_rows": int(len(num2))},
            fix_hint="Use add_missing_indicator/keep_na or fix upstream missingness.",
        )

    vmin = float(vals.min())
    vmax = float(vals.max())
    rng = vmax - vmin

    if rng < float(params.eps):
        # near-constant column => set non-missing to 0, warn
        out = num2.copy()
        out.loc[~out.isna()] = 0.0
        return_issue = return_issue or _issue(
            f"minmax: near-constant column {column!r} (range<eps); set non-missing values to 0.",
            severity="WARN",
            evidence={"column": column, "min": vmin, "max": vmax, "range": rng, "eps": float(params.eps)},
            fix_hint="Consider dropping the column upstream if it's uninformative.",
        )
    else:
        out = (num2 - vmin) / rng

    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "minmax: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # COMMIT
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, out.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]
    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return return_issue


def apply_datetime_to_epoch_column(
    df_to_change: pd.DataFrame,
    *,
    column: str,
    params: DateTimeToEpochParams,
) -> ValidationIssueModel | None:
    """
    Convert datetime column to epoch time IN-PLACE.

    Output unit is params.unit in {"s","ms","us","ns"} (epoch in that unit).
    Missingness is handled via params.missingness (NumericMissingnessSpec).

    PLAN/LLM faults => raise TransformPlanApplicationError
    DATA faults     => return ValidationIssueModel("FAIL") and do not mutate
    WARN            => return ValidationIssueModel("WARN") and apply
    """
    loc = _ensure_column_exists_unique(df_to_change, column=column, encoder="datetime_to_epoch")

    if params.unit not in ("s", "ms", "us", "ns"):
        raise TransformPlanApplicationError(
            "datetime_to_epoch: invalid unit (plan validation failure).",
            evidence={"column": column, "unit": str(params.unit)},
        )

    if not isinstance(
        params.missingness,
        (NumericKeepNA, NumericAddMissingIndicator, NumericErrorIfNA, NumericImputeMean, NumericImputeMedian),
    ): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TransformPlanApplicationError(
            "datetime_to_epoch: unsupported missingness strategy (row dropping disallowed / unknown type).",
            evidence={"column": column, "missingness_type": params.missingness.__class__.__name__},
        )

    return_issue: ValidationIssueModel | None = None

    s0 = df_to_change[column].map(_strip_only)

    # Parse
    dt = pd.to_datetime(s0, errors="coerce")
    invalid_mask = (~pd.isna(s0)) & dt.isna()
    n_invalid = int(invalid_mask.sum())

    if params.errors == "raise":
        if n_invalid > 0:
            sample = s0.loc[invalid_mask].astype(object).head(25).tolist()
            return _issue(
                f"datetime_to_epoch: errors='raise' but unparseable datetimes found in {column!r}.",
                severity="FAIL",
                evidence={"column": column, "n_invalid": n_invalid, "invalid_sample": sample},
                fix_hint="Use errors='coerce' or clean upstream / change encoding.",
            )
        # re-parse in strict mode (will not fail now)
        dt = pd.to_datetime(s0, errors="raise")
    elif params.errors == "coerce":
        if n_invalid > 0:
            sample = s0.loc[invalid_mask].astype(object).head(25).tolist()
            return_issue = _issue(
                f"datetime_to_epoch: coerced unparseable values to NA in {column!r}.",
                severity="WARN",
                evidence={"column": column, "n_coerced_to_na": n_invalid, "sample": sample},
                fix_hint="If unexpected, use errors='raise' or clean upstream.",
            )
    else:
        raise TransformPlanApplicationError(
            "datetime_to_epoch: invalid params.errors (plan validation failure).",
            evidence={"column": column, "errors": str(params.errors)},
        )

    # Normalize tz-aware to UTC then make tz-naive for integer conversion
    tz = getattr(dt.dtype, "tz", None)
    if tz is not None:
        dt_naive = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        dt_naive = dt

    # Convert to epoch in requested unit
    # dt_naive is datetime64[ns]; convert to int64 ns since epoch, then to float64
    ns_int_array: np.ndarray[Any, np.dtype[np.int64]] = dt_naive.values.view("int64") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAssignmentType]
    out_array = ns_int_array.astype("float64")
    
    # Create series and set NaN values properly using pandas mask
    out = pd.Series(out_array, index=dt_naive.index, dtype="float64")
    out = out.where(~dt_naive.isna(), np.nan)

    denom = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}[params.unit]
    out = out / denom

    # Apply numeric missingness AFTER conversion (no mutation yet)
    out2, ind, ind_name, miss_issue = _apply_numeric_output_missingness(
        out=out,
        output_missingness=params.missingness,
        base_column=column,
    )
    if miss_issue is not None and miss_issue.severity == "FAIL":
        return miss_issue
    if miss_issue is not None and miss_issue.severity == "WARN":
        return_issue = return_issue or miss_issue

    if ind is not None and ind_name is not None:
        if ind_name in df_to_change.columns and ind_name != column:
            raise TransformPlanApplicationError(
                "datetime_to_epoch: indicator column name already exists; cannot insert safely.",
                evidence={"column": column, "indicator_name": ind_name},
            )

    # COMMIT
    df_to_change.drop(columns=[column], inplace=True)
    df_to_change.insert(loc, column, out2.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]
    if ind is not None and ind_name is not None:
        df_to_change.insert(loc + 1, ind_name, ind.reindex(df_to_change.index)) # pyright: ignore[reportUnknownMemberType]

    return return_issue