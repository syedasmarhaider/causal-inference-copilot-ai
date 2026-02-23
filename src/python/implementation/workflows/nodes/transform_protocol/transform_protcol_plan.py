from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.implementation.workflows.utils.validation import ValidationIssueModel


# =============================================================================
# 0) Typed constraints (Pydantic v2-friendly, no conint() calls)
# =============================================================================
PosInt = Annotated[int, Field(ge=1)]
NonNegInt = Annotated[int, Field(ge=0)]
CatIdx = NonNegInt


# =============================================================================
# Common strict bases
# =============================================================================
class _BaseParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _BaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _BaseMissingness(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# =============================================================================
# 1) Missingness specs (ENCODER-SPECIFIC, DISCRIMINATED) — NO DEFAULTS
# =============================================================================

# ---- Numeric missingness (for numeric-producing encoders/transforms)
NumericMissingnessAction = Literal[
    "keep_na",
    "add_missing_indicator",
    "impute_median",
    "impute_mean",
    "drop_rows_if_na",
    "error_if_na",
]


class NumericKeepNA(_BaseMissingness):
    action: Literal["keep_na"]


class NumericAddMissingIndicator(_BaseMissingness):
    action: Literal["add_missing_indicator"]
    suffix: str = Field(..., min_length=1)


class NumericImputeMedian(_BaseMissingness):
    action: Literal["impute_median"]


class NumericImputeMean(_BaseMissingness):
    action: Literal["impute_mean"]



class NumericErrorIfNA(_BaseMissingness):
    action: Literal["error_if_na"]


NumericMissingnessSpec = Annotated[
    Union[
        NumericKeepNA,
        NumericAddMissingIndicator,
        NumericImputeMedian,
        NumericImputeMean,
        NumericErrorIfNA,
    ],
    Field(discriminator="action"),
]


# ---- One-hot missingness (special semantics)
OneHotMissingnessAction = Literal[
    "dummy_na",         # explicit NA dummy column
    "keep_all_zero",    # NA -> all zeros (danger with drop_first)
    "impute_token",     # fill NA with token before encoding
    "impute_mode",      # fill NA with mode before encoding
    "drop_rows_if_na",
    "error_if_na",
]


class OneHotDummyNA(_BaseMissingness):
    action: Literal["dummy_na"]


class OneHotImputeToken(_BaseMissingness):
    action: Literal["impute_token"]
    token: str = Field(..., min_length=1)


class OneHotImputeMode(_BaseMissingness):
    action: Literal["impute_mode"]


class OneHotDropRowsIfNA(_BaseMissingness):
    action: Literal["drop_rows_if_na"]

OneHotMissingnessSpec = Annotated[
    Union[
        OneHotDummyNA,
        OneHotImputeToken,
        OneHotImputeMode,
        OneHotDropRowsIfNA,
    ],
    Field(discriminator="action"),
]


# ---- Binary-map missingness (categorical -> numeric)
BinaryMapMissingnessAction = Literal[
    "as_unknown",       # propagate NA as unknown (requires allow_unknown)
    "impute_token",     # fill NA with token (token must exist in mapping)
    "impute_constant",  # fill NA with constant numeric output
    "drop_rows_if_na",
    "error_if_na",
]


class BinaryMapAsUnknown(_BaseMissingness):
    action: Literal["as_unknown"]


class BinaryMapImputeToken(_BaseMissingness):
    action: Literal["impute_token"]
    token: str = Field(..., min_length=1)


class BinaryMapImputeConstant(_BaseMissingness):
    action: Literal["impute_constant"]
    value: Union[int, float] = Field(...)

class BinaryMapErrorIfNA(_BaseMissingness):
    action: Literal["error_if_na"]


BinaryMapMissingnessSpec = Annotated[
    Union[
        BinaryMapAsUnknown,
        BinaryMapImputeToken,
        BinaryMapImputeConstant,
        BinaryMapErrorIfNA,
    ],
    Field(discriminator="action"),
]


# ---- Ordinal-map missingness (categorical -> ordinal int)
OrdinalMissingnessAction = Literal[
    "as_unknown",       # propagate NA as unknown (requires allow_unknown)
    "impute_token",     # fill NA with token AND define where token sits in order
    "impute_mode",
    "drop_rows_if_na",
    "error_if_na",
]


class OrdinalAsUnknown(_BaseMissingness):
    action: Literal["as_unknown"]


class OrdinalImputeToken(_BaseMissingness):
    action: Literal["impute_token"]
    token: str = Field(..., min_length=1)
    position: Literal["prepend", "append"] = Field(...)


class OrdinalImputeMode(_BaseMissingness):
    action: Literal["impute_mode"]


class OrdinalDropRowsIfNA(_BaseMissingness):
    action: Literal["drop_rows_if_na"]


class OrdinalErrorIfNA(_BaseMissingness):
    action: Literal["error_if_na"]


OrdinalMissingnessSpec = Annotated[
    Union[
        OrdinalAsUnknown,
        OrdinalImputeToken,
        OrdinalImputeMode,
        OrdinalDropRowsIfNA,
        OrdinalErrorIfNA,
    ],
    Field(discriminator="action"),
]


# ---- Index-based missingness (for *_idx encoders)
IdxMissingnessAction = Literal[
    "as_unknown",       # propagate NA as unknown (requires allow_unknown)
    "impute_index",     # fill NA with a specific category index
    "impute_mode",
    "drop_rows_if_na",
    "error_if_na",
]


class IdxAsUnknown(_BaseMissingness):
    action: Literal["as_unknown"]


class IdxImputeIndex(_BaseMissingness):
    action: Literal["impute_index"]
    index: CatIdx = Field(...)


class IdxImputeMode(_BaseMissingness):
    action: Literal["impute_mode"]

class IdxErrorIfNA(_BaseMissingness):
    action: Literal["error_if_na"]


IdxMissingnessSpec = Annotated[
    Union[
        IdxAsUnknown,
        IdxImputeIndex,
        IdxImputeMode,
        IdxErrorIfNA,
    ],
    Field(discriminator="action"),
]


# =============================================================================
# 2) EncodingSpec (STRICT) — params validated by discriminated union — NO DEFAULTS
# =============================================================================
EncodingType = Literal[
    "drop",
    "one_hot",
    "binary_map",
    "binary_map_idx",
    "ordinal_map",
    "ordinal_map_idx",
    "to_numeric",
    "log1p",
    "standardize",
    "minmax",
    "datetime_to_epoch_seconds",
]


# ---- one_hot
class OneHotParams(_BaseParams):
    missingness: OneHotMissingnessSpec = Field(...)
    drop_first: bool = Field(...)
    max_categories: Optional[PosInt] = Field(...)

class OneHotSpec(_BaseSpec):
    encoding: Literal["one_hot"]
    params: OneHotParams


# ---- binary_map
class BinaryMapParams(_BaseParams):
    mapping: Dict[str, Union[int, float]] = Field(..., min_length=1)

    allow_unknown: bool = Field(...)
    unknown_value: Optional[Union[int, float]] = Field(...)

    missingness: BinaryMapMissingnessSpec = Field(...)
    output_missingness: NumericMissingnessSpec = Field(...)

    @model_validator(mode="after")
    def _binary_map_sanity(self) -> "BinaryMapParams":
        if isinstance(self.missingness, BinaryMapAsUnknown) and not self.allow_unknown:
            raise ValueError("binary_map: missingness='as_unknown' requires allow_unknown=True.")

        if isinstance(self.missingness, BinaryMapImputeToken):
            if self.missingness.token not in self.mapping:
                raise ValueError(
                    f"binary_map: missingness.impute_token token={self.missingness.token!r} must exist in mapping."
                )

        return self


class BinaryMapSpec(_BaseSpec):
    encoding: Literal["binary_map"]
    params: BinaryMapParams


# ---- binary_map_idx
class BinaryMapIdxParams(_BaseParams):
    pos: List[CatIdx] = Field(..., min_length=1)
    neg: List[CatIdx] = Field(..., min_length=1)
    drop: List[CatIdx] = Field(...)  # require explicit [] if none

    allow_unknown: bool = Field(...)
    unknown_value: Optional[Union[int, float]] = Field(...)

    missingness: IdxMissingnessSpec = Field(...)
    output_missingness: NumericMissingnessSpec = Field(...)

    @model_validator(mode="after")
    def _disjoint_sets(self) -> "BinaryMapIdxParams":
        pos, neg, drp = set(self.pos), set(self.neg), set(self.drop)
        inter = (pos & neg) | (pos & drp) | (neg & drp)
        if inter:
            raise ValueError(f"binary_map_idx pos/neg/drop must be disjoint; overlap={sorted(inter)}")

        if isinstance(self.missingness, IdxAsUnknown) and not self.allow_unknown:
            raise ValueError("binary_map_idx: missingness='as_unknown' requires allow_unknown=True.")

        if isinstance(self.missingness, IdxImputeIndex):
            if self.missingness.index in drp:
                raise ValueError("binary_map_idx: impute_index must not be in drop list.")

        return self


class BinaryMapIdxSpec(_BaseSpec):
    encoding: Literal["binary_map_idx"]
    params: BinaryMapIdxParams


# ---- ordinal_map
class OrdinalMapParams(_BaseParams):
    order: List[str] = Field(..., min_length=1)
    start: int = Field(...)

    allow_unknown: bool = Field(...)
    unknown_value: Optional[int] = Field(...)

    missingness: OrdinalMissingnessSpec = Field(...)
    output_missingness: NumericMissingnessSpec = Field(...)

    @model_validator(mode="after")
    def _ordinal_sanity(self) -> "OrdinalMapParams":
        if len(self.order) != len(set(self.order)):
            raise ValueError("ordinal_map params.order must not contain duplicates.")

        if isinstance(self.missingness, OrdinalAsUnknown) and not self.allow_unknown:
            raise ValueError("ordinal_map: missingness='as_unknown' requires allow_unknown=True.")

        # Token insertion semantics are execution-time; schema ensures position is explicit.
        return self


class OrdinalMapSpec(_BaseSpec):
    encoding: Literal["ordinal_map"]
    params: OrdinalMapParams


# ---- ordinal_map_idx
class OrdinalMapIdxParams(_BaseParams):
    order: List[CatIdx] = Field(..., min_length=1)
    start: int = Field(...)
    drop: List[CatIdx] = Field(...)  # explicit []

    allow_unknown: bool = Field(...)
    unknown_value: Optional[int] = Field(...)

    missingness: IdxMissingnessSpec = Field(...)
    output_missingness: NumericMissingnessSpec = Field(...)

    @model_validator(mode="after")
    def _valid_order(self) -> "OrdinalMapIdxParams":
        if len(self.order) != len(set(self.order)):
            raise ValueError("ordinal_map_idx params.order must not contain duplicates.")
        inter = set(self.order) & set(self.drop)
        if inter:
            raise ValueError(f"ordinal_map_idx order and drop must be disjoint; overlap={sorted(inter)}")

        if isinstance(self.missingness, IdxAsUnknown) and not self.allow_unknown:
            raise ValueError("ordinal_map_idx: missingness='as_unknown' requires allow_unknown=True.")

        if isinstance(self.missingness, IdxImputeIndex) and self.missingness.index in set(self.drop):
            raise ValueError("ordinal_map_idx: impute_index must not be in drop list.")

        return self


class OrdinalMapIdxSpec(_BaseSpec):
    encoding: Literal["ordinal_map_idx"]
    params: OrdinalMapIdxParams


# ---- to_numeric
class ToNumericParams(_BaseParams):
    errors: Literal["coerce", "raise"] = Field(...)
    missingness: NumericMissingnessSpec = Field(...)


class ToNumericSpec(_BaseSpec):
    encoding: Literal["to_numeric"]
    params: ToNumericParams


# ---- log1p
class Log1pParams(_BaseParams):
    allow_negative: bool = Field(...)
    missingness: NumericMissingnessSpec = Field(...)


class Log1pSpec(_BaseSpec):
    encoding: Literal["log1p"]
    params: Log1pParams


# ---- standardize
class StandardizeParams(_BaseParams):
    ddof: int = Field(...)
    eps: float = Field(...)
    missingness: NumericMissingnessSpec = Field(...)


class StandardizeSpec(_BaseSpec):
    encoding: Literal["standardize"]
    params: StandardizeParams


# ---- minmax
class MinMaxParams(_BaseParams):
    eps: float = Field(...)
    missingness: NumericMissingnessSpec = Field(...)


class MinMaxSpec(_BaseSpec):
    encoding: Literal["minmax"]
    params: MinMaxParams


# ---- datetime_to_epoch_seconds
class DateTimeToEpochParams(_BaseParams):
    errors: Literal["coerce", "raise"] = Field(...)
    unit: Literal["s", "ms", "us", "ns"] = Field(...)
    missingness: NumericMissingnessSpec = Field(...)


class DateTimeToEpochSpec(_BaseSpec):
    encoding: Literal["datetime_to_epoch_seconds"]
    params: DateTimeToEpochParams


EncodingSpec = Annotated[
    Union[
        OneHotSpec,
        BinaryMapSpec,
        BinaryMapIdxSpec,
        OrdinalMapSpec,
        OrdinalMapIdxSpec,
        ToNumericSpec,
        Log1pSpec,
        StandardizeSpec,
        MinMaxSpec,
        DateTimeToEpochSpec,
    ],
    Field(discriminator="encoding"),
]


# =============================================================================
# 3) ColumnPlan + TransformPlan (STRICT)
# =============================================================================
class ColumnPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    encoding: EncodingSpec = Field(...)


class TransformPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    columns: List[ColumnPlanModel] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_columns(self) -> "TransformPlanModel":
        cols = [c.column for c in self.columns]
        if len(cols) != len(set(cols)):
            dup = sorted({c for c in cols if cols.count(c) > 1})
            raise ValueError(f"Duplicate column plans are not allowed: {dup}")
        return self


# =============================================================================
# 4) Optional external check: verify column names exist in df
# =============================================================================
def validate_plan_against_df_columns(
    *,
    plan: TransformPlanModel,
    df_columns: Sequence[str],
    require_full_coverage: bool = False,
) -> List[ValidationIssueModel]:
    issues : List[ValidationIssueModel] = []
    df_set = set(df_columns)

    for cp in plan.columns:
        if cp.column not in df_set:
            issues.append(
                ValidationIssueModel(
                    severity="FAIL",
                    message=f"Plan column {cp.column!r} does not exist in dataset columns.",
                    evidence={"column": cp.column, "available_columns": sorted(df_set)},
                )
            )

    if require_full_coverage:
        plan_set = {c.column for c in plan.columns}
        missing = sorted([c for c in df_columns if c not in plan_set])
        if missing:
            issues.append(
                ValidationIssueModel(
                    severity="FAIL",
                    message=f"Plan does not cover all dataset columns; missing columns: {missing}",
                    evidence={"missing_columns": missing, "plan_columns": sorted(plan_set)},
                )
            )

    return issues