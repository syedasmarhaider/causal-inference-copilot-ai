from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.implementation.workflows.utils.validation import NonEmptyStr

PosInt = Annotated[int, Field(ge=1)]

EncodingPreset = Literal[
    # structural
    "drop",
    "passthrough",
    # categorical
    "cat_onehot",
    # numeric
    "num_standard",
    "num_minmax",
    "num_log1p",
    # datetime
    "datetime_epoch_seconds",
    # explicit mapping
    "map_binary",
    "map_ordinal",
]
     


# ----------------------------
# Params (small + optional)
# ----------------------------
class _BaseParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DropParams(_BaseParams):
    preset: Literal["drop"]


class PassthroughParams(_BaseParams):
    preset: Literal["passthrough"]


class CatOneHotParams(_BaseParams):
    preset: Literal["cat_onehot"]
    drop_first: bool = False
    handle_unknown: Literal["ignore", "error"] = "ignore"
    # truly optional (None means "no cap")
    max_categories: Optional[PosInt] = None
    missing: Literal["impute_token", "dummy_na", "error"] = "impute_token"
    missing_token: NonEmptyStr = "__MISSING__"

    @model_validator(mode="after")
    def _validate_cat_onehot(self) -> "CatOneHotParams":
        if self.missing in ("dummy_na", "error"):
            # token is ignored in these modes, but keep it non-empty anyway
            return self
        # missing == "impute_token"
        if not self.missing_token:
            raise ValueError("cat_onehot: missing_token must be non-empty when missing='impute_token'.")
        return self


class NumParams(_BaseParams):
    impute: Literal["median", "mean"] = "median"
    add_missing_indicator: bool = True


class NumStandardParams(NumParams):
    preset: Literal["num_standard"]


class NumMinMaxParams(NumParams):
    preset: Literal["num_minmax"]
    eps: float = 1e-12
    
    @model_validator(mode="after")
    def _validate_minmax(self) -> "NumMinMaxParams":
        if not (self.eps > 0.0):
            raise ValueError("num_minmax: eps must be > 0.")
        return self


class NumLog1pParams(NumParams):
    preset: Literal["num_log1p"]
    allow_negative: bool = False
    then_scale: Literal["none", "standard", "minmax"] = "none"


class DateTimeEpochParams(_BaseParams):
    preset: Literal["datetime_epoch_seconds"]
    errors: Literal["coerce", "raise"] = "coerce"
    unit: Literal["s", "ms", "us", "ns"] = "s"
    add_missing_indicator: bool = True


class MapBinaryParams(_BaseParams):
    preset: Literal["map_binary"]
    mapping: Dict[NonEmptyStr, float] = Field(..., min_length=1)

    allow_unknown: bool = True
    unknown_value: Optional[float] = None

    missing: Literal["as_unknown", "impute_token", "error"] = "as_unknown"
    missing_token: Optional[NonEmptyStr] = None  # required if missing="impute_token"

    @model_validator(mode="after")
    def _validate_map_binary(self) -> "MapBinaryParams":
        if self.missing == "impute_token" and self.missing_token is None:
            raise ValueError("map_binary: missing_token required when missing='impute_token'.")
        # Strong safety: avoid NaNs escaping unless explicitly configured
        if self.allow_unknown and self.unknown_value is None:
            raise ValueError("map_binary: unknown_value required when allow_unknown=True (avoid NaNs).")
        if self.missing == "as_unknown" and self.unknown_value is None:
            raise ValueError("map_binary: unknown_value required when missing='as_unknown' (avoid NaNs).")
        return self


class MapOrdinalParams(_BaseParams):
    preset: Literal["map_ordinal"]
    order: List[NonEmptyStr] = Field(..., min_length=1)
    start: int = 0

    allow_unknown: bool = True
    unknown_value: Optional[int] = None

    missing: Literal["as_unknown", "impute_token", "error"] = "as_unknown"
    missing_token: Optional[NonEmptyStr] = None
    token_position: Optional[Literal["prepend", "append"]] = None  # required if missing="impute_token"

    @model_validator(mode="after")
    def _validate_map_ordinal(self) -> "MapOrdinalParams":
        if len(self.order) != len(set(self.order)):
            raise ValueError("map_ordinal: 'order' must not contain duplicates.")
        if self.missing == "impute_token":
            if self.missing_token is None or self.token_position is None:
                raise ValueError(
                    "map_ordinal: missing_token and token_position required when missing='impute_token'."
                )
        # Strong safety: avoid NaNs escaping unless explicitly configured
        if self.allow_unknown and self.unknown_value is None:
            raise ValueError("map_ordinal: unknown_value required when allow_unknown=True (avoid NaNs).")
        if self.missing == "as_unknown" and self.unknown_value is None:
            raise ValueError("map_ordinal: unknown_value required when missing='as_unknown' (avoid NaNs).")
        return self


EncodingPresetSpec = Annotated[
    Union[
        DropParams,
        PassthroughParams,
        CatOneHotParams,
        NumStandardParams,
        NumMinMaxParams,
        NumLog1pParams,
        DateTimeEpochParams,
        MapBinaryParams,
        MapOrdinalParams,
    ],
    Field(discriminator="preset"),
]


# ----------------------------
# Plan models
# ----------------------------
class ColumnEncodingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    column: NonEmptyStr
    role: Literal["covariate", "effect_modifier"]  
    encoding: EncodingPresetSpec


class TransformPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    columns: List[ColumnEncodingPlan] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_plan(self) -> "TransformPlan":
        cols = [c.column for c in self.columns]
        if len(cols) != len(set(cols)):
            dup = sorted({c for c in cols if cols.count(c) > 1})
            raise ValueError(f"TransformPlan: duplicate column entries are not allowed: {dup}")

        has_covariate = any(c.role == "covariate" for c in self.columns)
        has_effect_modifier = any(c.role == "effect_modifier" for c in self.columns)
        if not has_covariate and not has_effect_modifier:
            raise ValueError("TransformPlan: must contain at least one covariate or effect_modifier column.")

        return self