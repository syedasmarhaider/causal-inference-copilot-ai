from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.models.validation import NonEmptyStr
from python.domain.workflows.tool import Tool

SimpleScalar = str | int | float | bool | None
SimpleTargetDType = Literal[
    "string",
    "str",
    "integer",
    "int",
    "float",
    "boolean",
    "bool",
    "datetime",
    "category",
    "object",
]
CoercionMode = Literal["raise", "coerce"]


class ValueReplacementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    from_value: SimpleScalar
    to_value: SimpleScalar


class ColumnTransformationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: NonEmptyStr
    target_dtype: SimpleTargetDType | None = None
    value: SimpleScalar = None
    fill_value: SimpleScalar = None
    replacements: list[ValueReplacementSpec] = Field(default_factory=list)
    datetime_format: str | None = None
    errors: CoercionMode = "raise"

    @model_validator(mode="after")
    def _validate_transformation(self) -> ColumnTransformationSpec:
        has_value = "value" in self.model_fields_set
        has_fill_value = "fill_value" in self.model_fields_set
        if (
            self.target_dtype is None
            and not has_value
            and not has_fill_value
            and not self.replacements
        ):
            raise ValueError(
                "column transformation must define target_dtype, value, fill_value, "
                "or at least one replacement"
            )
        return self

    @property
    def has_value(self) -> bool:
        return "value" in self.model_fields_set

    @property
    def has_fill_value(self) -> bool:
        return "fill_value" in self.model_fields_set


class SimpleDataTransformationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    columns: list[ColumnTransformationSpec] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_columns(self) -> SimpleDataTransformationSpec:
        columns = [str(item.column).strip() for item in self.columns]
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        if duplicates:
            raise ValueError(f"duplicate column transformations are not allowed: {duplicates}")
        return self


@dataclass(frozen=True)
class SimpleDataTransformationTool(Tool):
    NAME: ClassVar[str] = "SIMPLE_DATA_TRANSFORMATION"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Tool for deterministic dataframe column transformations from a Pydantic "
            "specification. It can set static column values, replace values, fill missing "
            "values, and cast columns to simple pandas-compatible data types without LLMs."
        )

    def transform(
        self,
        *,
        dataframe: pd.DataFrame,
        specification: SimpleDataTransformationSpec | dict[str, Any],
        copy: bool = True,
    ) -> pd.DataFrame:
        spec = (
            specification
            if isinstance(specification, SimpleDataTransformationSpec)
            else SimpleDataTransformationSpec.model_validate(specification)
        )
        result = dataframe.copy(deep=True) if copy else dataframe

        for column_spec in spec.columns:
            column = str(column_spec.column)
            if column not in result.columns:
                raise KeyError(f"Column not found in dataframe: {column!r}")

            series = result[column]
            if column_spec.has_value:
                series = pd.Series([column_spec.value] * len(result), index=result.index)

            for replacement in column_spec.replacements:
                series = _replace_value(
                    series=series,
                    from_value=replacement.from_value,
                    to_value=replacement.to_value,
                )

            if column_spec.has_fill_value:
                series = series.where(~series.isna(), column_spec.fill_value)

            if column_spec.target_dtype is not None:
                series = _cast_series(
                    series=series,
                    target_dtype=column_spec.target_dtype,
                    errors=column_spec.errors,
                    datetime_format=column_spec.datetime_format,
                )

            result[column] = series

        return result


def _replace_value(
    *,
    series: pd.Series,
    from_value: SimpleScalar,
    to_value: SimpleScalar,
) -> pd.Series:
    if from_value is None:
        return series.where(~series.isna(), to_value)
    return series.replace({from_value: to_value})


def _cast_series(
    *,
    series: pd.Series,
    target_dtype: SimpleTargetDType,
    errors: CoercionMode,
    datetime_format: str | None,
) -> pd.Series:
    normalized_dtype = _normalize_target_dtype(target_dtype)
    if normalized_dtype == "string":
        return series.astype("string")
    if normalized_dtype == "object":
        return series.astype("object")
    if normalized_dtype == "category":
        return series.astype("category")
    if normalized_dtype == "float":
        return pd.to_numeric(series, errors=errors).astype("float64")
    if normalized_dtype == "integer":
        return _cast_to_integer(series=series, errors=errors)
    if normalized_dtype == "boolean":
        return _cast_to_boolean(series=series, errors=errors)
    if normalized_dtype == "datetime":
        return pd.to_datetime(series, errors=errors, format=datetime_format)
    raise ValueError(f"Unsupported target dtype: {target_dtype!r}")


def _normalize_target_dtype(target_dtype: SimpleTargetDType) -> str:
    if target_dtype == "str":
        return "string"
    if target_dtype == "int":
        return "integer"
    if target_dtype == "bool":
        return "boolean"
    return str(target_dtype)


def _cast_to_integer(*, series: pd.Series, errors: CoercionMode) -> pd.Series:
    numeric = pd.to_numeric(series, errors=errors)
    fractional_mask = numeric.notna() & (numeric % 1 != 0)
    if bool(fractional_mask.any()):
        if errors == "raise":
            sample = numeric.loc[fractional_mask].head(25).tolist()
            raise ValueError(f"Cannot convert fractional values to integer: {sample}")
        numeric = numeric.mask(fractional_mask, pd.NA)

    dtype = "Int64" if bool(pd.isna(numeric).any()) else "int64"
    return numeric.astype(dtype)


def _cast_to_boolean(*, series: pd.Series, errors: CoercionMode) -> pd.Series:
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n"}

    def convert(value: Any) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        normalized = str(value).strip().lower()
        if normalized in true_values:
            return True
        if normalized in false_values:
            return False
        if errors == "coerce":
            return pd.NA
        raise ValueError(f"Cannot convert value to boolean: {value!r}")

    return series.map(convert).astype("boolean")


__all__ = [
    "ColumnTransformationSpec",
    "SimpleDataTransformationSpec",
    "SimpleDataTransformationTool",
    "ValueReplacementSpec",
]
