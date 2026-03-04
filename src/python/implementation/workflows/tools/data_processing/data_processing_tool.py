from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Sequence, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal

from python.domain.models.models import NonEmptyStr
from python.domain.workflows.tool import Tool


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

InclusionOperator = Literal["==", "in", "not_in", ">=", "<=", ">", "<"]

# Allow numerics/bools too (no coercion; we validate and compare "as-is")
InclusionValue = Union[NonEmptyStr, int, float, bool]


class InclusionRuleModel(BaseModel):
    """
    A single rule applied to df[column], ANDed with other rules.

    Notes:
      - values are not parsed/coerced. They are used as provided.
      - for ops requiring a scalar threshold, values must have exactly 1 element.
      - for membership ops, values must be non-empty.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: NonEmptyStr
    op: InclusionOperator
    values: List[InclusionValue] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]

    @field_validator("values")
    @classmethod
    def _strip_string_values(cls, v: List[InclusionValue]) -> List[InclusionValue]:
        # NonEmptyStr already strips, but if caller passes plain str, keep things clean.
        out: List[InclusionValue] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    raise ValueError("values cannot contain empty strings")
                out.append(s)
            else:
                out.append(item)
        return out
 

class InclusionRulesModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    group_key: NonEmptyStr
    inclusion_rules: List[InclusionRuleModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    is_counterfactual: bool = False  
    
class InclusionPlanModel(BaseModel):
     model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
     rules: List[InclusionRulesModel] = Field(default_factory=list)     # pyright: ignore[reportUnknownVariableType]
    
     
     


# -----------------------------------------------------------------------------
# Tool
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DataProcessingTool(Tool):
    NAME: ClassVar[str] = "DATA_PROCESSING"

    """
    Strict semantics:
      - No value coercion/parsing. Values are applied as provided.
      - If the operation cannot be applied (dtype mismatch, invalid compare), raise.
      - Rows with NA in the target column are excluded for all ops.
      - Rules are ANDed together.
    """

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Tool for processing datasets, including applying inclusion rules "
            "to filter rows based on column values."
        )

    def apply_inclusion_rules(
        self,
        df: pd.DataFrame,
        rules: Sequence[InclusionRuleModel],
        *,
        copy: bool = True,
        deep_copy: bool = True,
    ) -> pd.DataFrame:
        """
        Apply inclusion rules to `df` and return filtered df.

        Parameters
        ----------
        df:
            Input dataframe.
        rules:
            Rules ANDed together.
        copy / deep_copy:
            If copy=True, return a copy of the filtered df (deep=deep_copy).

        Returns
        -------
        pd.DataFrame
            Filtered dataframe (possibly empty).
        """
        if not rules:
            return df.copy(deep=deep_copy) if copy else df

        # Column existence validation up front
        missing_cols = [str(r.column) for r in rules if str(r.column) not in df.columns]
        if missing_cols:
            raise KeyError(
                f"InclusionRuleModel.column not found in df: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        mask = pd.Series(True, index=df.index, dtype=bool)

        for r in rules:
            col = str(r.column)
            s = df[col]
            non_na = s.notna()

            self._validate_rule_values(col=col, op=r.op, values=r.values)

            try:
                rule_mask = self._apply_op(series=s, non_na=non_na, op=r.op, values=r.values, column=col)
            except Exception as e:
                raise TypeError(
                    f"Failed applying inclusion rule: column='{col}', op='{r.op}', "
                    f"values={r.values}, series_dtype='{s.dtype}'. Reason: {e}"
                ) from e

            mask &= rule_mask
            if not mask.any():
                empty = df.iloc[0:0]
                return empty.copy(deep=deep_copy) if copy else empty

        out = df.loc[mask]
        return out.copy(deep=deep_copy) if copy else out

    @staticmethod
    def _validate_rule_values(*, col: str, op: InclusionOperator, values: List[InclusionValue]) -> None:
        if op in ("==", ">=", "<=", ">", "<"):
            if len(values) != 1:
                raise ValueError(
                    f"Rule on '{col}' with op '{op}' requires exactly 1 value; got {len(values)}."
                )
        elif op in ("in", "not_in"):
            if len(values) < 1:
                raise ValueError(f"Rule on '{col}' with op '{op}' requires a non-empty values list.")
        else:
            raise ValueError(f"Unsupported op '{op}' for column '{col}'.")

    @staticmethod
    def _is_numeric_like_series(series: pd.Series) -> bool:
        # “numeric-like” for threshold comparisons
        return (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_timedelta64_dtype(series)
        )

    @classmethod
    def _apply_op(
        cls,
        *,
        series: pd.Series,
        non_na: pd.Series,
        op: InclusionOperator,
        values: List[InclusionValue],
        column: str,
    ) -> pd.Series:
        # Equality / membership work for most dtypes (strict: no coercion)
        if op == "==":
            v = values[0]
            return non_na & (series == v)

        if op == "in":
            return non_na & series.isin(values)

        if op == "not_in":
            return non_na & ~series.isin(values)

        # Threshold comparisons: guard against silent lexicographic object comparisons
        if not cls._is_numeric_like_series(series):
            raise TypeError(
                f"Threshold op '{op}' requires numeric/bool/datetime-like column; "
                f"got dtype '{series.dtype}' for column '{column}'."
            )

        v = values[0]
        if op == ">=":
            return non_na & (series >= v)
        if op == "<=":
            return non_na & (series <= v)
        if op == ">":
            return non_na & (series > v)
        if op == "<":
            return non_na & (series < v)

        # unreachable due to Literal typing
        raise ValueError(f"Unsupported op '{op}' for column '{column}'.")
