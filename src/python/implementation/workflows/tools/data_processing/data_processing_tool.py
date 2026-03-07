from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Sequence, Union, cast

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)
from typing_extensions import Literal

from python.domain.models.models import NonEmptyStr
from python.domain.workflows.tool import Tool


# -----------------------------------------------------------------------------
# Types / constants
# -----------------------------------------------------------------------------

IncExcOperator = Literal["==", "in", "not_in", ">=", "<=", ">", "<"]
RuleCombineMode = Literal["all", "any"]

SCALAR_OPS = {"==", ">=", "<=", ">", "<"}
SET_OPS = {"in", "not_in"}
ALLOWED_OPS = SCALAR_OPS | SET_OPS

# Strict, no coercion, plus explicit NA/null support via None.
ExcIncValue = Union[NonEmptyStr, StrictInt, StrictFloat, StrictBool, None]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class IncExcRuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: NonEmptyStr
    op: IncExcOperator
    values: List[ExcIncValue] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]

    @field_validator("values")
    @classmethod
    def _strip_string_values(cls, v: List[ExcIncValue]) -> List[ExcIncValue]:
        out: List[ExcIncValue] = []
        for item in v:
            if item is None:
                out.append(item)
                continue

            if isinstance(item, str):
                s = item.strip()
                if not s:
                    raise ValueError("values cannot contain empty strings")
                out.append(cast(ExcIncValue, s))
            else:
                out.append(item)
        return out

    @model_validator(mode="after")
    def _validate_op_values(self) -> "IncExcRuleModel":
        if self.op in SCALAR_OPS:
            if len(self.values) != 1:
                raise ValueError(
                    f"Rule on '{self.column}' with op '{self.op}' requires exactly 1 value; "
                    f"got {len(self.values)}."
                )
            if self.op in {">=", "<=", ">", "<"} and self.values[0] is None:
                raise ValueError(
                    f"Rule on '{self.column}' with op '{self.op}' cannot use None/NA."
                )

        elif self.op in SET_OPS:
            if len(self.values) < 1:
                raise ValueError(
                    f"Rule on '{self.column}' with op '{self.op}' requires a non-empty values list."
                )
        else:
            raise ValueError(f"Unsupported op '{self.op}' for column '{self.column}'.")

        return self


class ExclusionRulesModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    exclusion_rules: List[IncExcRuleModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]


class InclusionRulesModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    group_key: NonEmptyStr
    inclusion_rules: List[IncExcRuleModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    is_counterfactual: bool = False


class InclusionPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rules: List[InclusionRulesModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]


# -----------------------------------------------------------------------------
# Tool
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DataProcessingTool(Tool):
    NAME: ClassVar[str] = "DATA_PROCESSING"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Tool for processing datasets, including applying inclusion and exclusion "
            "rules to filter rows based on column values."
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def apply_inclusion_rules(
        self,
        df: pd.DataFrame,
        rules: Sequence[IncExcRuleModel],
        *,
        combine: RuleCombineMode = "all",
        copy: bool = True,
        deep_copy: bool = True,
    ) -> pd.DataFrame:
        """
        Keep rows that match the inclusion rules.

        Defaults to AND semantics via combine='all'.
        """
        matched_mask = self._evaluate_rules(df=df, rules=rules, combine=combine)
        out = df.loc[matched_mask]
        return out.copy(deep=deep_copy) if copy else out

    def apply_exclusion_rules(
        self,
        df: pd.DataFrame,
        rules: Sequence[IncExcRuleModel],
        *,
        combine: RuleCombineMode = "any",
        copy: bool = True,
        deep_copy: bool = True,
    ) -> pd.DataFrame:
        """
        Remove rows that match the exclusion rules.

        Defaults to OR semantics via combine='any':
        if a row matches any exclusion rule, it is removed.
        """
        matched_mask = self._evaluate_rules(df=df, rules=rules, combine=combine)
        out = df.loc[~matched_mask]
        return out.copy(deep=deep_copy) if copy else out

    def apply_inclusion_model(
        self,
        df: pd.DataFrame,
        model: InclusionRulesModel,
        *,
        copy: bool = True,
        deep_copy: bool = True,
    ) -> pd.DataFrame:
        return self.apply_inclusion_rules(
            df=df,
            rules=model.inclusion_rules,
            combine="all",
            copy=copy,
            deep_copy=deep_copy,
        )

    def apply_exclusion_model(
        self,
        df: pd.DataFrame,
        model: ExclusionRulesModel,
        *,
        copy: bool = True,
        deep_copy: bool = True,
    ) -> pd.DataFrame:
        return self.apply_exclusion_rules(
            df=df,
            rules=model.exclusion_rules,
            combine="any",
            copy=copy,
            deep_copy=deep_copy,
        )

    # -------------------------------------------------------------------------
    # Core evaluation
    # -------------------------------------------------------------------------

    def _evaluate_rules(
        self,
        *,
        df: pd.DataFrame,
        rules: Sequence[IncExcRuleModel],
        combine: RuleCombineMode,
    ) -> pd.Series:
        if combine not in {"all", "any"}:
            raise ValueError(f"Unsupported combine mode: {combine!r}")

        if not rules:
            identity = True if combine == "all" else False
            return pd.Series(identity, index=df.index, dtype=bool)

        self._validate_columns_exist(df=df, rules=rules)

        masks: List[pd.Series] = []
        for r in rules:
            s = df[str(r.column)]

            try:
                rule_mask = self._build_rule_mask(
                    series=s,
                    op=r.op,
                    values=r.values,
                    column=str(r.column),
                )
            except Exception as e:
                raise TypeError(
                    f"Failed applying rule: column='{r.column}', op='{r.op}', "
                    f"values={r.values}, series_dtype='{s.dtype}'. Reason: {e}"
                ) from e

            masks.append(rule_mask)

        if combine == "all":
            out = pd.Series(True, index=df.index, dtype=bool)
            for m in masks:
                out &= m
                if not out.any():
                    break
            return out

        out = pd.Series(False, index=df.index, dtype=bool)
        for m in masks:
            out |= m
            if out.all():
                break
        return out

    @staticmethod
    def _validate_columns_exist(
        *,
        df: pd.DataFrame,
        rules: Sequence[IncExcRuleModel],
    ) -> None:
        missing_cols = [str(r.column) for r in rules if str(r.column) not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Rule columns not found in df: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

    # -------------------------------------------------------------------------
    # Rule application
    # -------------------------------------------------------------------------

    @classmethod
    def _build_rule_mask(
        cls,
        *,
        series: pd.Series,
        op: IncExcOperator,
        values: List[ExcIncValue],
        column: str,
    ) -> pd.Series:
        is_na = series.isna()
        non_na = ~is_na

        if op == "==":
            v = values[0]
            if v is None:
                return is_na
            return non_na & (series == v)

        if op == "in":
            non_null_values = [v for v in values if v is not None]
            wants_null = any(v is None for v in values)

            mask = pd.Series(False, index=series.index, dtype=bool)

            if non_null_values:
                mask |= series.isin(non_null_values)
            if wants_null:
                mask |= is_na

            return mask

        if op == "not_in":
            non_null_values = [v for v in values if v is not None]
            excludes_null = any(v is None for v in values)

            # Base strict behavior: missing rows are not kept unless explicitly
            # expressing "not_in [None]" semantics, which still means "keep non-missing only".
            mask = non_na.copy()

            if non_null_values:
                mask &= ~series.isin(non_null_values)
            if excludes_null:
                mask &= non_na

            return mask

        if not cls._is_threshold_compatible_series(series):
            raise TypeError(
                f"Threshold op '{op}' requires numeric/bool/datetime/timedelta-like column; "
                f"got dtype '{series.dtype}' for column '{column}'."
            )

        v = values[0]
        if v is None:
            raise TypeError(
                f"Threshold op '{op}' cannot be used with None/NA on column '{column}'."
            )

        if op == ">=":
            return non_na & (series >= v)
        if op == "<=":
            return non_na & (series <= v)
        if op == ">":
            return non_na & (series > v)
        if op == "<":
            return non_na & (series < v)

        raise ValueError(f"Unsupported op '{op}' for column '{column}'.")

    @staticmethod
    def _is_threshold_compatible_series(series: pd.Series) -> bool:
        return (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_timedelta64_dtype(series)
        )