from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from python.domain.models.models import NonEmptyStr

InclusionOperator = Literal["==", "in", "not_in", ">=", "<=", ">", "<"]


class InclusionRuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    column: NonEmptyStr
    op: InclusionOperator
    values: List[NonEmptyStr] = Field(default_factory=list)


@dataclass(frozen=True)
class DataProcessingTool:
    """
    Strict semantics:
      - No value coercion/parsing. Values are applied as provided.
      - If the operation cannot be applied (dtype mismatch, invalid compare), raise.
      - Rows with NA in the target column are excluded for all ops.
      - Rules are ANDed together.
    """

    def apply_inclusion_rules(
        self,
        df: pd.DataFrame,
        rules: Sequence[InclusionRuleModel],
        *,
        copy: bool = True,
        deep_copy: bool = True,
    ) -> pd.DataFrame:

        if not rules:
            return df.copy(deep=deep_copy) if copy else df

        missing = [str(r.column) for r in rules if str(r.column) not in df.columns]
        if missing:
            raise KeyError(
                f"InclusionRuleModel.column not found in df: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        mask = pd.Series(True, index=df.index)

        for r in rules:
            col = str(r.column)
            s = df[col]
            non_na = s.notna()

            # Validate values cardinality by op
            if r.op in ("==", ">=", "<=", ">", "<"):
                if len(r.values) != 1:
                    raise ValueError(
                        f"Rule on '{col}' with op '{r.op}' requires exactly 1 value; got {len(r.values)}."
                    )
            elif r.op in ("in", "not_in"):
                if len(r.values) < 1:
                    raise ValueError(f"Rule on '{col}' with op '{r.op}' requires a non-empty values list.")

            try:
                rule_mask = self._apply_op(series=s, non_na=non_na, op=r.op, values=r.values, column=col)
            except Exception as e:
                # Wrap with context, preserve original exception as cause
                raise TypeError(
                    f"Failed applying inclusion rule: column='{col}', op='{r.op}', "
                    f"values={r.values}, series_dtype='{s.dtype}'. Reason: {e}"
                ) from e

            mask &= rule_mask
            if not mask.any():
                out = df.iloc[0:0]
                return out.copy(deep=deep_copy) if copy else out

        out = df.loc[mask]
        return out.copy(deep=deep_copy) if copy else out

    @staticmethod
    def _apply_op(
        *,
        series: pd.Series,
        non_na: pd.Series,
        op: InclusionOperator,
        values: List[str],
        column: str,
    ) -> pd.Series:
        if op == "==":
            v = values[0]
            return non_na & (series == v)

        if op == "in":
            # pandas will compare as-is; if dtype mismatch causes odd behavior, that's on the data/spec
            return non_na & series.isin(values)

        if op == "not_in":
            return non_na & ~series.isin(values)

        # Comparisons: will raise if invalid (e.g., numeric column vs string threshold, etc.)
        v = values[0]
        if op == ">=":
            return non_na & (series >= v)
        if op == "<=":
            return non_na & (series <= v)
        if op == ">":
            return non_na & (series > v)
        if op == "<":
            return non_na & (series < v)

        # should be unreachable due to Literal typing
        raise ValueError(f"Unsupported op '{op}' for column '{column}'.")