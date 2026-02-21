from __future__ import annotations

from typing import List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.implementation.workflows.nodes.transform_protocol.transform_protocol_encoding import EncodingType
from python.implementation.workflows.utils.validation import NonEmptyStr


OutcomeKind = Literal["binary", "continuous", "duration", "categorical"]
TreatmentKind = Literal["binary", "continuous", "categorical"]

FeatureKind = Literal[
    "continuous",  # real-valued numeric feature
    "binary",      # 0/1 feature (including one-hot dummies)
    "one_hot",     # explicitly produced dummy column (still 0/1)
    "ordinal",     # integer-encoded ordered categories
    "count",       # non-negative integer-ish
    "unknown",     # fallback when you cannot classify
]

class ColumnRefModel(BaseModel):
    """
    A single concrete column in the transformed dataframe.

    - name: transformed column name (used to slice df_after)
    - feature_kind: drives invariant checks (binary/one_hot must be {0,1}, etc.)
    - source_raw: enables group-wise checks (one-hot row-sum, expansion caps)
    - encoding: optional trace for post-condition checks (minmax in [0,1], etc.)
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: NonEmptyStr
    feature_kind: FeatureKind = "unknown"

    # IMPORTANT: enables group-wise validation (e.g., one-hot row sums per raw feature)
    source_raw: Optional[NonEmptyStr] = None

    # Optional trace of which encoding produced this column (useful for post-conditions)
    encoding: Optional[EncodingType] = None


class RoleColumnsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    columns: List[ColumnRefModel] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _no_duplicates(self) -> "RoleColumnsModel":
        seen: Set[str] = set()
        dups: List[str] = []
        for c in self.columns:
            if c.name in seen:
                dups.append(c.name)
            seen.add(c.name)
        if dups:
            raise ValueError(f"Duplicate column names in role columns: {sorted(set(dups))}")
        return self


class TransformedProtocolSpec(BaseModel):
    """
    Runner-facing spec:
      - Y/T/W/X resolved column names (+ minimal kind metadata)
    No dataset IDs. No protocol semantics.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    y_kind: OutcomeKind
    t_kind: TreatmentKind

    y: RoleColumnsModel
    t: RoleColumnsModel
    w: Optional[RoleColumnsModel] = None
    x: Optional[RoleColumnsModel] = None

    @property
    def y_cols(self) -> List[str]:
        return [c.name for c in self.y.columns]

    @property
    def t_cols(self) -> List[str]:
        return [c.name for c in self.t.columns]

    @property
    def w_cols(self) -> List[str]:
        return [c.name for c in self.w.columns] if self.w else []

    @property
    def x_cols(self) -> List[str]:
        return [c.name for c in self.x.columns] if self.x else []

    @property
    def wx_overlap(self) -> List[str]:
        # Allowed in EconML
        return sorted(set(self.w_cols) & set(self.x_cols))

    @property
    def all_input_cols(self) -> List[str]:
        # Useful for existence checks against df_after.columns
        out: List[str] = []
        out.extend(self.y_cols)
        out.extend(self.t_cols)
        out.extend(self.w_cols)
        out.extend(self.x_cols)
        return out

    @model_validator(mode="after")
    def _basic_role_constraints(self) -> "TransformedProtocolSpec":
        # Structural constraints that match common estimator adapters.
        if self.y_kind != "duration" and len(self.y.columns) != 1:
            raise ValueError("For non-duration outcomes, y must contain exactly 1 column.")

        if self.t_kind in ("binary", "continuous") and len(self.t.columns) != 1:
            raise ValueError("For binary/continuous treatment, t must contain exactly 1 column.")

        # Disallow overlap with Y/T (leakage / misuse). Allow W∩X.
        y_set = set(self.y_cols)
        t_set = set(self.t_cols)
        w_set = set(self.w_cols)
        x_set = set(self.x_cols)

        if y_set & t_set:
            raise ValueError(f"Y and T share columns: {sorted(y_set & t_set)}")
        if y_set & w_set:
            raise ValueError(f"Y and W share columns: {sorted(y_set & w_set)}")
        if y_set & x_set:
            raise ValueError(f"Y and X share columns: {sorted(y_set & x_set)}")
        if t_set & w_set:
            raise ValueError(f"T and W share columns: {sorted(t_set & w_set)}")
        if t_set & x_set:
            raise ValueError(f"T and X share columns: {sorted(t_set & x_set)}")

        return self