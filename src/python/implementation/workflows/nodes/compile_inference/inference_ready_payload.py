from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Final, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

# --- literals / enums ---
TreatmentKind = Literal["binary", "continuous", "categorical"]
OutcomeKind = Literal["binary", "continuous", "categorical"]

PreparedColumnRole = Literal["T", "Y", "W", "X", "other"]
FeatureSetKey = Literal["W", "X", "XW"]

FeatureEncoding = Literal["none", "one_hot", "ordinal"]
ImputationStrategy = Literal["none", "drop_rows", "mean", "median", "mode", "constant"]
ScaleStrategy = Literal["none", "standardize", "minmax", "log"]

DEFAULT_FEATURE_SET_KEYS: Final[List[FeatureSetKey]] = ["W", "X", "XW"]


class ColumnTransformModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    encoding: FeatureEncoding
    imputation: ImputationStrategy
    fill_value: Optional[Union[str, int, float]] = None
    scaling: ScaleStrategy

    @model_validator(mode="after")
    def _validate_fill_value(self) -> "ColumnTransformModel":
        if self.imputation == "constant" and self.fill_value is None:
            raise ValueError("fill_value is required when imputation=='constant'")
        if self.imputation != "constant" and self.fill_value is not None:
            # keep strict: don't allow irrelevant fill_value
            raise ValueError("fill_value is only allowed when imputation=='constant'")
        return self


class PreparedColumnMetaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    role: PreparedColumnRole
    source_dtype: str
    inferred_kind: Literal["NUMERIC", "DATETIME", "BOOLEAN", "CATEGORICAL", "OTHER"]

    missing_rate: float
    n_unique: int

    transform: ColumnTransformModel
    output_names: List[str]

    notes: Optional[str] = None


# =============================================================================
# Treatment / Outcome specs (prepared = already normalized)
# =============================================================================

class PreparedBinaryTreatmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["binary"]
    column: str
    treated: str
    control: str


class PreparedCategoricalTreatmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["categorical"]
    column: str
    levels: List[str] = Field(..., min_length=2)
    baseline: str


class PreparedContinuousTreatmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["continuous"]
    column: str
    unit: Optional[str] = None
    transform: Optional[ScaleStrategy] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


PreparedTreatmentModel = Union[
    PreparedBinaryTreatmentModel,
    PreparedCategoricalTreatmentModel,
    PreparedContinuousTreatmentModel,
]


class PreparedBinaryOutcomeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["binary"]
    column: str
    event: str
    non_event: str


class PreparedCategoricalOutcomeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["categorical"]
    column: str
    levels: List[str] = Field(..., min_length=2)
    baseline: str


class PreparedContinuousOutcomeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["continuous"]
    column: str
    unit: Optional[str] = None
    transform: Optional[ScaleStrategy] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


PreparedOutcomeModel = Union[
    PreparedBinaryOutcomeModel,
    PreparedCategoricalOutcomeModel,
    PreparedContinuousOutcomeModel,
]


# =============================================================================
# Audit + metrics
# =============================================================================

class ExclusionApplicationSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    n_before: int
    n_after: int
    rules: List[Dict[str, Any]]


class PreparationMetricsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    n_rows_source: Optional[int] = None
    n_rows_after_exclusions: Optional[int] = None
    n_rows_final: Optional[int] = None

    n_treated: Optional[int] = None
    n_control: Optional[int] = None
    treated_share: Optional[float] = None

    n_event: Optional[int] = None
    n_non_event: Optional[int] = None

    max_missing_rate_W: Optional[float] = None
    max_missing_rate_X: Optional[float] = None

    n_features_W: Optional[int] = None
    n_features_X: Optional[int] = None
    n_features_XW: Optional[int] = None


# =============================================================================
# Helpers
# =============================================================================

def build_feature_sets(*, W_cols: List[str], X_cols: List[str]) -> Dict[FeatureSetKey, List[str]]:
    seen: set[str] = set()
    xw: List[str] = []
    for c in [*X_cols, *W_cols]:
        if c not in seen:
            seen.add(c)
            xw.append(c)
    return {"W": list(W_cols), "X": list(X_cols), "XW": xw}


# =============================================================================
# InferenceReady payload (Pydantic)
# =============================================================================

class InferenceReadyPayloadModel(BaseModel):
    """
    Pydantic version of your FINAL v1 contract.

    prepared_dataset:
      - keep it flexible: either a workflow State-like object exposing to_json_dict(),
        or a mapping (already serialized).
    """
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    prepared_dataset: Any

    treatment: PreparedTreatmentModel = Field(..., discriminator="kind")
    outcome: PreparedOutcomeModel = Field(..., discriminator="kind")

    T_col: str
    Y_col: str

    W_cols: List[str]
    X_cols: List[str]

    feature_sets: Dict[FeatureSetKey, List[str]]

    prepared_columns: List[PreparedColumnMetaModel]
    exclusions_summary: ExclusionApplicationSummaryModel
    metrics: PreparationMetricsModel

    error: Optional[str] = None

    # --- prepared_dataset normalization/serialization ---
    @field_validator("prepared_dataset")
    @classmethod
    def _validate_prepared_dataset(cls, v: Any) -> Any:
        if isinstance(v, Mapping):
            return dict(v) # type: ignore[return-value]
        if hasattr(v, "to_json_dict") and callable(getattr(v, "to_json_dict")):
            return v
        raise ValueError("prepared_dataset must be a Mapping or an object with to_json_dict().")

    @field_serializer("prepared_dataset")
    def _serialize_prepared_dataset(self, v: Any) -> Any:
        if isinstance(v, Mapping):
            return dict(v) # type: ignore[return-value]
        to_json = getattr(v, "to_json_dict", None)
        if callable(to_json):
            return to_json()
        return v  # should not happen due to validator

    # --- contract validations (your validate_inference_ready_state, but enforced in-model) ---
    @model_validator(mode="after")
    def _validate_contract(self) -> "InferenceReadyPayloadModel":
        if isinstance(self.error, str) and self.error.strip():
            # fail-fast like your helper
            return self

        # treatment column consistency
        if getattr(self.treatment, "column", None) != self.T_col:
            raise ValueError(f"Mismatch: treatment.column='{self.treatment.column}' vs T_col='{self.T_col}'")

        # outcome column consistency
        if getattr(self.outcome, "column", None) != self.Y_col:
            raise ValueError(f"Mismatch: outcome.column='{self.outcome.column}' vs Y_col='{self.Y_col}'")

        # feature_sets consistency
        if self.feature_sets.get("W") != self.W_cols:
            raise ValueError("feature_sets['W'] must equal W_cols")
        if self.feature_sets.get("X") != self.X_cols:
            raise ValueError("feature_sets['X'] must equal X_cols")

        expected_xw = build_feature_sets(W_cols=self.W_cols, X_cols=self.X_cols)["XW"]
        if self.feature_sets.get("XW") != expected_xw:
            raise ValueError("feature_sets['XW'] must equal stable union of X_cols then W_cols")

        return self
