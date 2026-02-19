from __future__ import annotations

from typing import Any, Dict, Final, List, Literal, NotRequired, TypedDict, Union

from python.workflows.state.dataset_state import DatasetState


# =============================================================================
# Inference-ready state (FINAL v1)
# - This state describes a *prepared* dataset ready for EconML:
#   - no NaNs (unless you *explicitly* allow missing later)
#   - categorical features encoded
#   - numeric features optionally scaled/transformed
#   - strict, deterministic column lists for T / Y / W / X
# - No alias mapping: normalization is already applied to prepared_dataset.
# =============================================================================

TreatmentKind = Literal["binary", "continuous", "categorical"]
OutcomeKind = Literal["binary", "continuous", "categorical"]

PreparedColumnRole = Literal["T", "Y", "W", "X", "other"]
FeatureSetKey = Literal["W", "X", "XW"]

FeatureEncoding = Literal["none", "one_hot", "ordinal"]
ImputationStrategy = Literal["none", "drop_rows", "mean", "median", "mode", "constant"]
ScaleStrategy = Literal["none", "standardize", "minmax", "log"]

DEFAULT_FEATURE_SET_KEYS: Final[List[FeatureSetKey]] = ["W", "X", "XW"]


# =============================================================================
# Column-level transformation plan (per-column, deterministic)
# =============================================================================

class ColumnTransform(TypedDict, total=False):
    """
    Per-column transformation instructions applied by the preparation node.

    encoding:
      - none: pass through as-is (must already be numeric if role is W/X)
      - one_hot: expand to multiple columns (names stored in PreparedColumnMeta.output_names)
      - ordinal: map categories to integers (mapping is stored in notes or external artifact if needed)

    imputation:
      - none: column must have no missing after exclusions (prep node enforces)
      - drop_rows: any row with missing in this column is removed
      - mean/median: numeric imputation
      - mode: categorical/boolean imputation
      - constant: fill_value must be provided

    scaling:
      - none: pass through
      - standardize/minmax/log: numeric-only transforms
    """

    encoding: FeatureEncoding
    imputation: ImputationStrategy
    fill_value: NotRequired[Union[str, int, float]]
    scaling: ScaleStrategy


class PreparedColumnMeta(TypedDict):
    """
    Metadata for a column *after* preparation.

    Important:
      - name = original source column name
      - output_names = actual columns produced in prepared dataset
        - if encoding=="none": output_names=[name]
        - if encoding=="one_hot": output_names=[<expanded cols...>]
      - role is with respect to EconML inputs (T/Y/W/X).
    """

    name: str
    role: PreparedColumnRole
    source_dtype: str
    inferred_kind: Literal["NUMERIC", "DATETIME", "BOOLEAN", "CATEGORICAL", "OTHER"]

    # profiling on prepared data
    missing_rate: float
    n_unique: int

    transform: ColumnTransform
    output_names: List[str]

    notes: NotRequired[str]


# =============================================================================
# Treatment / Outcome specs (prepared = already normalized)
# =============================================================================

class PreparedBinaryTreatment(TypedDict):
    kind: Literal["binary"]
    column: str
    treated: str
    control: str


class PreparedCategoricalTreatment(TypedDict):
    kind: Literal["categorical"]
    column: str
    levels: List[str]
    baseline: str


class PreparedContinuousTreatment(TypedDict, total=False):
    kind: Literal["continuous"]
    column: str
    unit: NotRequired[str]
    transform: NotRequired[ScaleStrategy]
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]


PreparedTreatment = PreparedBinaryTreatment | PreparedCategoricalTreatment | PreparedContinuousTreatment


class PreparedBinaryOutcome(TypedDict):
    kind: Literal["binary"]
    column: str
    event: str
    non_event: str


class PreparedCategoricalOutcome(TypedDict):
    kind: Literal["categorical"]
    column: str
    levels: List[str]
    baseline: str


class PreparedContinuousOutcome(TypedDict, total=False):
    kind: Literal["continuous"]
    column: str
    unit: NotRequired[str]
    transform: NotRequired[ScaleStrategy]
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]


PreparedOutcome = PreparedBinaryOutcome | PreparedCategoricalOutcome | PreparedContinuousOutcome


# =============================================================================
# Audit + metrics (prep node outputs)
# =============================================================================

class ExclusionApplicationSummary(TypedDict):
    n_before: int
    n_after: int
    rules: List[Dict[str, Any]]  # node-specific audit trail


class PreparationMetrics(TypedDict, total=False):
    n_rows_source: int
    n_rows_after_exclusions: int
    n_rows_final: int

    n_treated: int
    n_control: int
    treated_share: float

    n_event: int
    n_non_event: int

    max_missing_rate_W: float
    max_missing_rate_X: float

    n_features_W: int
    n_features_X: int
    n_features_XW: int


# =============================================================================
# InferenceReadyState (FINAL)
# =============================================================================

class InferenceReadyState(TypedDict):
    """
    FINAL v1 contract:

    prepared_dataset:
      - DatasetState descriptor for the *prepared* artifact (not raw df)
      - should include schema/summary for the prepared columns

    Authoritative model inputs:
      - T_col: single treatment column name in prepared dataset (after normalization)
      - Y_col: single outcome column name in prepared dataset (v1)
      - W_cols: list of prepared covariate columns (already encoded)
      - X_cols: list of prepared effect-modifier columns (already encoded)

    feature_sets:
      - derived, must be consistent:
        - feature_sets["W"] == W_cols
        - feature_sets["X"] == X_cols
        - feature_sets["XW"] == X_cols + W_cols (stable union: X then W, de-duped)
    """

    prepared_dataset: DatasetState

    treatment: PreparedTreatment
    outcome: PreparedOutcome

    T_col: str
    Y_col: str

    W_cols: List[str]
    X_cols: List[str]

    feature_sets: Dict[FeatureSetKey, List[str]]

    prepared_columns: List[PreparedColumnMeta]
    exclusions_summary: ExclusionApplicationSummary
    metrics: PreparationMetrics

    error: NotRequired[str]


# =============================================================================
# Helpers (pure, structural)
# =============================================================================

def build_feature_sets(*, W_cols: List[str], X_cols: List[str]) -> Dict[FeatureSetKey, List[str]]:
    # stable union, X then W, de-duped
    seen: set[str] = set()
    xw: List[str] = []
    for c in [*X_cols, *W_cols]:
        if c not in seen:
            seen.add(c)
            xw.append(c)
    return {"W": list(W_cols), "X": list(X_cols), "XW": xw}


def validate_inference_ready_state(state: InferenceReadyState) -> List[str]:
    issues: List[str] = []

    err = state.get("error")
    if isinstance(err, str) and err.strip():
        issues.append(f"state.error set: {err}")
        return issues  # fail-fast

    # Basic presence
    for k in ("prepared_dataset", "treatment", "outcome", "T_col", "Y_col", "W_cols", "X_cols", "feature_sets", "prepared_columns"):
        if k not in state:
            issues.append(f"Missing key: {k}")

    # treatment column consistency
    t = state.get("treatment")
    tcol = state.get("T_col")

    if t.get("column") != tcol:
            issues.append(f"Mismatch: treatment.column='{t.get('column')}' vs T_col='{tcol}'")

    # outcome column consistency
    o = state.get("outcome")
    ycol = state.get("Y_col")
    if o.get("column") != ycol:
            issues.append(f"Mismatch: outcome.column='{o.get('column')}' vs Y_col='{ycol}'")

    # feature_sets consistency
    fs = state.get("feature_sets")
    W_cols = state.get("W_cols", [])
    X_cols = state.get("X_cols", [])
    if fs.get("W") != W_cols:
            issues.append("feature_sets['W'] must equal W_cols")
    if fs.get("X") != X_cols:
            issues.append("feature_sets['X'] must equal X_cols")

    expected = build_feature_sets(W_cols=W_cols, X_cols=X_cols)["XW"]
    if fs.get("XW") != expected:
            issues.append("feature_sets['XW'] must equal stable union of X_cols then W_cols")

    return issues
