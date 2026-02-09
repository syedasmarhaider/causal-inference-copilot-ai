from __future__ import annotations

from typing import Any, Dict, Final, List, Literal, NotRequired, TypedDict
from uuid import UUID

from python.workflows.state.protocol_state import ProtocolState

# ============================================================
# Inference-ready metadata state (NO raw data stored here)
# ============================================================

TreatmentKind = Literal["binary", "continuous", "categorical"]
OutcomeKind = Literal["binary", "continuous", "categorical", "duration"]

FeatureEncoding = Literal["none", "one_hot", "ordinal"]
ImputationStrategy = Literal["none", "drop_rows", "mean", "median", "mode", "constant"]
ScaleStrategy = Literal["none", "standardize", "minmax", "log"]

PreparedColumnRole = Literal["T", "Y", "Y_duration", "Y_event", "W", "X", "other"]


# -------------------------
# Column metadata (no data)
# -------------------------
class PreparedColumnMeta(TypedDict):
    """
    Metadata describing a single column as used in the prepared dataset.

    Notes:
      - role encodes how the column is used for modeling (EconML convention).
      - missing_rate is a float in [0, 1].
      - n_unique is computed on the data used for profiling (possibly sampled).
    """

    name: str
    role: PreparedColumnRole
    dtype: str

    missing_rate: float
    n_unique: int

    encoding: FeatureEncoding
    imputation: ImputationStrategy

    notes: NotRequired[str]


# -------------------------
# Treatment / outcome metadata
# -------------------------
class PreparedBinaryLabels(TypedDict, total=False):
    """
    For binary treatment T (or sometimes binary outcomes), define canonical labels and alias mapping.
    value_map: aliases -> canonical label (e.g., {"yes": "treated", "1": "treated", "no": "control"}).
    """

    treated: str
    control: str
    value_map: Dict[str, str]


class PreparedCategoricalLabels(TypedDict, total=False):
    """
    For categorical treatment T (multi-valued), define levels and baseline.
    value_map: aliases -> canonical level.
    """

    levels: List[str]
    baseline: str
    value_map: Dict[str, str]


class PreparedContinuousMeta(TypedDict, total=False):
    """
    For continuous treatment, optionally store unit/transform/clipping information.
    """

    unit: str
    transform: ScaleStrategy
    clip_min: float
    clip_max: float


class PreparedTreatment(TypedDict):
    kind: TreatmentKind
    column: str
    labels: NotRequired[PreparedBinaryLabels | PreparedCategoricalLabels]
    numeric: NotRequired[PreparedContinuousMeta]


class PreparedBinaryOutcome(TypedDict, total=False):
    """
    Binary outcome Y coding.
    column: the outcome column name
    event / non_event: canonical labels
    value_map: aliases -> canonical label
    """

    column: str
    event: str
    non_event: str
    value_map: Dict[str, str]


class PreparedCategoricalOutcome(TypedDict, total=False):
    """
    Categorical outcome coding.
    """

    column: str
    levels: List[str]
    baseline: str
    value_map: Dict[str, str]


class PreparedContinuousOutcome(TypedDict, total=False):
    """
    Continuous outcome meta.
    """

    column: str
    unit: str
    transform: ScaleStrategy
    clip_min: float
    clip_max: float


class PreparedDurationOutcome(TypedDict, total=False):
    """
    Duration outcome coding (survival-style):
      - duration_column: time-to-event (or follow-up time)
      - event_column: event indicator column
      - event_value / censor_value: how event is encoded
      - value_map: aliases -> canonical for event column
    """

    duration_column: str
    event_column: str
    event_value: str
    censor_value: str
    value_map: Dict[str, str]


class PreparedOutcome(TypedDict):
    kind: OutcomeKind
    binary: NotRequired[PreparedBinaryOutcome]
    categorical: NotRequired[PreparedCategoricalOutcome]
    continuous: NotRequired[PreparedContinuousOutcome]
    duration: NotRequired[PreparedDurationOutcome]


# -------------------------
# Audit / filtering summary
# -------------------------
class ExclusionApplicationSummary(TypedDict):
    """
    Summary of exclusions / eligibility rules applied to the dataset.
    rules is intentionally flexible so nodes can store:
      - rule identifiers
      - human-readable description
      - removed row counts
      - sample of removed indices
      - etc.
    """

    n_before: int
    n_after: int
    rules: List[Dict[str, Any]]


class PreparationMetrics(TypedDict, total=False):
    """
    Metrics to make the preparation step auditable and debuggable.
    All fields are optional because some metrics depend on outcome/treatment type.
    """

    n_rows_source: int
    n_rows_after_exclusions: int
    n_rows_final: int

    n_treated: int
    n_control: int
    treated_share: float

    n_event: int
    n_non_event: int
    n_censor: int

    max_missing_rate_W: float
    max_missing_rate_X: float


# -------------------------
# Prepared dataset artifact (pointer only)
# -------------------------
class PreparedDatasetArtifact(TypedDict):
    dataset_id: UUID
    storage_kind: Literal["DATA_REPO_CSV", "DATA_REPO_PARQUET"]
    schema_fingerprint: str
    row_count: int
    created_from_dataset_id: UUID


# -------------------------
# Inference-ready state (metadata only)
# -------------------------
class InferenceReadyState(TypedDict):
    # lineage
    source_dataset_id: UUID
    prepared: NotRequired[PreparedDatasetArtifact]  # present only if READY

    # snapshot of protocol used (for auditability)
    protocol: ProtocolState

    # resolved modeling variables
    treatment: PreparedTreatment
    outcome: PreparedOutcome

    # econml conventions
    T_col: str
    Y_cols: List[str]  # [Y] or [Y_event, Y_duration] for duration outcomes
    W_cols: List[str]  # adjustment covariates
    X_cols: List[str]  # effect modifiers

    # convenience bundles (must include DEFAULT_FEATURE_SET_KEYS)
    feature_sets: Dict[str, List[str]]  # {"W":..., "X":..., "XW":...}

    prepared_columns: List[PreparedColumnMeta]
    exclusions_summary: ExclusionApplicationSummary
    metrics: PreparationMetrics

    summary_text: NotRequired[str]
    error: NotRequired[str]


DEFAULT_FEATURE_SET_KEYS: Final[List[str]] = ["W", "X", "XW"]