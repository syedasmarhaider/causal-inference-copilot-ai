from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType


class OutcomeType(Enum):
    CONTINUOUS = auto()
    # -> Use regression losses/metrics; ATE via mean differences or AIPW/TMLE.
    #    Common in clinical endpoints (lab values).  See: Robins & Rotnitzky (AIPW),
    #    van der Laan & Rose (TMLE).  Library refs: PyWhy/DoWhy (AIPW/TMLE), EconML.

    BINARY = auto()
    # -> Use classification-aware losses; uplift metrics (Qini/AUUC) for ranking.
    #    Marketing + binary clinical endpoints (event/no-event).  See: Künzel et al. (2019) S/T/X-Learners.
    #    Libraries: CausalML (uplift), EconML (learners), scikit-uplift.

    SURVIVAL = auto()
    # -> Triggers causal survival pipeline (Cox/AFT + orthogonalization / IPCW).
    #    Not the default tabular pipeline; needs specialized estimators.
    #    References: Athey et al. (GRF extensions), literature on causal survival analysis.


class TreatmentType(Enum):
    BINARY = auto()
    # -> Broadest support: S/T/X, DR-Learner (AIPW-style), DML/R-Learner, CausalForestDML, OrthoForest.
    #    Key refs: Chernozhukov et al. (2018, DML), Nie & Wager (R-Learner), Foster & Syrgkanis (DR-Learner),
    #    Wager & Athey (Causal Forest / GRF). Libraries: EconML (PyWhy), grf (R).

    MULTI_CLASS = auto()
    # -> Requires generalized propensity (multinomial); use DR/Causal Forest variants that handle K>2.
    #    See: Athey et al. (2019, GRF multi-valued), EconML: ForestDRLearner / CausalForestDML (multi-arm support).

    CONTINUOUS = auto()
    # -> Use continuous-treatment versions of DML/OrthoForest/GRF; generalized propensity densities.
    #    See: Oprescu et al. (Orthogonal Random Forests), Athey et al. (GRF).
    #    Libraries: EconML (DMLOrthoForest), grf (R) for continuous treatment.


class FeatureRole(Enum):
    EFFECT_MODIFIER = auto()
    # X: allowed to drive heterogeneity -> feed into final CATE model.
    #    Needed to estimate τ(x).  Rationale: heterogeneity learning in R-/DR- and forest learners.

    CONTROL = auto()
    # W: adjust confounding only -> used in nuisance models (m(x,w), e(x,w)),
    #    but *not* in the final heterogeneity surface if policy requires.  See DML/DR design.

    TREATMENT = auto()
    # T: treatment column(s). Its type (binary/multi/continuous) gates estimator families.

    OUTCOME = auto()
    # Y: determines loss family (regression/classification/survival) and ATE estimator (AIPW/TMLE).

    IDENTIFIER = auto()
    # IDs: for grouping/leakage prevention; excluded from learning.

    TIMESTAMP = auto()
    # Times: enforce temporal ordering; prevent post-treatment leakage. Important for EHR pipelines.

    EXCLUDED = auto()
    # Explicitly dropped (PII/leaky/forbidden).  Not used by any learner.


@dataclass(frozen=True)
class TimeSpec:
    """
    Timing info enforces causal ordering:
      - drop post-treatment features,
      - define outcome windows,
      - trigger survival variants when appropriate.
    In hospital/EHR settings, this is critical to maintain consistency/SUTVA in practice.
    """

    index_time_col: str | None = None  # e.g., "admit_time" (anchors windows)
    treatment_time_col: str | None = None  # when T is assigned; needed for leakage checks
    outcome_window: tuple[str, str] | None = None  # e.g., ('index+0d', 'index+30d')


@dataclass(frozen=True)
class FeatureInfo:
    """
    Each feature carries role and dtype so the pipeline can:
      - choose encoders/scalers (e.g., one-hot vs ordinal),
      - partition into X (heterogeneity) vs W (controls),
      - exclude illegal features (IDs/timestamps) from learning.

    Literature tie-in:
      - Proper X/W separation is assumed by DML (residual-on-residual) and DR (AIPW) frameworks.
      - See Chernozhukov et al. (2018, DML); Foster & Syrgkanis (2019, DR-Learner).
    """

    name: str
    role: FeatureRole
    dtype: str  # pandas dtype (e.g., 'float64', 'int64', 'category', 'datetime64[ns]')
    description: str | None = None  # human-readable (auditability)
    allowed_values: list[str | int | float] | None = (
        None  # validate categoricals; detect rare levels
    )
    unit: str | None = None  # for clinician-facing plots/interpretation


@dataclass(frozen=True)
class OverlapSummary:
    """
    Positivity snapshot informs learner choice:
      - If extreme propensity tails exist, prefer DML/R-style (natural down-weighting) or
        apply trimming/stabilization for DR/AIPW.
    References: AIPW/DR estimators (Robins), DML orthogonality (Chernozhukov et al.).
    Libraries: DoWhy (AIPW/TMLE), EconML (DRLearner, CausalForestDML).
    """

    propensity_col: str | None = None
    prop_low: float | None = None  # suggested lower trim (e.g., 0.02)
    prop_high: float | None = None  # suggested upper trim (e.g., 0.98)
    tail_fraction: float | None = None  # share outside [prop_low, prop_high]


@dataclass(frozen=True)
class ClassBalance:
    """
    Imbalance guides meta-learner choice and weighting:
      - Strong treated/control imbalance -> X-Learner often preferred (Künzel et al., 2019).
      - For binary outcomes, prevalence affects uplift metric stability (Qini/AUUC).
    Libraries: CausalML (X-Learner, uplift metrics), EconML (XLearner).
    """

    treatment_probs: dict[int | str, float] | None = None  # P(T=a)
    outcome_positive_rate: float | None = None  # for binary Y


@dataclass(frozen=True)
class MissingnessSummary:
    """
    Missingness drives preprocessing:
      - High missingness -> prefer tree-based models and indicator imputation.
      - Prevent leaky imputations using TimeSpec.
    Practical guidance from tabular ML and causal forests literature (Athey/Wager).
    """
    fraction_by_feature: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True)
class UnconfoundedCausalMetaData:
    """
    Canonical metadata used to programmatically (or via LLM agents) pick estimators & hyperparams.

    Why each field matters (with literature/library pointers):

    - treatment_type / outcome_type:
        Narrow learner families & loss functions.  Binary T -> S/T/X, DML (R-Learner), DR-Learner,
        CausalForestDML; Multi-class -> generalized propensity + DR/forest variants (EconML, grf);
        Continuous -> DMLOrthoForest / GRF continuous (Oprescu et al.; Athey et al.).

    - treatment_cols / outcome_col:
        Define targets for nuisance models m(x,w), e(x,w) and final CATE τ(x).
        Required by DML (Chernozhukov et al., 2018) and DR (AIPW; Robins) pipelines.

    - features / role_index:
        Partition covariates into X (effect modifiers) vs W (controls) per DML/DR design.
        See: Nie & Wager (R-Learner), Foster & Syrgkanis (DR-Learner); libraries EconML/DoWhy.

    - pandas_dtypes:
        Drives encoders/scalers and informs default choices (forests vs GBMs) for tabular causal ML.
        Libraries: scikit-learn, XGBoost/LightGBM, EconML wrappers.

    - time_spec:
        Enforces temporal logic in EHR pipelines to maintain consistency and avoid post-treatment leakage.

    - class_balance / overlap:
        Decide DR vs DML emphasis, trimming/stabilized weighting; prefer X-Learner under heavy imbalance.
        Refs: Künzel et al. (X-Learner); Chernozhukov et al. (DML); Robins (AIPW).

    - missingness:
        Choose robust preprocessing (tree-based models, indicator impute) and avoid leakage.
    """

    dataset_id: str
    n_rows: int
    n_cols: int

    treatment_type: TreatmentType
    outcome_type: OutcomeType

    # Column names
    treatment_cols: list[str]  # usually ['T']; multi-arm -> one categorical col or encoded
    outcome_col: str  # 'Y'

    # Feature catalog
    features: list[FeatureInfo]  # complete schema with roles and dtypes
    role_index: dict[FeatureRole, list[str]]  # quick lookup: role -> column names

    # Dtype map speeds encoder decisions and validation
    pandas_dtypes: dict[str, str]

    # Optional timing and data-quality summaries (steer learner choice & hyperparams)
    time_spec: TimeSpec | None = None
    class_balance: ClassBalance | None = None
    missingness: MissingnessSummary | None = None
    overlap: OverlapSummary | None = None

    # Free-form notes (cohort definition, exclusions, ICD code filters, etc.) -> for audit & reproducibility
    notes: str | None = None
