from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge

from python.domain.repo.models_repo import ModelRecord
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.encoding.encoding_util import EncodingUtil
from python.implementation.workflows.tools.causal.inference.causal_command import (
    CATECommand,
    CATEInputs,
    CommandFailure,
    FitCommand,
    FitInputs,
    FitSuccess,
    ValidateCommand,
    ValidateSuccess,
)
from python.implementation.workflows.tools.causal.inference.econml.dr import (
    drtester_nuisance_models as dr_nuisance,
)
from python.implementation.workflows.tools.causal.inference.econml.dr import (
    validate_dr as validate_dr_module,
)
from python.implementation.workflows.tools.causal.inference.econml.dr.dr_learner import (
    ForestDRLearnerCausalModel,
    LinearDRLearnerCausalModel,
    _BaseDRLearnerAdapter,
)
from python.implementation.workflows.tools.causal.inference.econml.dr.validate_dr import (
    _add_within_fold_ranking_columns,
    _BaseValidateDR,
    _exclusive_quantile_groups,
    _HeldOutDRResult,
    _ValidationFoldError,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


def _numeric_profile(name: str, *, n_missing: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "float64",
        "n_rows": 8,
        "n_missing": n_missing,
        "missing_rate": n_missing / 8,
        "distinct_count": 8,
        "inferred_kind": "NUMERIC",
        "summary": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1, "quantiles": None},
    }


def _categorical_profile(name: str, values: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "object",
        "n_rows": 8,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": len(values),
        "inferred_kind": "CATEGORICAL",
        "summary": {
            "top_categories": [{"value": value, "count": 4} for value in values],
            "other_count": 0,
        },
    }


def _summary_model(*profiles: dict[str, Any]) -> DatasetSummaryModel:
    return DatasetSummaryModel.model_validate({"n_rows": 8, "profiles": list(profiles)})


def _causal_spec(
    *,
    covariates: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
) -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "drug",
                "control": "placebo",
            },
            "outcome_spec": {
                "kind": "continuous",
                "column": "outcome",
                "unit": "score",
            },
            "covariates": covariates if covariates is not None else ["age", "income"],
            "effect_modifiers": (
                effect_modifiers if effect_modifiers is not None else ["segment_score"]
            ),
            "experiment_type": "OBSERVATIONAL",
            "id_col": "patient_id",
        }
    )


def _transform_plan(*, columns: list[dict[str, Any]]) -> TransformPlan:
    return TransformPlan.model_validate({"columns": columns})


def _inference_ready_spec(
    *,
    covariates: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
    plan_columns: list[dict[str, Any]],
    age_missing: int = 0,
    income_missing: int = 0,
    segment_missing: int = 0,
    risk_missing: int = 0,
) -> InferenceReadyCausalSpec:
    return InferenceReadyCausalSpec(
        causal_spec=_causal_spec(
            covariates=covariates,
            effect_modifiers=effect_modifiers,
        ),
        transformation_plan=_transform_plan(columns=plan_columns),
        data_summary=_summary_model(
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("patient_id"),
            _numeric_profile("outcome"),
            _numeric_profile("age", n_missing=age_missing),
            _numeric_profile("income", n_missing=income_missing),
            _numeric_profile("segment_score", n_missing=segment_missing),
            _categorical_profile("segment_label", ["A", "B"]),
            _numeric_profile("risk_score", n_missing=risk_missing),
        ),
    )


def _df(
    *,
    income_missing: bool = False,
    segment_missing: bool = False,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "treatment": [
                "placebo",
                "drug",
                "placebo",
                "drug",
                "placebo",
                "drug",
                "placebo",
                "drug",
            ],
            "patient_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "outcome": [1.0, 2.2, 1.3, 2.6, 1.1, 2.9, 1.4, 3.0],
            "age": [32.0, 45.0, 37.0, 51.0, 43.0, 39.0, 58.0, 49.0],
            "income": [
                55_000.0,
                63_000.0,
                58_000.0,
                67_000.0,
                61_000.0,
                70_000.0,
                73_000.0,
                69_000.0,
            ],
            "segment_score": [0.1, 0.8, 0.2, 0.7, 0.4, 0.9, 0.3, 0.6],
            "risk_score": [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1],
        }
    )
    if income_missing:
        df.loc[0, "income"] = np.nan
    if segment_missing:
        df.loc[1, "segment_score"] = np.nan
    return df


def _validation_df(row_count: int = 40) -> pd.DataFrame:
    row = np.arange(row_count, dtype=float)
    treatment_binary = np.arange(row_count) % 2
    segment_score = np.linspace(-1.0, 1.0, row_count)
    age = 30.0 + np.mod(row, 35.0)
    income = 50_000.0 + 425.0 * row
    baseline = 0.4 + 0.015 * age + 0.000004 * income + 0.3 * segment_score
    treatment_effect = 0.8 + 0.5 * segment_score
    outcome = baseline + treatment_binary * treatment_effect
    return pd.DataFrame(
        {
            "treatment": np.where(treatment_binary == 1, "drug", "placebo"),
            "patient_id": np.arange(1, row_count + 1),
            "outcome": outcome,
            "age": age,
            "income": income,
            "segment_score": segment_score,
            "risk_score": np.linspace(1.0, 0.0, row_count),
        }
    )


class _InMemoryModelsRepo:
    def __init__(self) -> None:
        self._records: dict[tuple[UUID, UUID, UUID], ModelRecord] = {}

    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._records[(user_id, conversation_id, model_id)] = ModelRecord(
            model_id=model_id,
            model=model,
            metadata=dict(metadata or {}),
        )

    def load_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> ModelRecord | None:
        return self._records.get((user_id, conversation_id, model_id))

    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool:
        return (user_id, conversation_id, model_id) in self._records

    def delete_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> None:
        self._records.pop((user_id, conversation_id, model_id), None)


class _RecordingEncodingUtil(EncodingUtil):
    last_drop_first_effect_modifier_onehot: ClassVar[bool | None] = None

    @classmethod
    def reset(cls) -> None:
        cls.last_drop_first_effect_modifier_onehot = None

    def compile(
        self,
        plan: TransformPlan,
        *,
        effect_modifiers_order,
        covariates_order,
        dense_output: bool = True,
        drop_first_effect_modifier_onehot: bool = False,
    ):
        type(self).last_drop_first_effect_modifier_onehot = drop_first_effect_modifier_onehot
        return super().compile(
            plan,
            effect_modifiers_order=effect_modifiers_order,
            covariates_order=covariates_order,
            dense_output=dense_output,
            drop_first_effect_modifier_onehot=drop_first_effect_modifier_onehot,
        )


class _RecordingEstimator:
    last_init_kwargs: ClassVar[dict[str, Any] | None] = None
    last_fit_payload: ClassVar[dict[str, Any] | None] = None
    last_effect_payload: ClassVar[dict[str, Any] | None] = None

    @classmethod
    def reset(cls) -> None:
        cls.last_init_kwargs = None
        cls.last_fit_payload = None
        cls.last_effect_payload = None

    def __init__(
        self,
        *,
        model_propensity: Any = "auto",
        model_regression: Any = "auto",
        discrete_outcome: bool = False,
        featurizer: Any = None,
        categories: Any = "auto",
        allow_missing: bool = False,
    ) -> None:
        type(self).last_init_kwargs = {
            "model_propensity": model_propensity,
            "model_regression": model_regression,
            "discrete_outcome": discrete_outcome,
            "featurizer": featurizer,
            "categories": categories,
            "allow_missing": allow_missing,
        }
        self.categories = categories

    def fit(
        self, Y, T, X=None, W=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        if self.categories != [0.0, 1.0]:
            raise ValueError("categories must match the encoded treatment values reaching EconML.")
        type(self).last_fit_payload = {"Y": Y, "T": T, "X": X, "W": W}
        return self

    def ate(
        self, X=None, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return 1.25

    def ate_interval(
        self, X=None, T0=None, T1=None, alpha=0.05
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return (1.0, 1.5)

    def ate_inference(
        self, X=None, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return None

    def effect(
        self, X, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        stored_x = X.copy() if isinstance(X, pd.DataFrame) else X
        type(self).last_effect_payload = {"X": stored_x, "T0": T0, "T1": T1}
        rows = int(getattr(X, "shape", [0])[0])
        return np.ones(rows, dtype=float)

    def effect_interval(
        self, X, T0=None, T1=None, alpha=0.05
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        rows = int(getattr(X, "shape", [0])[0])
        return (np.zeros(rows, dtype=float), np.ones(rows, dtype=float))

    def effect_inference(
        self, X, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return None


@dataclass(frozen=True, slots=True)
class _TestDRModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = _RecordingEstimator
    BACKEND_NAME: ClassVar[str] = "test.dr"
    INFO: ClassVar[str] = "test"


@dataclass(frozen=True, slots=True)
class _TestLinearDRModel(LinearDRLearnerCausalModel):
    ESTIMATOR_CLS: ClassVar[Any] = _RecordingEstimator
    INFO: ClassVar[str] = "test"


@dataclass(frozen=True, slots=True)
class _TestForestDRModel(ForestDRLearnerCausalModel):
    ESTIMATOR_CLS: ClassVar[Any] = _RecordingEstimator
    INFO: ClassVar[str] = "test"


class _LightweightValidateDR(_BaseValidateDR):
    def _fit_held_out_dr_scores(
        self,
        *,
        command: FitCommand,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> _HeldOutDRResult:
        _ = (command, train_df)
        treatment = (test_df["treatment"].to_numpy() == "drug").astype(float)
        propensity = np.full(len(test_df), 0.5, dtype=float)
        mu0 = (
            0.4
            + 0.015 * test_df["age"].to_numpy(dtype=float)
            + 0.000004 * test_df["income"].to_numpy(dtype=float)
            + 0.3 * test_df["segment_score"].to_numpy(dtype=float)
        )
        mu1 = mu0 + 0.8 + 0.5 * test_df["segment_score"].to_numpy(dtype=float)
        outcome = test_df["outcome"].to_numpy(dtype=float)
        residual_correction = (
            treatment * (outcome - mu1) / propensity
            - (1.0 - treatment) * (outcome - mu0) / (1.0 - propensity)
        )
        dr_outcome = mu1 - mu0 + residual_correction
        return _HeldOutDRResult(
            dr_outcome=dr_outcome,
            propensity=propensity,
            propensity_used=propensity.copy(),
            propensity_clipped=np.zeros(len(test_df), dtype=bool),
            mu0=mu0,
            mu1=mu1,
            residual_correction=residual_correction,
            treatment_binary=treatment,
            warnings=[],
            diagnostics={
                "outer_fold": outer_fold,
                "train_rows": len(train_df),
                "held_out_rows": len(test_df),
            },
        )


class _ConstantPropensityModel(ClassifierMixin, BaseEstimator):
    def __init__(self, treated_probability: float = 0.4) -> None:
        self.treated_probability = treated_probability

    def fit(self, X: Any, y: Any) -> _ConstantPropensityModel:
        _ = X
        self.classes_ = np.sort(np.unique(np.asarray(y)))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        treated = np.full(len(X), self.treated_probability, dtype=float)
        return np.column_stack([1.0 - treated, treated])


class _MeanOutcomeModel(RegressorMixin, BaseEstimator):
    def fit(self, X: Any, y: Any) -> _MeanOutcomeModel:
        _ = X
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=float)


def _fit_command(*, df: pd.DataFrame, inference_ready_spec: InferenceReadyCausalSpec) -> FitCommand:
    return FitCommand(
        model_name="linear_dr",
        df=df,
        run_id=uuid4(),
        inference_ready_spec=inference_ready_spec,
        inputs=FitInputs(),
    )


def test_fit_passes_numeric_categories_matching_encoded_treatment() -> None:
    _RecordingEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestDRModel(models_repo=repo, encoding_util=EncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "income", "role": "covariate", "encoding": {"preset": "num_standard"}},
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ]
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(df=_df(), inference_ready_spec=spec),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingEstimator.last_init_kwargs is not None
    assert _RecordingEstimator.last_init_kwargs["categories"] == [0.0, 1.0]


def test_fit_sets_allow_missing_for_unhandled_covariate_missingness() -> None:
    _RecordingEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestDRModel(models_repo=repo, encoding_util=EncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "income", "role": "covariate", "encoding": {"preset": "passthrough"}},
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ],
        income_missing=1,
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(df=_df(income_missing=True), inference_ready_spec=spec),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingEstimator.last_init_kwargs is not None
    assert _RecordingEstimator.last_init_kwargs["allow_missing"] is True


def test_linear_dr_requests_drop_first_effect_modifier_onehot() -> None:
    _RecordingEstimator.reset()
    _RecordingEncodingUtil.reset()
    repo = _InMemoryModelsRepo()
    model = _TestLinearDRModel(models_repo=repo, encoding_util=_RecordingEncodingUtil())
    spec = _inference_ready_spec(
        effect_modifiers=["segment_label"],
        covariates=["age"],
        plan_columns=[
            {
                "column": "segment_label",
                "role": "effect_modifier",
                "encoding": {"preset": "cat_onehot", "handle_unknown": "ignore"},
            },
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ],
    )
    df = _df()
    df["segment_label"] = ["A", "B", "A", "B", "A", "B", "A", "B"]

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(df=df, inference_ready_spec=spec),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingEncodingUtil.last_drop_first_effect_modifier_onehot is True
    assert _RecordingEstimator.last_init_kwargs is not None
    featurizer = _RecordingEstimator.last_init_kwargs["featurizer"]
    transformed = featurizer.fit_transform(np.asarray([["A"], ["B"]], dtype=object))
    assert transformed.shape == (2, 1)


def test_forest_dr_does_not_request_drop_first_effect_modifier_onehot() -> None:
    _RecordingEstimator.reset()
    _RecordingEncodingUtil.reset()
    repo = _InMemoryModelsRepo()
    model = _TestForestDRModel(models_repo=repo, encoding_util=_RecordingEncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "income", "role": "covariate", "encoding": {"preset": "num_standard"}},
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ]
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(df=_df(), inference_ready_spec=spec),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingEncodingUtil.last_drop_first_effect_modifier_onehot is False


def test_fit_rejects_unhandled_missing_effect_modifiers_for_dr() -> None:
    _RecordingEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestDRModel(models_repo=repo, encoding_util=EncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "passthrough"},
            },
            {"column": "income", "role": "covariate", "encoding": {"preset": "num_standard"}},
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ],
        segment_missing=1,
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(df=_df(segment_missing=True), inference_ready_spec=spec),
    )

    assert isinstance(result, CommandFailure)
    assert result.error.code == "OPTIONS_INVALID"
    assert "does not support missing values in X via allow_missing" in result.error.message


def test_cate_uses_transformation_plan_order_for_effect_modifiers() -> None:
    _RecordingEstimator.reset()
    user_id = uuid4()
    conversation_id = uuid4()
    fitted_model_id = uuid4()
    repo = _InMemoryModelsRepo()
    model = _TestDRModel(models_repo=repo, encoding_util=EncodingUtil())
    spec = _inference_ready_spec(
        effect_modifiers=["segment_score", "risk_score"],
        covariates=["age"],
        plan_columns=[
            {
                "column": "risk_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ],
    )
    repo.save_model(
        user_id=user_id,
        conversation_id=conversation_id,
        model_id=fitted_model_id,
        model=_RecordingEstimator(categories=[0.0, 1.0]),
        metadata={},
    )

    result = model.execute(
        user_id=user_id,
        conversation_id=conversation_id,
        command=CATECommand(
            model_name="linear_dr",
            df=_df(),
            run_id=uuid4(),
            inference_ready_spec=spec,
            fitted_model_id=fitted_model_id,
            inputs=CATEInputs(
                x_rows=pd.DataFrame(
                    {
                        "risk_score": [0.11, 0.22],
                        "segment_score": [0.33, 0.44],
                    }
                )
            ),
        ),
    )

    assert not isinstance(result, CommandFailure)
    assert result.x_cols == ["risk_score", "segment_score"]
    assert _RecordingEstimator.last_effect_payload is not None
    recorded_x = _RecordingEstimator.last_effect_payload["X"]
    assert isinstance(recorded_x, pd.DataFrame)
    assert list(recorded_x.columns) == ["risk_score", "segment_score"]


def test_validate_dr_returns_one_fold_aware_oof_row_for_every_patient(monkeypatch) -> None:
    monkeypatch.setenv("PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE", "2")
    repo = _InMemoryModelsRepo()
    model = _TestDRModel(models_repo=repo, encoding_util=EncodingUtil())
    validation_df = _validation_df()
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "income", "role": "covariate", "encoding": {"preset": "num_standard"}},
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ]
    )
    result = _LightweightValidateDR(
        run_dr=model._build_run_dr(),
    ).execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=ValidateCommand(
            fit_command=_fit_command(df=validation_df, inference_ready_spec=spec)
        ),
    )

    assert isinstance(result, ValidateSuccess)
    oof = result.validation_dataframe
    assert oof["effect_row"].tolist() == list(range(1, len(validation_df) + 1))
    assert oof["outer_fold"].value_counts().to_dict() == {1: 20, 2: 20}
    assert oof["cate_oof"].tolist() == [1.0] * len(validation_df)
    assert oof["cate_quartile_within_fold"].value_counts().to_dict() == {
        1: 10,
        2: 10,
        3: 10,
        4: 10,
    }
    assert {
        "treatment_oof",
        "mu0_oof",
        "mu1_oof",
        "propensity_oof",
        "propensity_used_oof",
        "propensity_clipped_oof",
        "dr_residual_correction_oof",
        "dr_outcome_oof",
        "cate_percentile_within_fold",
        "cate_quartile_within_fold",
    }.issubset(oof.columns)
    assert np.isfinite(
        oof[
            [
                "cate_oof",
                "mu0_oof",
                "mu1_oof",
                "propensity_oof",
                "dr_outcome_oof",
            ]
        ].to_numpy(dtype=float)
    ).all()
    assert result.dr_test_summary["evaluation_scope"].tolist() == ["fold_aware_oof"]
    assert {
        "blp_est",
        "blp_se",
        "blp_pval",
        "cal_r_squared",
        "qini_est",
        "qini_se",
        "qini_pval",
        "autoc_est",
        "autoc_se",
        "autoc_pval",
    }.issubset(result.dr_test_summary.columns)
    assert result.meta["dr_evaluation_scope"] == "fold_aware_oof"
    assert len(result.meta["gate_summary"]) == 4
    assert sum(group["n"] for group in result.meta["gate_summary"]) == len(validation_df)
    assert len(result.meta["dr_nuisance_diagnostics"]) == 2
    assert repo._records == {}


def test_fit_held_out_dr_scores_uses_only_outer_training_nuisance_models(monkeypatch) -> None:
    repo = _InMemoryModelsRepo()
    model = _TestDRModel(models_repo=repo, encoding_util=EncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "income", "role": "covariate", "encoding": {"preset": "num_standard"}},
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ]
    )
    command = _fit_command(df=_df(), inference_ready_spec=spec)
    monkeypatch.setattr(
        validate_dr_module,
        "get_drtester_models_for_t_and_y",
        lambda *args, **kwargs: (_MeanOutcomeModel(), _ConstantPropensityModel()),
    )

    result = _BaseValidateDR(
        run_dr=model._build_run_dr(),
    )._fit_held_out_dr_scores(
        command=command,
        train_df=_df().iloc[:4].reset_index(drop=True),
        test_df=_df().iloc[4:].reset_index(drop=True),
        outer_fold=1,
    )

    expected_treatment = np.array([0.0, 1.0, 0.0, 1.0])
    expected_mu0 = np.full(4, np.mean([1.0, 1.3]))
    expected_mu1 = np.full(4, np.mean([2.2, 2.6]))
    expected_propensity = np.full(4, 0.4)
    expected_outcome = np.array([1.1, 2.9, 1.4, 3.0])
    expected_residual = (
        expected_treatment * (expected_outcome - expected_mu1) / expected_propensity
        - (1.0 - expected_treatment)
        * (expected_outcome - expected_mu0)
        / (1.0 - expected_propensity)
    )

    np.testing.assert_array_equal(result.treatment_binary, expected_treatment)
    np.testing.assert_allclose(result.propensity, expected_propensity)
    np.testing.assert_allclose(result.mu0, expected_mu0)
    np.testing.assert_allclose(result.mu1, expected_mu1)
    np.testing.assert_allclose(result.residual_correction, expected_residual)
    np.testing.assert_allclose(
        result.dr_outcome,
        expected_mu1 - expected_mu0 + expected_residual,
    )
    assert result.diagnostics["train_rows"] == 4
    assert result.diagnostics["held_out_rows"] == 4


def test_drtester_factory_uses_dr_candidate_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    propensity_candidates = [LogisticRegression(max_iter=100)]
    outcome_candidates = [Ridge()]
    recorded: dict[str, Any] = {}

    def build_propensity(**kwargs: Any) -> list[Any]:
        recorded["propensity"] = kwargs
        return propensity_candidates

    def build_outcome(**kwargs: Any) -> list[Any]:
        recorded["outcome"] = kwargs
        return outcome_candidates

    monkeypatch.setattr(dr_nuisance, "_build_propensity_candidates", build_propensity)
    monkeypatch.setattr(dr_nuisance, "_build_regression_candidates", build_outcome)
    preprocessor = object()

    model_regression, model_propensity = dr_nuisance.get_drtester_models_for_t_and_y(
        _causal_spec(),  # type: ignore[arg-type]
        pre_XW=preprocessor,  # type: ignore[arg-type]
        n_xw=3,
        missingness=False,
        random_state=23,
    )

    assert model_regression.candidates is outcome_candidates  # type: ignore[attr-defined]
    assert len(model_propensity.candidates) == 1  # type: ignore[attr-defined]
    assert model_propensity.candidates[0].model is propensity_candidates[0]  # type: ignore[attr-defined]
    assert recorded["propensity"].pop("pre_XW") is preprocessor
    assert recorded["propensity"] == {
        "missingness_W": False,
        "random_state": 23,
        "n_jobs": 1,
    }
    assert recorded["outcome"].pop("pre_XW") is preprocessor
    assert recorded["outcome"] == {
        "n_xw": 3,
        "discrete_outcome": False,
        "missingness_W": False,
        "random_state": 23,
        "n_jobs": 1,
    }


def test_drtester_factory_probability_scores_binary_outcome_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    propensity_candidate = LogisticRegression(max_iter=100)
    outcome_candidate = LogisticRegression(max_iter=100)

    monkeypatch.setattr(
        dr_nuisance,
        "_build_propensity_candidates",
        lambda **_: [propensity_candidate],
    )
    monkeypatch.setattr(
        dr_nuisance,
        "_build_regression_candidates",
        lambda **_: [outcome_candidate],
    )

    class _BinaryOutcome:
        kind = "binary"

    class _BinarySpec:
        outcome_spec = _BinaryOutcome()

    model_regression, model_propensity = dr_nuisance.get_drtester_models_for_t_and_y(
        _BinarySpec(),  # type: ignore[arg-type]
        pre_XW=object(),  # type: ignore[arg-type]
        n_xw=3,
        missingness=False,
        random_state=31,
    )

    assert model_regression.candidates[0].model is outcome_candidate  # type: ignore[attr-defined]
    assert model_propensity.candidates[0].model is propensity_candidate  # type: ignore[attr-defined]


def test_fold_aware_oof_evaluation_returns_metrics_gates_and_diagnostics() -> None:
    rng = np.random.default_rng(29)
    row_count = 120
    cate = np.linspace(-1.5, 1.5, row_count)
    treatment = np.resize(np.array([0, 1]), row_count)
    validation_dataframe = pd.DataFrame(
        {
            "outer_fold": np.repeat([1, 2], row_count // 2),
            "treatment_oof": treatment,
            "cate_oof": cate,
            "dr_outcome_oof": 0.4 * cate + rng.normal(scale=0.7, size=row_count),
            "propensity_oof": np.full(row_count, 0.5),
        }
    )
    validation_dataframe = _add_within_fold_ranking_columns(
        validation_dataframe,
        n_groups=4,
    )
    validator = _BaseValidateDR(
        run_dr=object(),  # type: ignore[arg-type]
    )

    summary, gate_summary, diagnostics = validator._evaluate_cross_fitted_oof(
        validation_dataframe=validation_dataframe,
        treatment=treatment,
    )

    assert len(summary) == 1
    assert summary["evaluation_scope"].tolist() == ["fold_aware_oof"]
    assert {
        "blp_est",
        "blp_se",
        "blp_pval",
        "cal_r_squared",
        "qini_est",
        "qini_se",
        "qini_pval",
        "autoc_est",
        "autoc_se",
        "autoc_pval",
    }.issubset(summary.columns)
    assert [group["quartile"] for group in gate_summary] == ["Q1", "Q2", "Q3", "Q4"]
    assert sum(group["n"] for group in gate_summary) == row_count
    assert diagnostics["blp_covariance"] == "HC3"
    assert diagnostics["blp_fold_fixed_effects"] is True


def test_fold_aware_oof_quartiles_assign_every_row_exactly_once() -> None:
    values = np.ones(121, dtype=float)

    groups = _exclusive_quantile_groups(values, n_groups=4)

    assert groups.tolist() == sorted(groups.tolist())
    assert np.bincount(groups, minlength=4).tolist() == [31, 30, 30, 30]


def test_validate_dr_rejects_wrong_held_out_row_count_and_cleans_temporary_model(
    monkeypatch,
) -> None:
    repo = _InMemoryModelsRepo()
    model = _TestDRModel(models_repo=repo, encoding_util=EncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "income", "role": "covariate", "encoding": {"preset": "num_standard"}},
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ]
    )

    def wrong_row_count(self: _BaseValidateDR, **kwargs: Any) -> _HeldOutDRResult:
        _ = self
        rows = len(kwargs["test_df"]) + 1
        values = np.zeros(rows, dtype=float)
        return _HeldOutDRResult(
            dr_outcome=values,
            propensity=values,
            propensity_used=values,
            propensity_clipped=np.zeros(rows, dtype=bool),
            mu0=values,
            mu1=values,
            residual_correction=values,
            treatment_binary=values,
            warnings=[],
            diagnostics={},
        )

    monkeypatch.setattr(_BaseValidateDR, "_fit_held_out_dr_scores", wrong_row_count)
    validator = _BaseValidateDR(run_dr=model._build_run_dr())
    fit_command = _fit_command(df=_df(), inference_ready_spec=spec)

    with pytest.raises(
        _ValidationFoldError,
        match="Held-out DR construction returned the wrong row count",
    ):
        validator._run_fold(
            user_id=uuid4(),
            conversation_id=uuid4(),
            command=ValidateCommand(fit_command=fit_command),
            outer_fold=1,
            train_indices=np.array([0, 1, 2, 3]),
            test_indices=np.array([4, 5, 6, 7]),
            progress_queue=None,
        )

    assert repo._records == {}
