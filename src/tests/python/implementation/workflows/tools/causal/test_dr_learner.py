from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest

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
from python.implementation.workflows.tools.causal.inference.econml.dr.dr_learner import (
    ForestDRLearnerCausalModel,
    LinearDRLearnerCausalModel,
    _BaseDRLearnerAdapter,
)
from python.implementation.workflows.tools.causal.inference.econml.dr.validate_dr import (
    _BaseValidateDR,
    _CATEOnCombinedFeatures,
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


class _ColumnVectorEffectEstimator(_RecordingEstimator):
    def effect(
        self, X, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        rows = int(getattr(X, "shape", [0])[0])
        return np.ones((rows, 1), dtype=float)


def test_drtester_cate_adapter_flattens_column_vector_predictions() -> None:
    adapter = _CATEOnCombinedFeatures(
        estimator=_ColumnVectorEffectEstimator(categories=[0.0, 1.0]),
        effect_modifier_columns=["segment_score"],
    )

    result = adapter.effect(
        pd.DataFrame(
            {
                "segment_score": [0.1, 0.2, 0.3],
                "covariate": [10.0, 20.0, 30.0],
            }
        ),
        T0=0,
        T1=1,
    )

    np.testing.assert_array_equal(result, np.ones(3))
    assert result.ndim == 1


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


class _FakeDRTester:
    """Small source-compatible DRTester double that preserves indexing semantics."""

    def __init__(
        self,
        *,
        model_regression: Any,
        model_propensity: Any,
        cate: Any,
        cv: int,
    ) -> None:
        assert model_regression is not None
        assert model_propensity is not None
        assert cv == 5
        self.cate = cate
        self.dr_val_ = np.asarray([], dtype=float)
        self.treatments = np.asarray([], dtype=np.int64)

    def fit_nuisance(
        self,
        *,
        Xval: Any,
        Dval: Any,
        yval: Any,
        Xtrain: Any,
        Dtrain: Any,
        ytrain: Any,
    ) -> None:
        dval = np.asarray(Dval)
        dtrain = np.asarray(Dtrain)
        assert dval.dtype == np.dtype(np.int64)
        assert dtrain.dtype == np.dtype(np.int64)
        assert dval.ndim == dtrain.ndim == 1
        assert len(Xval) == len(dval) == len(yval)
        assert len(Xtrain) == len(dtrain) == len(ytrain)
        assert np.asarray(Xval).shape[1] == np.asarray(Xtrain).shape[1] == 3

        self.treatments = np.sort(np.unique(dval))
        np.testing.assert_array_equal(np.unique(dtrain), self.treatments)

        # EconML's calculate_dr_outcomes performs equivalent direct column
        # indexing. This intentionally raises IndexError if Dval remains float.
        regression_predictions = np.zeros((len(dval), len(self.treatments)))
        regression_predictions[np.arange(len(dval)), dval]
        self.dr_val_ = np.arange(len(Xval), dtype=float).reshape(-1, 1)

    def get_cate_preds(self, *, Xval: Any, Xtrain: Any) -> None:
        base = self.treatments[0]
        validation_predictions = [
            self.cate.effect(X=Xval, T0=base, T1=treatment) for treatment in self.treatments[1:]
        ]
        training_predictions = [
            self.cate.effect(X=Xtrain, T0=base, T1=treatment) for treatment in self.treatments[1:]
        ]
        assert all(len(prediction) == len(Xval) for prediction in validation_predictions)
        assert all(len(prediction) == len(Xtrain) for prediction in training_predictions)

    def evaluate_all(self, *, n_bootstrap: int) -> _FakeDRTester:
        assert n_bootstrap == 1_000
        return self

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"test_name": ["fake_dr_test"]})


class _WrongRowCountDRTester(_FakeDRTester):
    def fit_nuisance(self, **kwargs: Any) -> None:
        super().fit_nuisance(**kwargs)
        self.dr_val_ = np.append(self.dr_val_.reshape(-1), 999.0).reshape(-1, 1)


class _ExplodingDRTester(_FakeDRTester):
    def fit_nuisance(self, **_: Any) -> None:
        raise IndexError("simulated DRTester treatment-index failure")


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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_validate_dr_returns_one_oof_row_for_every_patient(monkeypatch, n_jobs: int) -> None:
    monkeypatch.setenv("PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE", "2")
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
    result = _BaseValidateDR(
        run_dr=model._build_run_dr(),
        n_jobs=n_jobs,
        dr_tester_cls=_FakeDRTester,
    ).execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=ValidateCommand(fit_command=_fit_command(df=_df(), inference_ready_spec=spec)),
    )

    assert isinstance(result, ValidateSuccess)
    assert result.validation_dataframe["effect_row"].tolist() == list(range(1, 9))
    assert result.validation_dataframe["outer_fold"].value_counts().to_dict() == {1: 4, 2: 4}
    assert result.validation_dataframe["cate_oof"].tolist() == [1.0] * 8
    assert sorted(result.validation_dataframe["dr_outcome_oof"].tolist()) == [
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        3.0,
    ]
    assert result.dr_test_summary["outer_fold"].tolist() == [1, 2]
    assert repo._records == {}


@pytest.mark.parametrize(
    ("dr_tester_cls", "exception_message"),
    [
        (_WrongRowCountDRTester, "DRTester returned the wrong held-out row count"),
        (_ExplodingDRTester, "simulated DRTester treatment-index failure"),
    ],
    ids=["wrong-dr-row-count", "drtester-index-error"],
)
def test_validate_dr_converts_drtester_contract_violations_to_failure_and_cleans_models(
    monkeypatch,
    dr_tester_cls: type[_FakeDRTester],
    exception_message: str,
) -> None:
    monkeypatch.setenv("PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE", "2")
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

    result = _BaseValidateDR(
        run_dr=model._build_run_dr(),
        n_jobs=1,
        dr_tester_cls=dr_tester_cls,
    ).execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=ValidateCommand(fit_command=_fit_command(df=_df(), inference_ready_spec=spec)),
    )

    assert isinstance(result, CommandFailure)
    assert result.error.code == "ESTIMATOR_ERROR"
    assert exception_message in result.error.details["exception"]
    assert repo._records == {}
