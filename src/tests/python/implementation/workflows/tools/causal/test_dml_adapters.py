from __future__ import annotations

import importlib
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
from python.implementation.workflows.tools.causal.inference.econml.dml.kernel_dml import (
    KernelDMLCausalModel,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.linear_dml import (
    LinearDMLCausalModel,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.sparse_linear_dml import (
    SparseLinearDMLCausalModel,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.validate_dml import (
    _BaseValidateDML,
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


def _categorical_profile(
    name: str,
    values: list[str],
    *,
    n_missing: int = 0,
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "object",
        "n_rows": 8,
        "n_missing": n_missing,
        "missing_rate": n_missing / 8,
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
    profiles: list[dict[str, Any]] | None = None,
) -> InferenceReadyCausalSpec:
    default_profiles = [
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("patient_id"),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
        _numeric_profile("income"),
        _numeric_profile("segment_score"),
        _numeric_profile("risk_score"),
    ]
    return InferenceReadyCausalSpec(
        causal_spec=_causal_spec(
            covariates=covariates,
            effect_modifiers=effect_modifiers,
        ),
        transformation_plan=_transform_plan(columns=plan_columns),
        data_summary=_summary_model(*(profiles if profiles is not None else default_profiles)),
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


def _df_with_categorical_effect_modifier() -> pd.DataFrame:
    return pd.DataFrame(
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
            "segment_label": ["A", "B", "A", "B", "A", "B", "A", "B"],
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


class _RecordingFeaturizedEstimator:
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
        model_y: Any = "auto",
        model_t: Any = "auto",
        featurizer: Any = None,
        discrete_outcome: bool = False,
        discrete_treatment: bool = False,
        categories: Any = "auto",
        allow_missing: bool = False,
    ) -> None:
        type(self).last_init_kwargs = {
            "model_y": model_y,
            "model_t": model_t,
            "featurizer": featurizer,
            "discrete_outcome": discrete_outcome,
            "discrete_treatment": discrete_treatment,
            "categories": categories,
            "allow_missing": allow_missing,
        }

    def fit(
        self, Y, T, X=None, W=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
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


class _RecordingKernelEstimator:
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
        model_y: Any = "auto",
        model_t: Any = "auto",
        discrete_outcome: bool = False,
        discrete_treatment: bool = False,
        categories: Any = "auto",
        allow_missing: bool = False,
    ) -> None:
        type(self).last_init_kwargs = {
            "model_y": model_y,
            "model_t": model_t,
            "discrete_outcome": discrete_outcome,
            "discrete_treatment": discrete_treatment,
            "categories": categories,
            "allow_missing": allow_missing,
        }

    def fit(
        self, Y, T, X=None, W=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        type(self).last_fit_payload = {"Y": Y, "T": T, "X": X, "W": W}
        return self

    def ate(
        self, X=None, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return 1.0

    def ate_interval(
        self, X=None, T0=None, T1=None, alpha=0.05
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return (0.8, 1.2)

    def ate_inference(
        self, X=None, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return None

    def effect(
        self, X, T0=None, T1=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        type(self).last_effect_payload = {"X": X, "T0": T0, "T1": T1}
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
class _TestLinearDMLModel(LinearDMLCausalModel):
    ESTIMATOR_CLS: ClassVar[Any] = _RecordingFeaturizedEstimator
    INFO: ClassVar[str] = "test"


@dataclass(frozen=True, slots=True)
class _TestSparseLinearDMLModel(SparseLinearDMLCausalModel):
    ESTIMATOR_CLS: ClassVar[Any] = _RecordingFeaturizedEstimator
    INFO: ClassVar[str] = "test"


@dataclass(frozen=True, slots=True)
class _TestKernelDMLModel(KernelDMLCausalModel):
    ESTIMATOR_CLS: ClassVar[Any] = _RecordingKernelEstimator
    INFO: ClassVar[str] = "test"


class _FakeDRTester:
    def __init__(self, **_: Any) -> None:
        self.dr_val_ = np.asarray([], dtype=float)

    def fit_nuisance(self, *, Xval: Any, **_: Any) -> None:
        self.dr_val_ = np.arange(len(Xval), dtype=float)

    def get_cate_preds(self, **_: Any) -> None:
        return None

    def evaluate_all(self, **_: Any) -> _FakeDRTester:
        return self

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"test_name": ["fake_dr_test"]})


def _fit_command(
    *,
    model_name: str,
    df: pd.DataFrame,
    inference_ready_spec: InferenceReadyCausalSpec,
) -> FitCommand:
    return FitCommand(
        model_name=model_name,
        df=df,
        run_id=uuid4(),
        inference_ready_spec=inference_ready_spec,
        inputs=FitInputs(),
    )


def test_linear_dml_fit_passes_featurizer_and_discrete_treatment() -> None:
    _RecordingFeaturizedEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestLinearDMLModel(models_repo=repo, encoding_util=EncodingUtil())
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
        command=_fit_command(
            model_name="linear_dml",
            df=_df(),
            inference_ready_spec=spec,
        ),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingFeaturizedEstimator.last_init_kwargs is not None
    assert _RecordingFeaturizedEstimator.last_init_kwargs["featurizer"] is not None
    assert _RecordingFeaturizedEstimator.last_init_kwargs["discrete_treatment"] is True


def test_sparse_linear_dml_fit_passes_featurizer() -> None:
    _RecordingFeaturizedEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestSparseLinearDMLModel(models_repo=repo, encoding_util=EncodingUtil())
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
        command=_fit_command(
            model_name="sparse_linear_dml",
            df=_df(),
            inference_ready_spec=spec,
        ),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingFeaturizedEstimator.last_init_kwargs is not None
    assert _RecordingFeaturizedEstimator.last_init_kwargs["featurizer"] is not None


def test_kernel_dml_fit_uses_allow_missing_without_featurizer() -> None:
    _RecordingKernelEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestKernelDMLModel(models_repo=repo, encoding_util=EncodingUtil())
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
        profiles=[
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("patient_id"),
            _numeric_profile("outcome"),
            _numeric_profile("age"),
            _numeric_profile("income", n_missing=1),
            _numeric_profile("segment_score"),
            _numeric_profile("risk_score"),
        ],
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(
            model_name="kernel_dml",
            df=_df(income_missing=True),
            inference_ready_spec=spec,
        ),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingKernelEstimator.last_init_kwargs is not None
    assert _RecordingKernelEstimator.last_init_kwargs["allow_missing"] is True


def test_linear_dml_requests_drop_first_effect_modifier_onehot() -> None:
    _RecordingFeaturizedEstimator.reset()
    _RecordingEncodingUtil.reset()
    repo = _InMemoryModelsRepo()
    model = _TestLinearDMLModel(models_repo=repo, encoding_util=_RecordingEncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_label",
                "role": "effect_modifier",
                "encoding": {"preset": "cat_onehot", "handle_unknown": "ignore"},
            },
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ],
        covariates=["age"],
        effect_modifiers=["segment_label"],
        profiles=[
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("patient_id"),
            _numeric_profile("outcome"),
            _numeric_profile("age"),
            _categorical_profile("segment_label", ["A", "B"]),
        ],
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(
            model_name="linear_dml",
            df=_df_with_categorical_effect_modifier(),
            inference_ready_spec=spec,
        ),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingEncodingUtil.last_drop_first_effect_modifier_onehot is True
    featurizer = _RecordingFeaturizedEstimator.last_init_kwargs["featurizer"]  # type: ignore[index]
    transformed = featurizer.fit_transform(np.asarray([["A"], ["B"]], dtype=object))
    assert transformed.shape == (2, 1)


def test_kernel_dml_does_not_request_drop_first_effect_modifier_onehot() -> None:
    _RecordingKernelEstimator.reset()
    _RecordingEncodingUtil.reset()
    repo = _InMemoryModelsRepo()
    model = _TestKernelDMLModel(models_repo=repo, encoding_util=_RecordingEncodingUtil())
    spec = _inference_ready_spec(
        plan_columns=[
            {
                "column": "segment_score",
                "role": "effect_modifier",
                "encoding": {"preset": "num_standard"},
            },
            {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
        ],
        covariates=["age"],
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(
            model_name="kernel_dml",
            df=_df(),
            inference_ready_spec=spec,
        ),
    )

    assert isinstance(result, FitSuccess)
    assert _RecordingEncodingUtil.last_drop_first_effect_modifier_onehot is False


def test_kernel_dml_rejects_non_numeric_x() -> None:
    _RecordingKernelEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestKernelDMLModel(models_repo=repo, encoding_util=EncodingUtil())
    spec = InferenceReadyCausalSpec(
        causal_spec=_causal_spec(
            covariates=["age"],
            effect_modifiers=["segment_label"],
        ),
        transformation_plan=_transform_plan(
            columns=[
                {
                    "column": "segment_label",
                    "role": "effect_modifier",
                    "encoding": {"preset": "cat_onehot"},
                },
                {"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}},
            ]
        ),
        data_summary=_summary_model(
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("patient_id"),
            _numeric_profile("outcome"),
            _numeric_profile("age"),
            _categorical_profile("segment_label", ["A", "B"]),
        ),
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(
            model_name="kernel_dml",
            df=_df_with_categorical_effect_modifier(),
            inference_ready_spec=spec,
        ),
    )

    assert isinstance(result, CommandFailure)
    assert result.error.code == "OPTIONS_INVALID"
    assert "KernelDML requires numeric X" in result.error.message


def test_linear_dml_fit_rejects_unhandled_missing_effect_modifiers() -> None:
    _RecordingFeaturizedEstimator.reset()
    repo = _InMemoryModelsRepo()
    model = _TestLinearDMLModel(models_repo=repo, encoding_util=EncodingUtil())
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
        profiles=[
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("patient_id"),
            _numeric_profile("outcome"),
            _numeric_profile("age"),
            _numeric_profile("income"),
            _numeric_profile("segment_score", n_missing=1),
            _numeric_profile("risk_score"),
        ],
    )

    result = model.execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=_fit_command(
            model_name="linear_dml",
            df=_df(segment_missing=True),
            inference_ready_spec=spec,
        ),
    )

    assert isinstance(result, CommandFailure)
    assert result.error.code == "OPTIONS_INVALID"
    assert "does not support missing values in X via allow_missing" in result.error.message


def test_linear_dml_cate_uses_transformation_plan_order() -> None:
    _RecordingFeaturizedEstimator.reset()
    user_id = uuid4()
    conversation_id = uuid4()
    fitted_model_id = uuid4()
    repo = _InMemoryModelsRepo()
    model = _TestLinearDMLModel(models_repo=repo, encoding_util=EncodingUtil())
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
        model=_RecordingFeaturizedEstimator(),
        metadata={},
    )

    result = model.execute(
        user_id=user_id,
        conversation_id=conversation_id,
        command=CATECommand(
            model_name="linear_dml",
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
    assert _RecordingFeaturizedEstimator.last_effect_payload is not None
    recorded_x = _RecordingFeaturizedEstimator.last_effect_payload["X"]
    assert isinstance(recorded_x, pd.DataFrame)
    assert list(recorded_x.columns) == ["risk_score", "segment_score"]


def test_causal_forest_dml_module_imports_cleanly() -> None:
    importlib.import_module(
        "python.implementation.workflows.tools.causal.inference.econml.dml.causal_forest_dml"
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_validate_dml_returns_one_oof_row_for_every_patient(monkeypatch, n_jobs: int) -> None:
    monkeypatch.setenv("PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE", "2")
    repo = _InMemoryModelsRepo()
    model = _TestLinearDMLModel(models_repo=repo, encoding_util=EncodingUtil())
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
    result = _BaseValidateDML(
        run_dml=model._build_run_dml(),
        n_jobs=n_jobs,
        dr_tester_cls=_FakeDRTester,
    ).execute(
        user_id=uuid4(),
        conversation_id=uuid4(),
        command=ValidateCommand(
            fit_command=_fit_command(
                model_name="linear_dml",
                df=_df(),
                inference_ready_spec=spec,
            )
        ),
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
