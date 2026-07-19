from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.implementation.workflows.tools.causal.inference.econml.utils import (
    ModelSpecError,
    get_input_params_from_spec,
    normalize_drtester_cate_predictions,
    normalize_drtester_treatment_codes,
    normalize_drtester_treatment_pair,
    serialize_econml_sensitivity_analysis,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec


def _binary_spec() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "Drug",
                "control": "Placebo",
            },
            "outcome_spec": {
                "kind": "binary",
                "column": "outcome",
                "event": "Yes",
                "non_event": "No",
            },
            "covariates": ["age"],
            "effect_modifiers": [],
            "experiment_type": "OBSERVATIONAL",
            "id_col": "patient_id",
        }
    )


def _continuous_outcome_spec() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "Drug",
                "control": "Placebo",
            },
            "outcome_spec": {
                "kind": "continuous",
                "column": "outcome",
                "unit": "score",
            },
            "covariates": ["age"],
            "effect_modifiers": [],
            "experiment_type": "OBSERVATIONAL",
            "id_col": "patient_id",
        }
    )


def _df(*, outcome: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4],
            "treatment": ["drug", "PLACEBO", "Drug", "placebo"],
            "outcome": outcome,
            "age": [40.0, 55.0, 48.0, 62.0],
        }
    )


def test_get_input_params_maps_binary_treatment_and_outcome_from_spec_labels() -> None:
    y, t, x, w, meta = get_input_params_from_spec(
        _df(outcome=["YES", "no", "Yes", "No"]),
        _binary_spec(),
    )

    np.testing.assert_array_equal(y, np.array([1.0, 0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(t, np.array([1.0, 0.0, 1.0, 0.0]))
    assert x is None
    np.testing.assert_array_equal(w, np.array([[40.0], [55.0], [48.0], [62.0]]))
    assert meta["y"] == "outcome"
    assert meta["t"] == "treatment"


def test_get_input_params_rejects_unmapped_binary_outcome_values() -> None:
    with pytest.raises(ValueError, match="Unmapped binary outcome value 'Maybe'"):
        get_input_params_from_spec(
            _df(outcome=["YES", "no", "Maybe", "No"]),
            _binary_spec(),
        )


def test_get_input_params_does_not_guess_binary_mapping_for_continuous_outcome() -> None:
    with pytest.raises(ValueError):
        get_input_params_from_spec(
            _df(outcome=["YES", "no", "Yes", "No"]),
            _continuous_outcome_spec(),
        )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0, 1, 1, 0], dtype=np.int64),
        ),
        (
            np.array([[0.0], [1.0], [0.0]]),
            np.array([0, 1, 0], dtype=np.int64),
        ),
        (
            np.array([[0.0, 1.0, 0.0]]),
            np.array([0, 1, 0], dtype=np.int64),
        ),
        (
            np.array([2, 0, 1, 2], dtype=np.int16),
            np.array([2, 0, 1, 2], dtype=np.int64),
        ),
        (
            np.array([False, True, False]),
            np.array([0, 1, 0], dtype=np.int64),
        ),
    ],
    ids=["binary-float", "column-vector", "row-vector", "multiclass", "boolean"],
)
def test_normalize_drtester_treatment_codes_returns_contiguous_integer_indices(
    values: np.ndarray,
    expected: np.ndarray,
) -> None:
    result = normalize_drtester_treatment_codes(values)

    np.testing.assert_array_equal(result, expected)
    assert result.ndim == 1
    assert result.dtype == np.dtype(np.int64)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "must not be empty"),
        (np.array(0), "one-dimensional"),
        (np.zeros((2, 2)), "one-dimensional"),
        ([[0, 1], [0]], "rectangular array"),
        (np.array([0]), "at least two treatment groups"),
        (np.array([0.0, 1.5]), "integer-valued codes"),
        (np.array([1.0, 2.0]), "starting at zero"),
        (np.array([0, 2]), "starting at zero"),
        (np.array([-1, 0]), "starting at zero"),
        (np.array([0.0, np.nan]), "must be finite"),
        (np.array([0.0, np.inf]), "must be finite"),
        (np.array([0.0, -np.inf]), "must be finite"),
        (np.array([0 + 1j, 1 + 0j]), "real numeric codes"),
        (np.array(["0", "1"]), "numeric codes"),
        (np.array([None, 1], dtype=object), "numeric codes"),
    ],
    ids=[
        "empty",
        "scalar",
        "matrix",
        "ragged",
        "one-group",
        "fractional",
        "shifted",
        "gapped",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "complex",
        "numeric-strings",
        "object",
    ],
)
def test_normalize_drtester_treatment_codes_rejects_values_drtester_cannot_index(
    values: object,
    message: str,
) -> None:
    with pytest.raises(ModelSpecError, match=message):
        normalize_drtester_treatment_codes(values)


def test_normalize_drtester_treatment_pair_normalizes_both_samples() -> None:
    train, validation = normalize_drtester_treatment_pair(
        train=np.array([0.0, 1.0, 1.0, 0.0]),
        validation=np.array([[1.0], [0.0]]),
    )

    np.testing.assert_array_equal(train, np.array([0, 1, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(validation, np.array([1, 0], dtype=np.int64))


def test_normalize_drtester_treatment_pair_rejects_different_code_sets() -> None:
    with pytest.raises(ModelSpecError, match="must contain the same treatment codes"):
        normalize_drtester_treatment_pair(
            train=np.array([0, 1, 2]),
            validation=np.array([0, 1]),
        )


def test_normalize_drtester_cate_predictions_collapses_singleton_outcome_axis() -> None:
    result = normalize_drtester_cate_predictions(
        np.array([[0.1], [0.2], [0.3]]),
        expected_rows=3,
    )

    np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3]))
    assert result.ndim == 1


@pytest.mark.parametrize(
    ("predictions", "expected_rows", "message"),
    [
        (np.ones((3, 2)), 3, "expected shape"),
        (np.ones((3, 1, 1)), 3, "expected shape"),
        (np.ones(2), 3, "row count"),
        (np.array(1.0), 2, "row count"),
        (np.array(["not-a-number"]), 1, "must be numeric"),
    ],
    ids=[
        "multi-outcome",
        "three-dimensional",
        "wrong-row-count",
        "scalar-many-rows",
        "non-numeric",
    ],
)
def test_normalize_drtester_cate_predictions_rejects_invalid_shapes_and_values(
    predictions: np.ndarray,
    expected_rows: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_drtester_cate_predictions(predictions, expected_rows=expected_rows)


class _Summary:
    def as_text(self) -> str:
        return "summary text"


class _DmlSensitivityEstimator:
    def sensitivity_summary(self, alpha: float = 0.05) -> _Summary:
        _ = alpha
        return _Summary()

    def robustness_value(self, alpha: float = 0.05) -> np.float64:
        _ = alpha
        return np.float64(0.31)

    def sensitivity_interval(self, alpha: float = 0.05) -> tuple[np.float64, np.float64]:
        _ = alpha
        return np.float64(0.1), np.float64(0.7)


class _DrSensitivityEstimator:
    def sensitivity_summary(self, T: float, alpha: float = 0.05) -> str:
        return f"summary for T={T} alpha={alpha}"

    def robustness_value(self, T: float, alpha: float = 0.05) -> float:
        _ = T, alpha
        return 0.42

    def sensitivity_interval(self, T: float, alpha: float = 0.05) -> tuple[float, float]:
        _ = T, alpha
        return 0.2, 0.8


class _NoSensitivityEstimator:
    pass


class _FailingSensitivityEstimator:
    def sensitivity_summary(self) -> str:
        raise RuntimeError("not fitted for sensitivity")


def test_serialize_econml_sensitivity_analysis_handles_dml_style_methods() -> None:
    result, warnings = serialize_econml_sensitivity_analysis(
        _DmlSensitivityEstimator(),
        treatment_value=1.0,
        alpha=0.1,
    )

    assert warnings == []
    assert result == {
        "sensitivity_summary": "summary text",
        "robustness_value": 0.31,
        "sensitivity_interval": [0.1, 0.7],
    }


def test_serialize_econml_sensitivity_analysis_passes_treatment_when_required() -> None:
    result, warnings = serialize_econml_sensitivity_analysis(
        _DrSensitivityEstimator(),
        treatment_value=1.0,
        alpha=0.1,
    )

    assert warnings == []
    assert result == {
        "sensitivity_summary": "summary for T=1.0 alpha=0.1",
        "robustness_value": 0.42,
        "sensitivity_interval": [0.2, 0.8],
    }


def test_serialize_econml_sensitivity_analysis_skips_missing_methods() -> None:
    result, warnings = serialize_econml_sensitivity_analysis(
        _NoSensitivityEstimator(),
        treatment_value=1.0,
        alpha=0.1,
    )

    assert result == {}
    assert warnings == []


def test_serialize_econml_sensitivity_analysis_records_method_failures() -> None:
    result, warnings = serialize_econml_sensitivity_analysis(
        _FailingSensitivityEstimator(),
        treatment_value=1.0,
        alpha=0.1,
    )

    assert result == {"sensitivity_summary": None}
    assert len(warnings) == 1
    assert warnings[0].startswith("SENSITIVITY_NOT_AVAILABLE: sensitivity_summary failed:")
