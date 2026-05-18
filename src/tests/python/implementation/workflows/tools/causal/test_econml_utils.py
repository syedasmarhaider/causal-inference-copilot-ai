from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.implementation.workflows.tools.causal.inference.econml.utils import (
    get_input_params_from_spec,
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
