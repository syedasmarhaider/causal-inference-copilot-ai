from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.implementation.workflows.tools.causal.inference.econml.utils import (
    get_input_params_from_spec,
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
