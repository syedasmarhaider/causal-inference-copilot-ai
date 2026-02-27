from __future__ import annotations

from typing import Any, Dict


# =============================================================================
# Linear DML Causal Model Info
# =============================================================================

def linear_dml_causal_model_info() -> Dict[str, Any]:
    return {
        "name": "Linear DML Causal Model",
        "description": "Causal model using the Double Machine Learning (DML) estimator from the EconML library with linear nuisance models.",
        "estimator": "econml.dml.DML",
        "nuisance_models": {
            "model_y": "auto (linear if spec indicates, else default)",
            "model_t": "auto (linear if spec indicates, else default)",
            "model_final": "auto (linear if spec indicates, else default)"
        },
        "suitable_for": {
            "treatment_type": ["binary", "continuous"],
            "outcome_type": ["binary", "continuous"],
            "covariates": ["any"]
        },
        "limitations": [
            "Assumes linear relationships for nuisance models when 'auto' is used and spec indicates linearity.",
            "May not perform well with small sample sizes or highly non-linear relationships."
        ],
        "input_options_spec": {
            # This would be a detailed schema of the expected input options
        }
    }

    