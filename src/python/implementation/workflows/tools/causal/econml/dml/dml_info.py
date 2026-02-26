from __future__ import annotations

from typing import Any, Dict

def get_dml_info() -> Dict[str, Any]:
        return {
            "name": "DML",
            "backend": "econml.dml.DML",
            "supports": ["FIT"],
            "algorithm": {
                "family": "Double Machine Learning (orthogonal / cross-fit residual-on-residual)",
                "stages": [
                    "Stage 1: fit nuisances E[Y|X,W] and E[T|X,W] with cross-fitting",
                    "Stage 2: regress Y_res on T_res * phi(X) with linear model_final (CATE linear-in-features)",
                ],
                "notes": [
                    "model_final must be linear for correctness (EconML requirement).",
                    "Discrete treatment is internally one-hot encoded (baseline/control dropped).",
                ],
            },
            "data_contract": {
                "requires_columns": ["Y", "T", "optional X (effect modifiers)", "optional W (controls/confounders)"],
                "missingness_policy": {
                    "Y/T": "must be non-missing (hard gate; no silent dropping)",
                    "X/W": "must be non-missing unless allow_missing=True AND models handle missing",
                },
                "types": {
                    "Y/T/X/W": "numeric arrays; categorical T allowed if discrete_treatment=True (internally encoded)",
                    "categorical X/W": "must be encoded upstream (sklearn models don’t accept raw strings)",
                },
            },
            "defaults": {
                "model_final": "StatsModelsLinearRegression(fit_intercept=False) if available else LinearRegression(fit_intercept=False)",
                "model_y": "RFClassifier if discrete_outcome else RFRegressor",
                "model_t": "RFClassifier if discrete_treatment else RFRegressor",
                "fit_params": {"cache_values": True, "inference": "auto"},
            },
        }

    