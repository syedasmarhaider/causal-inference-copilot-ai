from __future__ import annotations

from typing import Literal

SupportedModelsLiteralType = Literal[
    "econml.dml.LinearDML",
    "econml.dml.SparseLinearDML",
    "econml.dml.KernelDML",
    "econml.dml.CausalForestDML",
    "econml.dr.LinearDRLearner",
    "econml.dr.SparseLinearDRLearner",
    "econml.dr.ForestDRLearner",
]


_DISPLAY_METADATA: dict[str, tuple[str, str, str]] = {
    "econml.dr.LinearDRLearner": (
        "Clinically Transparent Baseline Model",
        "Doubly Robust Linear Model",
        "the confirmed doubly robust linear model",
    ),
    "econml.dr.SparseLinearDRLearner": (
        "High-Dimensional Baseline Model",
        "Sparse Doubly Robust Linear Model",
        "the confirmed sparse doubly robust linear model",
    ),
    "econml.dr.ForestDRLearner": (
        "Flexible Subgroup Effect Model",
        "Doubly Robust Forest Model",
        "the confirmed doubly robust forest model",
    ),
    "econml.dml.LinearDML": (
        "Adjusted Baseline Effect Model",
        "Linear Double Machine Learning Model",
        "the confirmed linear double machine learning model",
    ),
    "econml.dml.SparseLinearDML": (
        "High-Dimensional Adjustment Model",
        "Sparse Linear Double Machine Learning Model",
        "the confirmed sparse linear double machine learning model",
    ),
    "econml.dml.KernelDML": (
        "Smooth Nonlinear Effect Model",
        "Kernel Double Machine Learning Model",
        "the confirmed kernel double machine learning model",
    ),
    "econml.dml.CausalForestDML": (
        "Flexible Heterogeneity Model",
        "Causal Forest Model",
        "the confirmed causal forest model",
    ),
}


def get_model_display_labels(fqcn: str) -> tuple[str, str]:
    display_name, family_label, _ = _DISPLAY_METADATA.get(
        fqcn,
        ("Causal Effect Model", "Supported Model", "the confirmed causal model"),
    )
    return display_name, family_label


def get_model_training_label(fqcn: str) -> str:
    _, _, training_label = _DISPLAY_METADATA.get(
        fqcn,
        ("Causal Effect Model", "Supported Model", "the confirmed causal model"),
    )
    return training_label
 
