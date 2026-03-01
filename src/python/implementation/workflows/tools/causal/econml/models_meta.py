from __future__ import annotations
from typing_extensions import Literal

SupportedModelsLiteralType = Literal[
    "econml.dml.LinearDML",
    "econml.dml.SparseLinearDML",
    "econml.dml.KernelDML",
    "econml.dml.CausalForestDML",
    "econml.dr.LinearDRLearner",
    "econml.dr.SparseLinearDRLearner",
    "econml.dr.ForestDRLearner",
]
 