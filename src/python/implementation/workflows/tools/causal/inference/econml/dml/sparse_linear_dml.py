from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from econml.dml import SparseLinearDML


from python.implementation.workflows.tools.causal.inference.econml.dml._base_dml import _BaseDMLAdapter
from python.implementation.workflows.tools.causal.inference.econml.models_info import (
    get_sparse_linear_dml_causal_model_info,
)


@dataclass(frozen=True, slots=True)
class SparseLinearDMLCausalModel(_BaseDMLAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = SparseLinearDML
    BACKEND_NAME: ClassVar[str] = "econml.dml.SparseLinearDML"
    INFO: ClassVar[str] = get_sparse_linear_dml_causal_model_info()
    DROP_FIRST_EFFECT_MODIFIER_ONEHOT: ClassVar[bool] = True
    FIT_INCLUDE_NAMES: ClassVar[set[str]] = {
        "cache_values",
        "inference",
        "sample_weight",
        "freq_weight",
        "sample_var",
        "groups",
    }
