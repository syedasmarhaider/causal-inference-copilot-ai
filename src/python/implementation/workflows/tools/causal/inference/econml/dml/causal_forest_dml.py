from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from econml.dml import CausalForestDML  # pyright: ignore[reportMissingTypeStubs]

from python.implementation.workflows.tools.causal.inference.econml.dml._base_dml import (
    _BaseDMLAdapter,
)
from python.implementation.workflows.tools.causal.inference.econml.models_info import (
    get_causal_forest_dml_causal_model_info,
)


@dataclass(frozen=True, slots=True)
class CausalForestDMLCausalModel(_BaseDMLAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = CausalForestDML
    BACKEND_NAME: ClassVar[str] = "econml.dml.CausalForestDML"
    INFO: ClassVar[str] = get_causal_forest_dml_causal_model_info()
    FIT_INCLUDE_NAMES: ClassVar[set[str]] = {
        "cache_values",
        "inference",
        "sample_weight",
        "groups",
    }
