from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from econml.dml import CausalForestDML  # pyright: ignore[reportMissingTypeStubs]

from python.implementation.workflows.tools.causal.inference.econml.dml._base_dml import (
    _BaseDMLAdapter,
    _to_jsonable,
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

    def _extra_fit_artifacts(self, est: Any) -> dict[str, Any]:
        forest_artifacts: dict[str, Any] = {}
        for attr in ("feature_importances_", "ate_", "ate_stderr_"):
            try:
                if hasattr(est, attr):
                    forest_artifacts[attr] = _to_jsonable(np.asarray(getattr(est, attr)))
            except Exception:
                pass
        return forest_artifacts
