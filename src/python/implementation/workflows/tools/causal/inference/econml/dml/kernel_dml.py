from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd
from econml.dml import KernelDML

from python.implementation.workflows.tools.causal.inference.econml.dml._base_dml import (
    _BaseDMLAdapter,
    _raise_if_x_not_numeric,
)
from python.implementation.workflows.tools.causal.inference.econml.models_info import (
    get_kernel_dml_causal_model_info,
)


@dataclass(frozen=True, slots=True)
class KernelDMLCausalModel(_BaseDMLAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = KernelDML
    BACKEND_NAME: ClassVar[str] = "econml.dml.KernelDML"
    INFO: ClassVar[str] = get_kernel_dml_causal_model_info()
    FIT_INCLUDE_NAMES: ClassVar[set[str]] = {
        "cache_values",
        "inference",
        "sample_weight",
        "groups",
    }
    USE_PRE_X_AS_FEATURIZER: ClassVar[bool] = False
    REQUIRE_NUMERIC_X: ClassVar[bool] = True

    def _prepare_cate_query(self, X_query_df: pd.DataFrame) -> Any:
        _raise_if_x_not_numeric(X_query_df)
        return X_query_df.to_numpy()
