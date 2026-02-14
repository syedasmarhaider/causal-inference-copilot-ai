from __future__ import annotations

from typing import TypedDict


from typing import Any, Dict, NotRequired


class ModelParamsFitState(TypedDict, total=False):
    params: Dict[str, Any]        # e.g. {"init": {...}, "fit": {...}, "feature_set_key": "XW"}
    confirmed: NotRequired[bool]  # True only after user explicitly confirms
    model_id: NotRequired[str]   # populated after FIT, for reference in EFFECT/INTERVAL commands


class ModelState(TypedDict, total=False):
    selected_model_fqcn: str
    model_params_fit: ModelParamsFitState | None
