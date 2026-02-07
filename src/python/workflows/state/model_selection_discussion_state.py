from __future__ import annotations

from typing import TypedDict


class ModelSelectionDiscussionState(TypedDict, total=False):
    selected_model_fqcn: str
