# src/python/workflows/state/model_selection_state.py
from __future__ import annotations

from typing import Any, List, NotRequired, TypedDict


class ModelSelectionState(TypedDict, total=False):
    # LLM#1 output (Prompt 1)
    draft_text: str

    # LLM#2 output (Prompt 2)
    final_json_raw: str
    final_json: dict[str, Any]
    selected_top3: List[str]
    selection_notes: List[str]
    rejected: List[dict[str, Any]]
    unknowns: List[str]

    # LLM#3 output (Prompt 3)
    rationale_text: str

    # --- validation/provenance (requested) ---
    allowed_estimators: NotRequired[List[str]]
    allowed_estimators_map: NotRequired[dict[str, bool]]
    top3_validated: NotRequired[bool]
    top3_invalid: NotRequired[List[str]]

    # Optional debug / provenance
    errors: NotRequired[List[str]]
