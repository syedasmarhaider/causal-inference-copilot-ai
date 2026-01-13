# src/python/workflows/state/metadata_state.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

ConfounderStrategy = Literal["USER_LIST", "ALL_EXCEPT_TY", "NONE"]

MetadataField = Literal[
    "dataset_summary",
    "treatment",
    "outcome",
    "confounder_strategy",
    "confounders",
    "controls",
    "effect_modifiers",
    "causal_question",
]


class MetadataState(TypedDict):
    # Core causal spec
    treatment: str
    outcome: str
    causal_question: str

    # Backdoor adjustment set (confounders)
    confounder_strategy: ConfounderStrategy
    confounders: List[str]

    # Optional extras (still supported)
    controls: List[str]
    effect_modifiers: List[str]

    # Workflow state
    accepted: bool

    # Optional dataset context
    dataset_summary: str

    # UX helpers
    locked_fields: List[MetadataField]
    notes: List[str]
    warnings: List[str]
    provenance: Dict[str, Any]


def empty_metadata() -> MetadataState:
    return {
        "treatment": "",
        "outcome": "",
        "causal_question": "",
        "confounder_strategy": "NONE",
        "confounders": [],
        "controls": [],
        "effect_modifiers": [],
        "accepted": False,
        "dataset_summary": "",
        "locked_fields": [],
        "notes": [],
        "warnings": [],
        "provenance": {},
    }
