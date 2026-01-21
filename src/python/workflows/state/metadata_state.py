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
    treatment: str
    outcome: str
    causal_question: str
    confounder_strategy: ConfounderStrategy
    confounders: List[str]
    controls: List[str]
    effect_modifiers: List[str]
    accepted: bool
    dataset_summary: str
    locked_fields: List[MetadataField]
    notes: List[str]
    warnings: List[str]
    provenance: Dict[str, Any]


def empty_metadata_state() -> MetadataState:
    return MetadataState(
        treatment="",
        outcome="",
        causal_question="",
        confounder_strategy="USER_LIST",
        confounders=[],
        controls=[],
        effect_modifiers=[],
        accepted=False,
        dataset_summary="",
        locked_fields=[],
        notes=[],
        warnings=[],
        provenance={},
    )


def get_string_metadata_state(metadata: MetadataState | None) -> str:
    if metadata is None:
        return "Metadata: None"
    return (
        f"Treatment: {metadata['treatment']} | "
        f"Outcome: {metadata['outcome']} | "
        f"Causal Question: {metadata['causal_question']} | "
        f"Confounder Strategy: {metadata['confounder_strategy']} | "
        f"Confounders: {', '.join(metadata['confounders'])} | "
        f"Controls: {', '.join(metadata['controls'])} | "
        f"Effect Modifiers: {', '.join(metadata['effect_modifiers'])} | "
        f"Accepted: {metadata['accepted']}"
    )    