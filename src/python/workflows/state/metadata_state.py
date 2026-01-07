from __future__ import annotations

from typing import List, Literal, NotRequired, Optional, TypedDict

from python.workflows.utils.types import JSONDict


CovariateStrategy = Literal["USER_LIST", "ALL_EXCEPT_TY", "NONE"]


class ProposedDesign(TypedDict):
    dataset_summary: str
    treatment_candidates: List[str]
    outcome_candidates: List[str]
    controls_candidates: List[str]
    effect_modifier_candidates: List[str]
    effect_examples: List[str]
    questions_for_user: List[str]


class DraftDesign(TypedDict):
    treatment: Optional[str]
    outcome: Optional[str]

    covariate_strategy: Optional[CovariateStrategy]
    covariates: List[str]

    effect_modifiers: List[str]
    causal_question: Optional[str]

    accept: bool


class FinalDesign(TypedDict):
    treatment: str
    outcome: str

    covariate_strategy: CovariateStrategy
    covariates: List[str]

    effect_modifiers: List[str]
    causal_question: Optional[str]

    accepted: bool


class MetadataState(TypedDict):
    """
    Metadata is always shape-complete:
    - draft always exists (so new_state() is valid)
    - proposed_design/final_design can be None depending on stage.
    """
    proposed_design: ProposedDesign | None
    draft: DraftDesign
    final_design: FinalDesign | None

    last_user_msg_idx: int
    canonical_metadata: dict[str, object] | None
    warnings: list[JSONDict]

    # Optional validation fields (owned by VALIDATE stage/node)
    validation_report: NotRequired[dict[str, object] | None]
    validation_passed: NotRequired[bool | None]


def default_draft() -> DraftDesign:
    return {
        "treatment": None,
        "outcome": None,
        "covariate_strategy": None,
        "covariates": [],
        "effect_modifiers": [],
        "causal_question": None,
        "accept": False,
    }


def empty_metadata_state() -> MetadataState:
    return {
        "proposed_design": None,
        "draft": default_draft(),
        "final_design": None,
        "last_user_msg_idx": -1,
        "canonical_metadata": None,
        "warnings": [],
        "validation_report": None,
        "validation_passed": None,
    }
