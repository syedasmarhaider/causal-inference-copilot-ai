from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

CovariateStrategy = Literal["USER_LIST", "ALL_EXCEPT_TY", "NONE"]

MetadataField = Literal[
    "dataset_summary",
    "treatment",
    "outcome",
    "covariate_strategy",
    "covariates",
    "controls",
    "effect_modifiers",
    "causal_question",
]

class MetadataState(TypedDict):
    treatment: str
    outcome: str
    covariate_strategy: CovariateStrategy
    
    controls: List[str]
    covariates: List[str]

    effect_modifiers: List[str]
    causal_question: str

    accepted: bool
    
    dataset_summary: str
    
    locked_fields: List[MetadataField] 
    notes: List[str]                  
    warnings: List[str]              
    provenance: Dict[str, Any]        

def empty_metadata() -> MetadataState:
    return {
        "treatment": "",
        "outcome": "",
        "covariate_strategy": "NONE",
        "controls": [],
        "covariates": [],
        "effect_modifiers": [],
        "causal_question": "",
        "accepted": False,
        "dataset_summary": "",
        "locked_fields": [],
        "notes": [],
        "warnings": [],
        "provenance": {},
    }