
from typing import TypedDict

from typing import Optional, List
from python.workflows.utils.types import JSONDict
class MetadataState(TypedDict, total=False):
    treatment_hint: Optional[str]
    outcome_hint: Optional[str]

    covariate_hint: Optional[str]        # V

    controls_hint: List[str]            # W
    effect_modifiers_hint: List[str]    # X

    proposed_design: Optional[JSONDict]    # LLM proposal
    final_design: Optional[JSONDict]       # after user confirmation

    metadata: Optional[JSONDict]           # canonical MetaData from MCP
    warnings: List[str]
    user_accepts: Optional[bool]