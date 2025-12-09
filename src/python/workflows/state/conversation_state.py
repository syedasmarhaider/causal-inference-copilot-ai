from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Annotated
from uuid import UUID

from langgraph.graph.message import add_messages # pyright: ignore[reportMissingTypeStubs]
from langchain_core.messages import BaseMessage


JSONValue = Any
JSONDict = Dict[str, JSONValue]


class ConversationState(TypedDict, total=False):
    # ------------------------------------------------------------------
    # Conversation / control
    # ------------------------------------------------------------------
    # Chat history, accumulated across turns
    messages: Annotated[List[BaseMessage], add_messages]

    conversation_id: str
    # HIGH-LEVEL intent of this run ("full_pipeline", "ATE", "CATE_SUBGROUP", ...)
    analysis_goal: str
    # Whether the planner thinks it needs more clarification from the user
    clarification_needed: bool

    # What kind of interrupt / review we’re currently waiting on
    # e.g. "REVIEW_METADATA", "REVIEW_METADATA_DIAGNOSTICS",
    #      "REVIEW_ESTIMATOR", "REVIEW_FIT", "REVIEW_EFFECT_PLAN"
    interrupt_type: str

    # Last error (tool failure, validation, etc.) – for user-facing explanation
    last_error: Optional[JSONDict]

    # ------------------------------------------------------------------
    # Dataset loading / schema
    # ------------------------------------------------------------------
    # Where to load data from (path or uploaded file reference)
    dataset_path: str
    # Logical id in DatasetRepo (UUID from infra)
    dataset_id: UUID

    # Raw pandas schema summary (cols + dtypes, maybe sample values)
    raw_schema: JSONDict
    # Basic info: n_rows, n_cols, maybe simple stats
    dataset_summary: JSONDict

    # If loading failed
    load_error: Optional[str]

    # Optional hints extracted from the user
    treatment_hint: str
    outcome_hint: str

    # ------------------------------------------------------------------
    # MetaData / causal design
    # ------------------------------------------------------------------
    # First guess from LLM + heuristics
    proposed_metadata_design: JSONDict
    # After user edits / confirmation
    final_metadata_design: JSONDict

    # Canonical MetaData returned by econml MCP
    metadata: JSONDict
    # Warnings from profiling / overlap / class balance
    metadata_warnings: List[str]
    # Did user accept current MetaData + diagnostics?
    user_accepts_metadata: Optional[bool]

    # ------------------------------------------------------------------
    # Estimator selection
    # ------------------------------------------------------------------
    # Full catalog from backdoor_list_estimators()
    estimator_catalog: List[JSONDict]
    # Filtered / scored candidates for this dataset
    candidate_estimators: List[JSONDict]
    # Planner’s recommended default
    default_estimator_id: str
    # Final choice after user review
    selected_estimator_id: str

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    # BackdoorFitResult JSON (includes model_id, cohort stats, etc.)
    fit_result: JSONDict
    # Any warnings from fitting (e.g. weak overlap, convergence issues)
    fit_warnings: List[str]
    # User decision after seeing fit diagnostics
    user_accepts_fit: Optional[bool]

    # Convenience: cached model_id pulled out of fit_result
    model_id: UUID

    # ------------------------------------------------------------------
    # Effect planning & execution
    # ------------------------------------------------------------------
    # Planner’s initial plan: list of effect query specs
    proposed_effect_queries: List[JSONDict]
    # Final, user-confirmed plan
    effect_queries: List[JSONDict]
    # Results for each query from backdoor_effects()
    effect_results: List[JSONDict]

    # ------------------------------------------------------------------
    # Reporting / UX
    # ------------------------------------------------------------------
    # Structured summary you might show in a UI (tables, metrics, etc.)
    final_report: JSONDict
    # Natural-language explanation for the user
    final_report_text: str
