from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.general_queries.general_queries_prompts import (
    get_general_queries_node_info,
    get_general_queries_system_prompt,
    get_general_queries_user_prompt,
)
from python.implementation.workflows.nodes.general_queries.general_queries_state import (
    GeneralQueriesPayloadModel,
    GeneralQueriesState,
)

log = get_app_logger(__name__, component="general_queries_node", log_type="node")


class _GeneralQueriesResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class GeneralQueriesNode(Node):
    NAME: ClassVar[str] = GeneralQueriesState.NAME

    def __init__(self, *, llm: LLMService) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_general_queries_node_info()

    def run(self, *, request: NodeRequest) -> NodeExecutionResult:
        if not isinstance(request.node_state, GeneralQueriesState):
            raise TypeError(
                f"{self.name}: expected GeneralQueriesState, got "
                f"{type(request.node_state).__name__}"
            )

        orchestrator_state = request.orchestrator_state
        user_question = _latest_user_message(request.read_only_messages_history)
        workflow_summary = self._build_workflow_summary(orchestrator_state)

        response = self._llm.generate_json(
            schema=_GeneralQueriesResponseModel,
            system_prompt=get_general_queries_system_prompt(),
            user_prompt=get_general_queries_user_prompt(
                user_question=user_question
                or "The user asked for a workflow-oriented general answer.",
                workflow_state_summary=workflow_summary,
            ),
            config=LLMConfig(model="basic", temperature=0.3),
            history=_recent_history(request.read_only_messages_history),
            max_attempts=2,
        )

        new_state = GeneralQueriesState(
            GeneralQueriesPayloadModel(assistant_message=response.assistant_message)
        )

        return NodeExecutionResult(
            new_node_state=new_state,
            new_orchestrator_state=orchestrator_state,
            status="DONE",
            action="NONE",
            response_messages=[
                ChatMessage(role="assistant", content=response.assistant_message)
            ],
        )

    def _build_workflow_summary(self, orchestrator_state: Any) -> str:
        sections: list[str] = []

        sections.append(self._build_workflow_position_summary(orchestrator_state))
        sections.append(self._build_stage1_summary(orchestrator_state))
        sections.append(self._build_stage2_summary(orchestrator_state))
        sections.append(self._build_stage3_summary(orchestrator_state))
        sections.append(self._build_stage4_summary(orchestrator_state))
        sections.append(self._build_stage5_summary(orchestrator_state))

        return "\n\n".join(section for section in sections if section.strip())

    def _build_workflow_position_summary(self, orchestrator_state: Any) -> str:
        current_node = _safe_call(orchestrator_state, "get_current_node_name")
        if not current_node:
            return "[UNKNOWN] Workflow position could not be derived from orchestrator state."

        companions = _safe_call(
            orchestrator_state,
            "get_current_node_companion_names",
            current_node,
        )
        companion_text = (
            ", ".join(str(name) for name in companions)
            if isinstance(companions, list) and companions
            else "none"
        )
        return (
            "[INFO] Current workflow position.\n"
            f"  Next required node: {current_node}\n"
            f"  Companion nodes available now: {companion_text}"
        )

    def _build_stage1_summary(self, orchestrator_state: Any) -> str:
        dataset_ids = list(orchestrator_state.get("working_dataset_ids") or [])
        summary = orchestrator_state.get("latest_dataset_summary")
        active_dataset_id = orchestrator_state.get("working_dataset_id")

        if not dataset_ids or summary is None:
            return (
                "[PENDING] Stage 1 — Dataset.\n"
                "  No active dataset has been accepted yet. The user needs to upload, "
                "select, or prepare a dataset before the workflow can proceed."
            )

        n_rows = getattr(summary, "n_rows", None)
        profiles = getattr(summary, "profiles", None) or []
        column_names = [
            str(profile.name).strip()
            for profile in profiles
            if str(profile.name).strip()
        ]
        preview_columns = ", ".join(column_names[:8]) if column_names else "none"
        more_suffix = " ..." if len(column_names) > 8 else ""
        row_text = f"{n_rows} row(s)" if isinstance(n_rows, int) else "unknown row count"

        return (
            "[DONE] Stage 1 — Dataset accepted.\n"
            f"  Active dataset ID: {active_dataset_id}\n"
            f"  Dataset history length: {len(dataset_ids)}\n"
            f"  Accepted summary: {row_text}, {len(column_names)} column(s)\n"
            f"  Columns preview: {preview_columns}{more_suffix}"
        )

    def _build_stage2_summary(self, orchestrator_state: Any) -> str:
        protocol_discussion = orchestrator_state.get("protocol_discussion")
        protocol_cleaning_instructions = orchestrator_state.get(
            "protocol_cleaning_instructions"
        )
        causal_spec_draft = orchestrator_state.get("causal_spec_draft")

        if protocol_discussion is None and causal_spec_draft is None:
            return (
                "[PENDING] Stage 2 — Protocol discussion and causal draft.\n"
                "  The causal question has not been confirmed yet."
            )

        if protocol_discussion is None or causal_spec_draft is None:
            missing_parts: list[str] = []
            if protocol_discussion is None:
                missing_parts.append("confirmed protocol discussion")
            if causal_spec_draft is None:
                missing_parts.append("causal draft")
            return (
                "[PENDING] Stage 2 — Protocol discussion and causal draft.\n"
                f"  Partially complete. Missing: {', '.join(missing_parts)}."
            )

        protocol_excerpt = str(protocol_discussion).strip()
        protocol_excerpt = protocol_excerpt[:240] + (
            "..." if len(protocol_excerpt) > 240 else ""
        )
        draft_summary = json.dumps(
            {
                "treatment_column": causal_spec_draft.treatment_column,
                "outcome_column": causal_spec_draft.outcome_column,
                "covariates": list(causal_spec_draft.covariates),
                "effect_modifiers": list(causal_spec_draft.effect_modifiers),
                "has_cleaning_instructions": protocol_cleaning_instructions is not None,
            },
            ensure_ascii=False,
        )
        return (
            "[DONE] Stage 2 — Protocol discussion and causal draft accepted.\n"
            f"  Protocol excerpt: {protocol_excerpt}\n"
            f"  Accepted draft: {draft_summary}"
        )

    def _build_stage3_summary(self, orchestrator_state: Any) -> str:
        causal_spec = orchestrator_state.get("causal_spec")
        transform_plan = orchestrator_state.get("data_transformation_plan")
        working_dataset_frozen = bool(orchestrator_state.get("working_dataset_frozen") or False)
        is_validated = bool(orchestrator_state.get("is_validated") or False)
        validation_issues = list(orchestrator_state.get("validation_issues") or [])

        if (
            causal_spec is not None
            and transform_plan is not None
            and working_dataset_frozen
            and is_validated
        ):
            return (
                "[DONE] Stage 3 — Compilation, transformation planning, and validation accepted.\n"
                f"  Frozen dataset: yes\n"
                f"  Validation issue count: {len(validation_issues)}"
            )

        missing_parts: list[str] = []
        if causal_spec is None:
            missing_parts.append("compiled causal specification")
        if transform_plan is None:
            missing_parts.append("accepted transformation plan")
        if not working_dataset_frozen:
            missing_parts.append("dataset freeze")
        if not is_validated:
            missing_parts.append("accepted validation state")

        return (
            "[PENDING] Stage 3 — Compilation, transformation planning, and validation.\n"
            f"  Not fully accepted yet. Missing: {', '.join(missing_parts)}."
        )

    def _build_stage4_summary(self, orchestrator_state: Any) -> str:
        selected_model = orchestrator_state.get("selected_model")
        selection_reasoning = orchestrator_state.get("selection_reasoning")

        if selected_model is None or selection_reasoning is None:
            return "[PENDING] Stage 4 — Model selection has not been accepted yet."

        return (
            "[DONE] Stage 4 — Model selection accepted.\n"
            f"  Selected model: {selected_model}\n"
            f"  Reasoning: {str(selection_reasoning).strip()}"
        )

    def _build_stage5_summary(self, orchestrator_state: Any) -> str:
        trained_model_id = orchestrator_state.get("trained_model_id")
        training_warnings = list(orchestrator_state.get("training_warnings") or [])

        if trained_model_id is None:
            return "[PENDING] Stage 5 — Model training has not been completed yet."

        warning_text = f"{len(training_warnings)} warning(s)" if training_warnings else "no warnings"
        return (
            "[DONE] Stage 5 — Model training completed.\n"
            f"  Trained model ID: {trained_model_id}\n"
            f"  Training warnings: {warning_text}"
        )


def _latest_user_message(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history:
        return None
    for message in reversed(messages_history):
        if message.role != "user":
            continue
        content = message.content.strip()
        if content:
            return content
    return None


def _recent_history(
    messages_history: Sequence[ChatMessage] | None,
) -> list[ChatMessage] | None:
    if not messages_history:
        return None
    recent = [message for message in messages_history[-6:] if message.content.strip()]
    return recent or None


def _safe_call(target: Any, method_name: str, *args: Any) -> Any:
    method = getattr(target, method_name, None)
    if not callable(method):
        return None
    try:
        return method(*args)
    except Exception as exc:
        log.warning(
            "general queries workflow summary call failed",
            method_name=method_name,
            error=repr(exc),
        )
        return None


__all__ = ["GeneralQueriesNode"]
