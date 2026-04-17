from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import LLMConfig, LLMService
from python.domain.models.models import ChatMessage
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.general_queries.general_queries_prompts import (
    get_general_queries_node_info,
    get_general_queries_system_prompt,
    get_general_queries_user_prompt,
)
from python.implementation.workflows.nodes.general_queries.general_queries_state import (
    GeneralQueriesPayloadModel,
    GeneralQueriesState,
)

log = get_logger(__name__)


class _GeneralQueriesResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class GeneralQueriesNode(Node):
    NAME: ClassVar[str] = "GENERAL_QUERIES"

    def __init__(self, *, llm: LLMService) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_general_queries_node_info()

    def run(self, *, request: NodeRequest) -> NodeExecutionResult:
        orchestrator_state = request.orchestrator_state
        history = request.read_only_messages_history

        # Extract the latest user message as the question
        user_question = ""
        if history:
            for msg in reversed(history):
                if msg.role == "user":
                    user_question = msg.content.strip()
                    break

        if not self._has_loaded_dataset(orchestrator_state):
            assistant_message = (
                "Please upload or select a dataset first to start the workflow. "
                "Once the dataset is loaded, I can help define the causal question "
                "and move through the next stages."
            )
            new_state = GeneralQueriesState(
                GeneralQueriesPayloadModel(assistant_message=assistant_message)
            )
            return NodeExecutionResult(
                new_node_state=new_state,
                new_orchestrator_state=orchestrator_state,
                status="DONE",
                action="NEEDS_INPUT",
                response_messages=[
                    ChatMessage(role="assistant", content=assistant_message)
                ],
            )

        # Build a workflow state summary from orchestration state
        workflow_summary = self._build_workflow_summary(orchestrator_state)

        response = self._llm.generate_json(
            schema=_GeneralQueriesResponseModel,
            system_prompt=get_general_queries_system_prompt(),
            user_prompt=get_general_queries_user_prompt(
                user_question=user_question,
                workflow_state_summary=workflow_summary,
            ),
            config=LLMConfig(model="basic", temperature=0.5),
            history=None,
            max_attempts=2,
        )

        new_state = GeneralQueriesState(
            GeneralQueriesPayloadModel(assistant_message=response.assistant_message)
        )

        return NodeExecutionResult(
            new_node_state=new_state,
            new_orchestrator_state=orchestrator_state,
            status="DONE",
            action="NEEDS_INPUT",
            response_messages=[
                ChatMessage(role="assistant", content=response.assistant_message)
            ],
        )

    # -------------------------------------------------------------------------
    # Workflow state summary builder
    # -------------------------------------------------------------------------

    def _has_loaded_dataset(self, orchestrator_state: Any) -> bool:
        dataset_ids: list[Any] = list(orchestrator_state.get("working_dataset_ids") or [])
        summary = orchestrator_state.get("latest_dataset_summary")
        return bool(dataset_ids) and summary is not None

    def _build_workflow_summary(self, orchestrator_state: Any) -> str:
        sections: list[str] = []

        # Stage 1 — dataset
        dataset_ids: list[Any] = list(orchestrator_state.get("working_dataset_ids") or [])
        summary = orchestrator_state.get("latest_dataset_summary")
        if self._has_loaded_dataset(orchestrator_state):
            summary_json = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else summary
            sections.append(
                f"[DONE] Stage 1 — Dataset loaded.\n"
                f"  Active dataset IDs: {[str(d) for d in dataset_ids]}\n"
                f"  Summary: {json.dumps(summary_json, ensure_ascii=False)}"
            )
        else:
            sections.append("[PENDING] Stage 1 — No dataset loaded yet. Start by uploading or selecting a dataset.")

        # Stage 2 — protocol discussion
        protocol = orchestrator_state.get("protocol_discussion")
        if protocol:
            proto_str = str(protocol)
            sections.append(
                f"[DONE] Stage 2 — Protocol discussion complete.\n"
                f"  Protocol excerpt: {proto_str[:300]}{'...' if len(proto_str) > 300 else ''}"
            )
        else:
            sections.append("[PENDING] Stage 2 — Protocol discussion not started. Define the causal question with the agent.")

        # Stage 3 — data cleaning
        data_cleaned: bool = bool(orchestrator_state.get("data_cleaned") or False)
        if data_cleaned:
            sections.append("[DONE] Stage 3 — Dataset cleaned and preprocessing confirmed.")
        else:
            sections.append("[PENDING] Stage 3 — Data cleaning not complete. Use the data manipulation node.")

        # Stage 4 — causal spec + freeze
        causal_spec = orchestrator_state.get("causal_spec")
        transform_plan = orchestrator_state.get("data_transformation_plan")
        frozen: bool = bool(orchestrator_state.get("working_dataset_frozen") or False)
        if causal_spec and transform_plan and frozen:
            sections.append("[DONE] Stage 4 — Causal specification compiled and dataset frozen.")
        else:
            missing: list[str] = []
            if not causal_spec:
                missing.append("causal spec")
            if not transform_plan:
                missing.append("transformation plan")
            if not frozen:
                missing.append("dataset freeze")
            sections.append(f"[PENDING] Stage 4 — Data compilation incomplete. Missing: {', '.join(missing)}.")

        # Stage 5 — validation
        validation_issues: list[Any] = list(orchestrator_state.get("validation_issues") or [])
        if causal_spec and transform_plan and frozen:
            if validation_issues:
                sections.append(
                    f"[DONE] Stage 5 — Validation ran. {len(validation_issues)} issue(s) found."
                )
            else:
                sections.append("[PENDING] Stage 5 — Validation not yet run.")
        else:
            sections.append("[BLOCKED] Stage 5 — Validation blocked until Stage 4 is complete.")

        # Stage 6 — model selection
        selected_model = orchestrator_state.get("selected_model")
        reasoning = orchestrator_state.get("selection_reasoning")
        if selected_model:
            sections.append(
                f"[DONE] Stage 6 — Model selected: {selected_model}.\n"
                f"  Reasoning: {reasoning or 'N/A'}"
            )
        else:
            sections.append("[PENDING] Stage 6 — Model not yet selected.")

        # Stage 7 — training
        trained_model_id = orchestrator_state.get("trained_model_id")
        training_warnings: list[Any] = list(orchestrator_state.get("training_warnings") or [])
        if trained_model_id:
            warn_txt = f" ({len(training_warnings)} warning(s))" if training_warnings else ""
            sections.append(f"[DONE] Stage 7 — Model trained{warn_txt}. Model ID: {trained_model_id}")
        else:
            sections.append("[PENDING] Stage 7 — Model training not started.")

        return "\n\n".join(sections)


__all__ = ["GeneralQueriesNode"]
