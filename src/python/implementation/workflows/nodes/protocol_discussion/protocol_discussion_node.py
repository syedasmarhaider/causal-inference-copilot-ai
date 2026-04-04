from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_update_and_gate_prompt,
    get_protocol_discussion_user_message_prompt,
    get_questions,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)

Gate = Literal["READY", "PENDING", "ABORT"]


class _DiscussionAndGateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    protocol_discussion: str = Field(..., min_length=1)
    readiness: Gate = Field(...)


class ProtocolDiscussionNode(Node):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    def __init__(self, *, llm: LLMService) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_protocol_discussion_get_node_info()

    @staticmethod
    def _fallback_user_message(*, readiness: Gate) -> str:
        if readiness == "READY":
            return "Protocol is complete and confirmed. I will proceed to protocol validation."
        if readiness == "ABORT":
            return (
                "This protocol cannot proceed with the current dataset/question constraints. "
                "Please change the protocol requirements or provide compatible data."
            )
        return (
            "I need a bit more information before proceeding. "
            "Please clarify the remaining treatment/outcome/feasibility details."
        )

    @staticmethod
    def _base_payload(*, questions: Sequence[str], summary_string: str) -> dict[str, Any]:
        return {
            "prev_questions_answers_discussion_state": list(questions),
            "dataset_columns_summary": summary_string,
        }

    @staticmethod
    def _initial_discussion(*, questions: Sequence[str]) -> str:
        return "\n".join(str(question).strip() for question in questions if str(question).strip())

    def _bind_payload_to_dataset(
        self,
        *,
        payload: ProtocolDiscussionPayloadModel,
        deps: ProtocolDiscussionDeps,
        questions: Sequence[str],
        reset_discussion: bool,
    ) -> ProtocolDiscussionPayloadModel:
        updates: dict[str, Any] = {
            "dataset_id": deps.dataset_id,
            "dataset_summary": deps.dataset_summary,
        }
        if reset_discussion:
            updates.update(
                {
                    "discussion": self._initial_discussion(questions=questions),
                    "readiness": "PENDING",
                    "node_message": None,
                    "error_message": None,
                }
            )
        elif payload.dataset_summary is None:
            updates["dataset_summary"] = deps.dataset_summary
        return payload.model_copy(update=updates)

    def _call_update_and_gate(
        self,
        *,
        base_payload: Mapping[str, Any],
        protocol_discussion: str,
        history: Sequence[ChatMessage] | None,
    ) -> _DiscussionAndGateModel:
        payload = dict(base_payload)
        payload["protocol_discussion"] = protocol_discussion

        return self._llm.generate_json(
            schema=_DiscussionAndGateModel,
            system_prompt=get_protocol_discussion_update_and_gate_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.2),
            history=history,
            max_attempts=2,
        )

    def _call_user_message(
        self,
        *,
        base_payload: Mapping[str, Any],
        gate: _DiscussionAndGateModel,
        history: Sequence[ChatMessage] | None,
    ) -> str:
        payload = dict(base_payload)
        payload["readiness"] = gate.readiness
        payload["protocol_discussion"] = gate.protocol_discussion

        try:
            out = self._llm.generate(
                system_prompt=get_protocol_discussion_user_message_prompt(),
                user_prompt=json.dumps(payload, ensure_ascii=False),
                config=LLMConfig(model="pro", temperature=0.6),
                history=history,
            )
            return out.content.strip() if out.content else self._fallback_user_message(readiness=gate.readiness)
        except Exception as e:          
            log.exception("PROTOCOL_DISCUSSION: message generation error detail: %s", safe_err(e))
            return self._fallback_user_message(readiness=gate.readiness)

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, Any],
        messages_history: Sequence[ChatMessage] | None,
        state: State,
    ) -> State:
        _ = user_id
        _ = conversation_id

        data_set_profiling_tool = cast(DatasetProfilingTool, tool_factory.get_tool(DatasetProfilingTool.NAME))
        if not isinstance(state, ProtocolDiscussionState):
            raise TypeError(f"{self.name}: expected ProtocolDiscussionState, got {type(state).__name__}")

        deps = ProtocolDiscussionDeps.from_loaded(previous_state_dependencies)
        questions = get_questions()
        prior_dataset_id = state.payload.dataset_id
        payload = state.payload.model_copy(deep=True)
        dataset_changed = prior_dataset_id != deps.dataset_id
        needs_initialization = not payload.discussion.strip()
        payload = self._bind_payload_to_dataset(
            payload=payload,
            deps=deps,
            questions=questions,
            reset_discussion=(dataset_changed or needs_initialization),
        )

        summary_string = data_set_profiling_tool.dataset_summary_to_json(deps.dataset_summary)
        last_6_messages = messages_history[-6:] if messages_history else None
        base_payload = self._base_payload(questions=questions, summary_string=summary_string)

        try:
            gate = self._call_update_and_gate(
                base_payload=base_payload,
                protocol_discussion=payload.discussion,
                history=last_6_messages,
            )
        except Exception as e:
            new_payload = payload.model_copy(
                update={
                    "error_message": f"Protocol discussion update+gate failed: {safe_err(e)}",
                    "node_message": "Protocol discussion update failed. Retrying...",
                    "readiness": "PENDING",
                }
            )
            return ProtocolDiscussionState(new_payload)

        user_message = self._call_user_message(
            base_payload=base_payload,
            gate=gate,
            history=last_6_messages,
        )

        effective_readiness: Gate = "PENDING" if dataset_changed else gate.readiness
        if dataset_changed and prior_dataset_id is not None:
            user_message = (
                "The active dataset changed, so I reset protocol discussion against the latest data. "
                f"{user_message}"
            )

        payload = payload.model_copy(
            update={
                "discussion": gate.protocol_discussion,
                "readiness": effective_readiness,
                "node_message": user_message,
                "error_message": user_message if effective_readiness == "ABORT" else None,
            }
        )
        return ProtocolDiscussionState(payload)
