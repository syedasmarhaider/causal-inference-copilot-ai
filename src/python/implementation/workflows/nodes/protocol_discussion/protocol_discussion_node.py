from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from typing import Any, ClassVar, Optional, Sequence, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import ProtocolDiscussionDeps
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_update_and_gate_prompt,
    get_protocol_discussion_user_message_prompt,
    get_questions,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetProfilingTool
from python.implementation.workflows.utils.utils import safe_err

log = logging.getLogger(__name__)

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

    def _call_update_and_gate(
        self,
        *,
        base_payload: Mapping[str, Any],
        protocol_discussion: str,
        history: Optional[Sequence[ChatMessage]],
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
        history: Optional[Sequence[ChatMessage]],
    ) -> str:
        payload = dict(base_payload)
        payload["readiness"] = gate.readiness
        payload["protocol_discussion"] = gate.protocol_discussion

        try:
            out = self._llm.generate(
                system_prompt=get_protocol_discussion_user_message_prompt(),
                user_prompt=json.dumps(payload, ensure_ascii=False),
                config=LLMConfig(model="basic", temperature=0.6),
                history=history,
            )
            return out.content.strip() if out.content else self._fallback_user_message(readiness=gate.readiness)
        except Exception as e:
            log.exception("PROTOCOL_DISCUSSION: user message generation failed; using fallback")
            log.error("PROTOCOL_DISCUSSION: message generation error detail: %s", safe_err(e))
            return self._fallback_user_message(readiness=gate.readiness)

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, Any],
        messages_history: Optional[Sequence[ChatMessage]],
        state: State,
    ) -> State:
        _ = user_id
        _ = conversation_id

        data_set_profiling_tool = cast(DatasetProfilingTool, tool_factory.get_tool(DatasetProfilingTool.NAME))
        if not isinstance(state, ProtocolDiscussionState):
            raise TypeError(f"{self.name}: expected ProtocolDiscussionState, got {type(state).__name__}")

        deps = ProtocolDiscussionDeps.from_loaded(previous_state_dependencies)
        summary = deps.dataset_summary
        summary_string = data_set_profiling_tool.dataset_summary_to_json(summary)
        questions = get_questions()
        last_6_messages = messages_history[-6:] if messages_history else None
        base_payload = self._base_payload(questions=questions, summary_string=summary_string)

        try:
            gate = self._call_update_and_gate(
                base_payload=base_payload,
                protocol_discussion=state.payload.discussion,
                history=last_6_messages,
            )
        except Exception as e:
            new_payload = state.payload.model_copy(
                update={
                    "error_message": f"Protocol discussion update+gate failed: {safe_err(e)}",
                    "node_message": "Protocol discussion update failed. Retrying...",
                    "readiness": "PENDING",
                }
            )
            return ProtocolDiscussionState(new_payload)

        state.payload.discussion = gate.protocol_discussion
        state.payload.readiness = gate.readiness

        user_message = self._call_user_message(
            base_payload=base_payload,
            gate=gate,
            history=last_6_messages,
        )
        state.payload.node_message = user_message
        state.payload.error_message = user_message if gate.readiness == "ABORT" else None
        state.payload.readiness = gate.readiness
        return ProtocolDiscussionState(state.payload)
