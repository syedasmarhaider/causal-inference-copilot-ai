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
    get_protocol_discussion_confirmation_prompt,
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_system_prompt,
    get_questions,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetProfilingTool
from python.implementation.workflows.utils.utils import safe_err

log = logging.getLogger(__name__)


def _llm_call_text(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_payload: dict[str, Any],
    empty_err: str,
    history: Optional[Sequence[ChatMessage]] = None,
) -> str:
    cfg = LLMConfig(model=model_name, temperature=temperature)
    resp = llm.generate(
        config=cfg,
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        history=history,
    )
    return resp.content.strip() if resp.content else str(empty_err)


class _MessageAndGateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    readiness: Literal["READY", "PENDING", "ABORT"] = Field(...)
    user_message: str = Field(..., min_length=1)


class ProtocolDiscussionNode(Node):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    def __init__(self, *, llm: LLMService, model_name: str) -> None:
        self._llm = llm
        self._model_name = model_name

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_protocol_discussion_get_node_info()

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
        data_set_profiling_tool = cast(DatasetProfilingTool, tool_factory.get_tool(DatasetProfilingTool.NAME))
        if not isinstance(state, ProtocolDiscussionState):
            raise TypeError(f"{self.name}: expected ProtocolDiscussionState, got {type(state).__name__}")

        d = ProtocolDiscussionDeps.from_loaded(previous_state_dependencies)
        summary_state = d.load_dataset.payload.summary
        assert summary_state is not None

        summary_string = data_set_profiling_tool.dataset_summary_to_json(summary_state)
        latest_12_messages = messages_history[-12:] if messages_history else None

        payload: dict[str, Any] = {
            "prev_questions_answers_discussion_state": get_questions(),
            "dataset_columns_summary": summary_string,
            # include the current discussion doc so LLM#1 can update it
            "protocol_discussion": state.payload.discussion,
        }

        # -------------------------
        # LLM #1: Update discussion (keep separate)
        # -------------------------
        try:
            updated_discussion = _llm_call_text(
                llm=self._llm,
                model_name=self._model_name,
                temperature=0.7,
                system_prompt=get_protocol_discussion_system_prompt(),
                user_payload=payload,
                empty_err="LLM#1 returned empty discussion",
                history=latest_12_messages,
            )
        except Exception as e:
            new_payload = state.payload.model_copy(
                update={
                    "error_message": f"Protocol discussion update failed: {safe_err(e)}",
                    "node_message": "Protocol discussion update failed. Retrying...",
                    "action": "NONE",
                    "node_status": "PENDING",
                }
            )
            return ProtocolDiscussionState(new_payload)

        state.payload.discussion = updated_discussion

        # -------------------------
        # LLM #2+3
        # -------------------------
        try:
            system_prompt = (
                get_protocol_discussion_confirmation_prompt()
            )

            user_payload = { # pyright: ignore[reportUnknownVariableType]
                # pass updated discussion doc + same context
                "protocol_discussion": updated_discussion,
                "prev_questions_answers_discussion_state": get_questions(),
                "dataset_columns_summary": summary_string,
            }

            out = self._llm.generate_json(
                schema=_MessageAndGateModel,
                system_prompt=system_prompt,
                user_prompt=json.dumps(user_payload, ensure_ascii=False),
                config=LLMConfig(model=self._model_name, temperature=0.2),
                history=latest_12_messages,
                max_attempts=2,
            )
        except Exception as e:
            log.exception("PROTOCOL_DISCUSSION: consolidated message+gate failed")
            state.payload.node_message = "Failed to generate user message/readiness. Retrying..."
            state.payload.error_message = f"Protocol discussion message+gate failed: {safe_err(e)}"
            state.payload.action = "NONE"
            state.payload.node_status = "PENDING"
            return state

        token = out.readiness

        if token == "READY":
            state.payload.node_message = out.user_message
            state.payload.error_message = None
            state.payload.action = "NONE"
            state.payload.node_status = "DONE"
            return state

        if token == "ABORT":
            state.payload.node_message = out.user_message
            state.payload.error_message = out.user_message
            state.payload.action = "NONE"
            state.payload.node_status = "ABORTED"
            return state

        state.payload.node_message = out.user_message
        state.payload.error_message = None
        state.payload.action = "NEEDS_INPUT"
        state.payload.node_status = "PENDING"
        return state