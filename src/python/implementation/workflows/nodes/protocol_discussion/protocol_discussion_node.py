from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from dataclasses import replace
from typing import Any, ClassVar, Optional, Sequence
from uuid import UUID

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.load_dataset.load_dataset_utils import DatasetStateHelpers
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import ProtocolDiscussionDeps
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import get_protocol_discussion_confirmation_prompt, get_protocol_discussion_readiness_prompt, get_protocol_discussion_system_prompt, get_questions
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState

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
        return "Runs protocol discussion using dataset summary + chat history."
    

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        tool_factory: Optional[ToolFactory],
        previous_state_dependencies: Mapping[str, State],
        user_message: Optional[str],
        router_message: Optional[str],
        mesages_history: Optional[Sequence[ChatMessage]],
        state: State,
    ) -> State:
        if not isinstance(state, ProtocolDiscussionState):
            raise TypeError(f"{self.name}: expected ProtocolDiscussionState, got {type(state).__name__}")

        d = ProtocolDiscussionDeps.from_loaded(previous_state_dependencies)
        summary_state = d.load_dataset.summary
        assert summary_state is not None

        summary_string = DatasetStateHelpers.dataset_summary_to_json(summary_state)
        latest_12_messages = mesages_history[-12:] if mesages_history else None

        payload: dict[str, Any] = {
            "prev_questions_answers_discussion_state": get_questions(),
            "dataset_columns_summary": summary_string,
            "user_message": user_message,
            "router_message": router_message,
        }

        # -------------------------
        # LLM #1: Update discussion
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
            log.exception("PROTOCOL_DISCUSSION: LLM#1 failed")
            return replace(
                state,
                error_message=f"Protocol discussion update failed: {e}",
                node_message="Protocol discussion update failed. Retrying...",
                action="NONE",
                node_status="PENDING",
            )

        state = replace(state, discussion=updated_discussion)

        # -------------------------
        # LLM #2: User-facing message
        # -------------------------
        try:
            node_msg = _llm_call_text(
                llm=self._llm,
                model_name=self._model_name,
                temperature=0.7,
                system_prompt=get_protocol_discussion_confirmation_prompt(),
                user_payload=payload,
                history=latest_12_messages,
                empty_err="LLM#2 returned empty message",
            )
        except Exception as e:
            log.exception("PROTOCOL_DISCUSSION: LLM#2 failed, using fallback")
            return replace(
                state,
                error_message=f"Protocol discussion user facing message failed: {e}",
                node_message="Protocol discussion user facing message failed. Retrying...",
                action="NONE",
                node_status="PENDING",
            )

        # -------------------------
        # LLM #3: Readiness token
        # -------------------------
        try:
            token = _llm_call_text(
                llm=self._llm,
                model_name=self._model_name,
                temperature=0.0,
                system_prompt=get_protocol_discussion_readiness_prompt(),
                user_payload=payload,
                history=latest_12_messages,
                empty_err="LLM#3 returned empty token",
            )
            token = (token or "").strip().splitlines()[0].strip().split()[0].strip().upper()
        except Exception as e:
            log.exception("PROTOCOL_DISCUSSION: LLM#3 failed; defaulting to PENDING")
            return replace(
                state,
                error_message=f"Protocol discussion readiness check failed: {e}",
                node_message="Protocol discussion readiness check failed. Retrying...",
                action="NONE",
                node_status="PENDING",
            )

        if token == "READY":
            return replace(
                state,
                node_message=node_msg,
                error_message=None,
                action="NONE",
                node_status="DONE",
            )

        if token == "ABORT":
            return replace(
                state,
                node_message=f"Protocol discussion aborted: {token}",
                error_message=token,
                action="NONE",
                node_status="ABORTED",
            )
            
        return replace(
            state,
            node_message=node_msg,
            error_message=None,
            action="NEEDS_INPUT",
            node_status="PENDING",
        )
