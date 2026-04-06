from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    confirmed_cleaning_system_message_preamble,
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_update_prompt,
    get_questions,
    initial_user_message,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)

NextAction = Literal["continue", "confirm"]


class _DiscussionDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    discussion: str = Field(..., min_length=1)
    next_action: NextAction
    assistant_message: str = Field(..., min_length=1)
    dataset_change_request: str | None = None

    @model_validator(mode="after")
    def _validate_dataset_change_request(self) -> _DiscussionDecisionModel:
        if self.next_action == "confirm" and not self.dataset_change_request:
            raise ValueError("dataset_change_request is required when next_action=confirm")
        if self.next_action != "confirm" and self.dataset_change_request is not None:
            raise ValueError("dataset_change_request must be null unless next_action=confirm")
        return self


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
    def _base_payload(
        *,
        questions: Sequence[str],
        summary_string: str,
    ) -> dict[str, Any]:
        return {
            "canonical_questions": list(questions),
            "dataset_columns_summary": summary_string,
        }

    @staticmethod
    def _initial_discussion(*, questions: Sequence[str]) -> str:
        return "\n".join(str(question).strip() for question in questions if str(question).strip())

    @staticmethod
    def _prefix_dataset_reset_message(
        *,
        assistant_message: str,
        dataset_changed: bool,
        prior_dataset_id: UUID | None,
    ) -> str:
        if dataset_changed and prior_dataset_id is not None:
            return (
                "The active dataset changed, so I reset protocol discussion against the latest data. "
                f"{assistant_message}"
            )
        return assistant_message

    @staticmethod
    def _build_confirmed_cleaning_system_message(dataset_change_request: str) -> str:
        return (
            f"{confirmed_cleaning_system_message_preamble()}\n\n"
            f"{dataset_change_request.strip()}"
        )

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
                    "phase": "DISCUSSING",
                    "assistant_message": None,
                    "system_message": None,
                }
            )
        elif payload.dataset_summary is None:
            updates["dataset_summary"] = deps.dataset_summary
        return payload.model_copy(update=updates)

    def _call_update(
        self,
        *,
        base_payload: Mapping[str, Any],
        protocol_discussion: str,
        history: Sequence[ChatMessage] | None,
    ) -> _DiscussionDecisionModel:
        payload = dict(base_payload)
        payload["protocol_discussion"] = protocol_discussion
        return self._llm.generate_json(
            schema=_DiscussionDecisionModel,
            system_prompt=get_protocol_discussion_update_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.2),
            history=history,
            max_attempts=2,
        )

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        previous_state_dependencies: Mapping[str, Any],
        messages_history: Sequence[ChatMessage] | None,
        state: State,
    ) -> State:
        _ = user_id
        _ = conversation_id

        if not isinstance(state, ProtocolDiscussionState):
            raise TypeError(
                f"{self.name}: expected ProtocolDiscussionState, got {type(state).__name__}"
            )

        deps = ProtocolDiscussionDeps.from_loaded(previous_state_dependencies)
        questions = get_questions()
        prior_dataset_id = state.payload.dataset_id
        dataset_changed = prior_dataset_id is not None and prior_dataset_id != deps.dataset_id
        needs_initialization = not state.payload.discussion.strip()

        payload = self._bind_payload_to_dataset(
            payload=state.payload.model_copy(deep=True),
            deps=deps,
            questions=questions,
            reset_discussion=(dataset_changed or needs_initialization),
        )

        latest_user_message = _latest_user_message(messages_history)
        if not latest_user_message:
            return ProtocolDiscussionState(
                payload.model_copy(
                    update={
                        "assistant_message": self._prefix_dataset_reset_message(
                            assistant_message=payload.assistant_message or initial_user_message(),
                            dataset_changed=dataset_changed,
                            prior_dataset_id=prior_dataset_id,
                        ),
                        "system_message": None,
                        "phase": "DISCUSSING",
                    }
                )
            )

        summary_string = deps.dataset_summary.model_dump_json()
        base_payload = self._base_payload(questions=questions, summary_string=summary_string)
        last_4_messages = list(messages_history[-4:]) if messages_history else None

        try:
            decision = self._call_update(
                base_payload=base_payload,
                protocol_discussion=payload.discussion,
                history=last_4_messages,
            )
        except Exception as e:
            log.exception("PROTOCOL_DISCUSSION update failure: %s", safe_err(e))
            return ProtocolDiscussionState(
                payload.model_copy(
                    update={
                        "phase": "DISCUSSING",
                        "assistant_message": "Protocol discussion update failed. Please try again.",
                        "system_message": None,
                    }
                )
            )

        assistant_message = self._prefix_dataset_reset_message(
            assistant_message=decision.assistant_message,
            dataset_changed=dataset_changed,
            prior_dataset_id=prior_dataset_id,
        )

        if decision.next_action == "confirm":
            return ProtocolDiscussionState(
                payload.model_copy(
                    update={
                        "discussion": decision.discussion,
                        "phase": "CONFIRMED",
                        "assistant_message": assistant_message,
                        "system_message": self._build_confirmed_cleaning_system_message(
                            cast(str, decision.dataset_change_request)
                        ),
                    }
                )
            )

        return ProtocolDiscussionState(
            payload.model_copy(
                update={
                    "discussion": decision.discussion,
                    "phase": "DISCUSSING",
                    "assistant_message": assistant_message,
                    "system_message": None,
                }
            )
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


__all__ = ["ProtocolDiscussionNode", "_DiscussionDecisionModel"]
