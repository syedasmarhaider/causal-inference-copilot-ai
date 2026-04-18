from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_review_decision_prompt,
    get_protocol_discussion_review_summary_prompt,
    get_protocol_discussion_update_prompt,
    get_questions,
    initial_user_message,
    summarize_upstream_data_prep_decisions,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
    compile_causal_spec_draft_from_discussion,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)

NextAction = Literal["continue", "confirm"]
ReviewAction = Literal["confirm", "revise", "clarify"]


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


class _ReviewSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class _ReviewDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: ReviewAction
    assistant_message: str = Field(..., min_length=1)


class ProtocolDiscussionNode(Node):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    def __init__(self, *, llm: LLMService, data_repo: DataRepo | None = None) -> None:
        self._llm = llm
        self._data_repo = data_repo

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

    def _bind_payload_to_dataset(
        self,
        *,
        payload: ProtocolDiscussionPayloadModel,
        deps: ProtocolDiscussionDeps,
        questions: Sequence[str],
        reset_discussion: bool,
    ) -> ProtocolDiscussionPayloadModel:
        updates: dict[str, Any] = {"dataset_id": deps.dataset_id}
        if reset_discussion:
            updates.update(
                {
                    "discussion": self._initial_discussion(questions=questions),
                    "phase": "DISCUSSING",
                    "pending_dataset_change_request": None,
                    "assistant_message": None,
                }
            )
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
            config=LLMConfig(model="pro", temperature=0.6),
            history=history,
            max_attempts=2,
        )

    def _call_review_summary(
        self,
        *,
        protocol_discussion: str,
        dataset_summary_json: str,
    ) -> _ReviewSummaryModel:
        return self._llm.generate_json(
            schema=_ReviewSummaryModel,
            system_prompt=get_protocol_discussion_review_summary_prompt(),
            user_prompt=json.dumps(
                {
                    "protocol_discussion": protocol_discussion,
                    "dataset_summary": json.loads(dataset_summary_json),
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.6),
            history=None,
            max_attempts=2,
        )

    def _call_review_decision(
        self,
        *,
        protocol_discussion: str,
        review_message: str | None,
        latest_user_message: str,
    ) -> _ReviewDecisionModel:
        return self._llm.generate_json(
            schema=_ReviewDecisionModel,
            system_prompt=get_protocol_discussion_review_decision_prompt(),
            user_prompt=json.dumps(
                {
                    "protocol_discussion": protocol_discussion,
                    "review_message": review_message,
                    "latest_user_message": latest_user_message,
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="mini", temperature=0.6),
            history=None,
            max_attempts=2,
        )

    @staticmethod
    def _fallback_review_summary(protocol_discussion: str) -> str:
        compact_lines = [line.strip() for line in protocol_discussion.splitlines() if line.strip()]
        preview = " ".join(compact_lines[:6])
        if len(preview) > 900:
            preview = preview[:897].rstrip() + "..."
        prep_decisions = summarize_upstream_data_prep_decisions(protocol_discussion)
        summary = (
            "I drafted the final protocol summary based on the current discussion. "
            f"{preview}"
        )
        if prep_decisions:
            summary += f" Before modeling, we will also follow these agreed data-preparation decisions: {prep_decisions}"
        return f"{summary} Please confirm this protocol, or tell me exactly what should change."

    def _compile_confirmed_causal_spec_draft(
        self,
        *,
        request: NodeRequest,
        deps: ProtocolDiscussionDeps,
        protocol_discussion: str,
    ) -> CausalSpecDraft:
        if self._data_repo is None:
            raise RuntimeError(
                "PROTOCOL_DISCUSSION requires data_repo to validate the compiled causal draft"
            )

        validation_df = self._data_repo.get_csv_data(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            dataset_id=deps.dataset_id,
            limit=1,
        )

        retry_feedback: str | None = None
        previous_draft: CausalSpecDraft | None = None
        last_issue_message: str | None = None
        for _ in range(2):
            draft = compile_causal_spec_draft_from_discussion(
                llm=self._llm,
                protocol_discussion=protocol_discussion,
                dataset_summary=deps.dataset_summary,
                retry_feedback=retry_feedback,
                previous_draft=previous_draft,
            )
            validation_issue = draft.validate_against_dataframe(df=validation_df)
            if validation_issue is None:
                return draft

            previous_draft = draft
            last_issue_message = f"{validation_issue.severity}: {validation_issue.message}"
            retry_feedback = (
                "The previous causal draft failed dataset validation. "
                f"Fix this exactly: {last_issue_message}"
            )

        raise ValueError(
            "Could not compile a grounded causal draft from the confirmed protocol. "
            f"Last validation issue: {last_issue_message or 'unknown validation failure'}"
        )

    @staticmethod
    def _causal_draft_compile_failure_message(error_message: str) -> str:
        return (
            "I could not finalize the confirmed protocol into a grounded causal draft after "
            f"two validation attempts against the active dataset. {error_message} "
            "Please revise the treatment, outcome, covariates, or effect modifiers so they "
            "match exact dataset columns."
        )

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, ProtocolDiscussionState):
            raise TypeError(
                f"{self.name}: expected ProtocolDiscussionState, got {type(request.node_state).__name__}"
            )

        deps = ProtocolDiscussionDeps.from_request(request)
        
        questions = get_questions()
        prior_dataset_id = request.node_state.payload.dataset_id
        dataset_changed = prior_dataset_id is not None and prior_dataset_id != deps.dataset_id
        needs_initialization = not request.node_state.payload.discussion.strip()

        payload = self._bind_payload_to_dataset(
            payload=request.node_state.payload.model_copy(deep=True),
            deps=deps,
            questions=questions,
            reset_discussion=(dataset_changed or needs_initialization),
        )

        latest_user_message = _latest_user_message(request.read_only_messages_history)
        if not latest_user_message:
            return self._needs_input_result(
                request=request,
                payload=payload.model_copy(
                    update={
                        "assistant_message": self._prefix_dataset_reset_message(
                            assistant_message=payload.assistant_message or initial_user_message(),
                            dataset_changed=dataset_changed,
                            prior_dataset_id=prior_dataset_id,
                        ),
                        "system_message": None,
                        "phase": "DISCUSSING",
                    }
                ),
            )

        summary_string = deps.dataset_summary.model_dump_json()
        base_payload = self._base_payload(questions=questions, summary_string=summary_string)
        last_4_messages = (
            list(request.read_only_messages_history[-4:])
            if request.read_only_messages_history
            else None
        )

        if payload.phase == "REVIEW_READY":
            try:
                review_decision = self._call_review_decision(
                    protocol_discussion=payload.discussion,
                    review_message=payload.assistant_message,
                    latest_user_message=latest_user_message,
                )
            except Exception as e:
                log.exception("PROTOCOL_DISCUSSION review decision failure: %s", safe_err(e))
                return self._needs_input_result(
                    request=request,
                    payload=payload.model_copy(
                        update={
                            "phase": "REVIEW_READY",
                            "assistant_message": (
                                "I could not interpret your reply to the protocol review. "
                                "Please confirm the protocol explicitly or say what should change."
                            ),
                        }
                    ),
                )

            if review_decision.action == "confirm":
                try:
                    causal_spec_draft = self._compile_confirmed_causal_spec_draft(
                        request=request,
                        deps=deps,
                        protocol_discussion=payload.discussion,
                    )
                except Exception as e:
                    log.exception(
                        "PROTOCOL_DISCUSSION causal draft compile failure: %s", safe_err(e)
                    )
                    return self._needs_input_result(
                        request=request,
                        payload=payload.model_copy(
                            update={
                                "phase": "DISCUSSING",
                                "pending_dataset_change_request": None,
                                "assistant_message": self._causal_draft_compile_failure_message(
                                    safe_err(e)
                                ),
                            }
                        ),
                    )

                confirmed_payload = payload.model_copy(
                    update={
                        "phase": "CONFIRMED",
                        "assistant_message": review_decision.assistant_message,
                    }
                )
                request.orchestrator_state.set(
                    request.node_state.name(),
                    {
                        "protocol_discussion": confirmed_payload.discussion,
                        "protocol_cleaning_instructions": cast(
                            str,
                            confirmed_payload.pending_dataset_change_request,
                        ),
                        "causal_spec_draft": causal_spec_draft,
                    },
                )
                return self._done_result(
                    request=request,
                    payload=confirmed_payload,
                    user_message=review_decision.assistant_message,
                )

            if review_decision.action == "clarify":
                return self._needs_input_result(
                    request=request,
                    payload=payload.model_copy(
                        update={
                            "phase": "REVIEW_READY",
                            "assistant_message": review_decision.assistant_message,
                        }
                    ),
                )

            payload = payload.model_copy(
                update={
                    "phase": "DISCUSSING",
                    "pending_dataset_change_request": None,
                    "assistant_message": None,
                }
            )

        try:
            decision = self._call_update(
                base_payload=base_payload,
                protocol_discussion=payload.discussion,
                history=last_4_messages,
            )
        except Exception as e:
            log.exception("PROTOCOL_DISCUSSION update failure: %s", safe_err(e))
            return self._needs_input_result(
                request=request,
                payload=payload.model_copy(
                    update={
                        "phase": "DISCUSSING",
                        "assistant_message": "Protocol discussion update failed. Please try again.",
                    }
                ),
            )

        assistant_message = self._prefix_dataset_reset_message(
            assistant_message=decision.assistant_message,
            dataset_changed=dataset_changed,
            prior_dataset_id=prior_dataset_id,
        )

        if decision.next_action == "confirm":
            try:
                review_summary = self._call_review_summary(
                    protocol_discussion=decision.discussion,
                    dataset_summary_json=summary_string,
                )
                review_message = review_summary.assistant_message
            except Exception as e:
                log.exception("PROTOCOL_DISCUSSION review summary failure: %s", safe_err(e))
                review_message = self._fallback_review_summary(decision.discussion)
            return self._needs_input_result(
                request=request,
                payload=payload.model_copy(
                    update={
                        "discussion": decision.discussion,
                        "phase": "REVIEW_READY",
                        "pending_dataset_change_request": cast(
                            str, decision.dataset_change_request
                        ),
                        "assistant_message": self._prefix_dataset_reset_message(
                            assistant_message=review_message,
                            dataset_changed=dataset_changed,
                            prior_dataset_id=prior_dataset_id,
                        ),
                    }
                ),
            )

        return self._needs_input_result(
            request=request,
            payload=payload.model_copy(
                update={
                    "discussion": decision.discussion,
                    "phase": "DISCUSSING",
                    "pending_dataset_change_request": None,
                    "assistant_message": assistant_message,
                }
            ),
        )

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: ProtocolDiscussionPayloadModel,
    ) -> NodeExecutionResult:
        messages: list[ChatMessage] = []
        if payload.assistant_message:
            messages.append(ChatMessage(role="assistant", content=payload.assistant_message))
        if not messages:
            messages.append(ChatMessage(role="assistant", content=initial_user_message()))
        return NodeExecutionResult(
            new_node_state=ProtocolDiscussionState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=messages,
        )

    def _needs_data_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ProtocolDiscussionState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _done_result(
        self,
        *,
        request: NodeRequest,
        payload: ProtocolDiscussionPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ProtocolDiscussionState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="DONE",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
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
