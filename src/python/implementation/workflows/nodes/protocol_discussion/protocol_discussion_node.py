from __future__ import annotations

import json
import re
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
    get_llm_blocker_message_prompt,
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_review_decision_prompt,
    get_protocol_discussion_review_summary_prompt,
    get_protocol_discussion_update_prompt,
    get_questions,
    initial_user_message,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_summary_blockers import (
    extract_protocol_answer_text,
    scan_protocol_summary_blockers,
    unresolved_summary_blockers,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
    compile_causal_spec_draft_from_discussion,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)

NextAction = Literal["continue", "confirm"]
ReviewAction = Literal["confirm", "revise", "clarify"]

_STRONG_IDENTIFIER_KEYS = frozenset(
    {
        "patientid",
        "subjectid",
        "personid",
        "memberid",
        "recordid",
        "encounterid",
        "visitid",
        "stayid",
        "mrn",
        "empi",
    }
)
_IDENTIFIER_CONTEXT_TOKENS = (
    "patient",
    "subject",
    "person",
    "member",
    "record",
    "encounter",
    "visit",
    "stay",
    "unit",
    "mrn",
    "empi",
)


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


class _ProtocolSummaryBlockersError(ValueError):
    """Raised when deterministic protocol blockers remain unresolved at confirmation time."""


def _protocol_blocker_error_message(error: Exception) -> str:
    message = str(error).strip()
    return message if message else error.__class__.__name__


class ProtocolDiscussionNode(Node):
    def _llm_blocker_message(
        self,
        blockers: Sequence[Any],
        protocol_discussion: str,
        dataset_summary: DatasetSummaryModel,
    ) -> str:
        """Use LLM to generate user-facing message for blockers."""
        prompt = get_llm_blocker_message_prompt()
        user_payload = {
            "blockers": [
                {
                    "column": b.column,
                    "role": b.role,
                    "issue": b.issue,
                    "user_question": b.user_question,
                }
                for b in blockers
            ],
            "protocol_discussion": protocol_discussion,
            "dataset_summary": (
                dataset_summary.model_dump_json()
                if hasattr(dataset_summary, "model_dump_json")
                else str(dataset_summary)
            ),
        }
        return self._generate_string(
            system_prompt=prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.4),
            history=None,
            max_attempts=2,
        )

    def _generate_string(
        self,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None = None,
        max_attempts: int = 2,
    ) -> str:
        """Call LLMService.generate and return the string content."""
        for _ in range(max_attempts):
            response = self._llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=config,
                history=history,
            )
            if response and response.content:
                return response.content.strip()
        raise RuntimeError("LLM did not return a valid string response after retries.")

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
        identifier_column_candidates: Sequence[str],
    ) -> dict[str, Any]:
        suggested_identifier_column = (
            str(identifier_column_candidates[0]).strip() if identifier_column_candidates else None
        )
        return {
            "canonical_questions": list(questions),
            "dataset_columns_summary": summary_string,
            "identifier_column_candidates": [
                str(candidate).strip() for candidate in identifier_column_candidates
            ],
            "suggested_identifier_column": suggested_identifier_column,
        }

    @staticmethod
    def _initial_discussion(*, questions: Sequence[str]) -> str:
        lines: list[str] = []
        for question in questions:
            question_text = str(question).strip()
            if not question_text:
                continue
            lines.append(question_text)
            question_number = question_text.split(")", 1)[0].strip()
            lines.append(f"A{question_number}) UNCLEAR")
        return "\n".join(lines)

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
        identifier_column_candidates: Sequence[str],
    ) -> _ReviewSummaryModel:
        suggested_identifier_column = (
            str(identifier_column_candidates[0]).strip() if identifier_column_candidates else None
        )
        return self._llm.generate_json(
            schema=_ReviewSummaryModel,
            system_prompt=get_protocol_discussion_review_summary_prompt(),
            user_prompt=json.dumps(
                {
                    "protocol_discussion": protocol_discussion,
                    "dataset_summary": json.loads(dataset_summary_json),
                    "suggested_identifier_column": suggested_identifier_column,
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
    def _fallback_review_summary(
        protocol_discussion: str,
        *,
        suggested_identifier_column: str | None = None,
    ) -> str:
        compact_lines = [line.strip() for line in protocol_discussion.splitlines() if line.strip()]
        preview = " ".join(compact_lines[:6])
        if len(preview) > 900:
            preview = preview[:897].rstrip() + "..."
        identifier_choice = summarize_identifier_choice(
            protocol_discussion,
            suggested_identifier_column=suggested_identifier_column,
        )
        prep_decisions = summarize_upstream_data_prep_decisions(protocol_discussion)
        summary = (
            "I drafted the final protocol summary based on the current discussion. " f"{preview}"
        )
        if identifier_choice:
            summary += f" {identifier_choice}"
        if prep_decisions:
            summary += (
                " Before modeling, we will also follow these agreed data-preparation "
                f"decisions: {prep_decisions}"
            )
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

        # Always use LLM to surface validation issues, no heuristics
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
            role_conflict_message = _negative_control_role_conflict_message(
                protocol_discussion=protocol_discussion,
                draft=draft,
                dataset_summary=deps.dataset_summary,
            )
            if role_conflict_message is not None:
                raise _ProtocolSummaryBlockersError(role_conflict_message)

            validation_issue = draft.validate_against_dataframe(df=validation_df)
            if validation_issue is None:
                blockers = scan_protocol_summary_blockers(
                    dataset_summary=deps.dataset_summary,
                    treatment_column=str(draft.treatment_column),
                    outcome_column=str(draft.outcome_column),
                    covariates=[str(c) for c in draft.covariates],
                    effect_modifiers=[str(c) for c in draft.effect_modifiers],
                )
                pending_blockers = unresolved_summary_blockers(
                    protocol_discussion=protocol_discussion,
                    blockers=blockers,
                )
                if pending_blockers:
                    blocker_message = self._llm_blocker_message(
                        pending_blockers,
                        protocol_discussion,
                        deps.dataset_summary,
                    )
                    raise _ProtocolSummaryBlockersError(blocker_message)
                return draft

            previous_draft = draft
            last_issue_message = f"{validation_issue.severity}: {validation_issue.message}"
            retry_feedback = (
                "The previous causal draft failed dataset validation. "
                f"Fix this exactly: {last_issue_message}"
            )

        # If still failing, always surface the last issue to the user for explicit clarification
        raise ValueError(
            f"Causal draft validation failed: {last_issue_message or 'unknown validation failure'}\n"
            "Please clarify or correct the protocol discussion to resolve this issue."
        )

    def _compile_preview_causal_spec_draft(
        self,
        *,
        protocol_discussion: str,
        dataset_summary: DatasetSummaryModel,
    ) -> CausalSpecDraft | None:
        try:
            draft = compile_causal_spec_draft_from_discussion(
                llm=self._llm,
                protocol_discussion=protocol_discussion,
                dataset_summary=dataset_summary,
            )
            role_conflict_message = _negative_control_role_conflict_message(
                protocol_discussion=protocol_discussion,
                draft=draft,
                dataset_summary=dataset_summary,
            )
            if role_conflict_message is not None:
                raise _ProtocolSummaryBlockersError(role_conflict_message)

            blockers = scan_protocol_summary_blockers(
                dataset_summary=dataset_summary,
                treatment_column=str(draft.treatment_column),
                outcome_column=str(draft.outcome_column),
                covariates=[str(c) for c in draft.covariates],
                effect_modifiers=[str(c) for c in draft.effect_modifiers],
            )
            pending_blockers = unresolved_summary_blockers(
                protocol_discussion=protocol_discussion,
                blockers=blockers,
            )
            if pending_blockers:
                # Use LLM to generate the user-facing message for blockers
                _ = self._llm_blocker_message(
                    pending_blockers, protocol_discussion, dataset_summary
                )
                return None
            return draft
        except _ProtocolSummaryBlockersError:
            raise
        except Exception as e:
            log.warning(
                "PROTOCOL_DISCUSSION preview causal draft compile failed before review: %s",
                safe_err(e),
            )
            return None

    @staticmethod
    def _causal_draft_compile_failure_message(error_message: str) -> str:
        return (
            "Causal draft validation failed: "
            f"{error_message}\n"
            "Please clarify or correct the protocol discussion to resolve this issue. "
            "Explicitly state how to handle any missing columns, type mismatches, or ambiguous values."
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
        identifier_column_candidates = _identifier_column_candidates(deps.dataset_summary)
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
        base_payload = self._base_payload(
            questions=questions,
            summary_string=summary_string,
            identifier_column_candidates=identifier_column_candidates,
        )
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
                except _ProtocolSummaryBlockersError as e:
                    log.warning(
                        "PROTOCOL_DISCUSSION unresolved protocol blockers at confirmation: %s",
                        _protocol_blocker_error_message(e),
                    )
                    return self._needs_input_result(
                        request=request,
                        payload=payload.model_copy(
                            update={
                                "phase": "DISCUSSING",
                                "pending_dataset_change_request": None,
                                "assistant_message": _protocol_blocker_error_message(e),
                            }
                        ),
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
        decision_discussion = _discussion_with_confirmed_unknown_category_decision(
            protocol_discussion=decision.discussion,
            previous_assistant_message=payload.assistant_message,
            latest_user_message=latest_user_message,
        )

        if decision.next_action == "confirm":
            try:
                preview_draft = self._compile_preview_causal_spec_draft(
                    protocol_discussion=decision_discussion,
                    dataset_summary=deps.dataset_summary,
                )
            except _ProtocolSummaryBlockersError as e:
                return self._needs_input_result(
                    request=request,
                    payload=payload.model_copy(
                        update={
                            "discussion": decision_discussion,
                            "phase": "DISCUSSING",
                            "pending_dataset_change_request": None,
                            "assistant_message": self._prefix_dataset_reset_message(
                                assistant_message=_protocol_blocker_error_message(e),
                                dataset_changed=dataset_changed,
                                prior_dataset_id=prior_dataset_id,
                            ),
                        }
                    ),
                )
            if preview_draft is not None:
                blockers = scan_protocol_summary_blockers(
                    dataset_summary=deps.dataset_summary,
                    treatment_column=str(preview_draft.treatment_column),
                    outcome_column=str(preview_draft.outcome_column),
                    covariates=[str(column) for column in preview_draft.covariates],
                    effect_modifiers=[str(column) for column in preview_draft.effect_modifiers],
                )
                pending_blockers = unresolved_summary_blockers(
                    protocol_discussion=decision_discussion,
                    blockers=blockers,
                )
                if pending_blockers:
                    blocker_message = self._llm_blocker_message(
                        pending_blockers,
                        decision_discussion,
                        deps.dataset_summary,
                    )
                    return self._needs_input_result(
                        request=request,
                        payload=payload.model_copy(
                            update={
                                "discussion": decision_discussion,
                                "phase": "DISCUSSING",
                                "pending_dataset_change_request": None,
                                "assistant_message": self._prefix_dataset_reset_message(
                                    assistant_message=blocker_message,
                                    dataset_changed=dataset_changed,
                                    prior_dataset_id=prior_dataset_id,
                                ),
                            }
                        ),
                    )
            try:
                review_summary = self._call_review_summary(
                    protocol_discussion=decision_discussion,
                    dataset_summary_json=summary_string,
                    identifier_column_candidates=identifier_column_candidates,
                )
                review_message = review_summary.assistant_message
            except Exception as e:
                log.exception("PROTOCOL_DISCUSSION review summary failure: %s", safe_err(e))
                review_message = self._fallback_review_summary(
                    decision_discussion,
                    suggested_identifier_column=(
                        identifier_column_candidates[0] if identifier_column_candidates else None
                    ),
                )
            return self._needs_input_result(
                request=request,
                payload=payload.model_copy(
                    update={
                        "discussion": decision_discussion,
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
                    "discussion": decision_discussion,
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


def _discussion_with_confirmed_unknown_category_decision(
    *,
    protocol_discussion: str,
    previous_assistant_message: str | None,
    latest_user_message: str,
) -> str:
    if not _is_affirmative_confirmation(latest_user_message):
        return protocol_discussion
    if not _assistant_asked_unknown_baseline_decision(previous_assistant_message):
        return protocol_discussion

    answer_text = extract_protocol_answer_text(protocol_discussion, "15)")
    if (
        answer_text is not None
        and "unknown" in answer_text.lower()
        and "unclear" not in answer_text.lower()
    ):
        return protocol_discussion

    decision = (
        "15) Baseline feature preparation decisions: Keep Unknown and unknown-like "
        "categories as their own category for all selected covariates and effect modifiers."
    )
    stripped = protocol_discussion.rstrip()
    if not stripped:
        return decision
    return f"{stripped}\n{decision}"


def _negative_control_role_conflict_message(
    *,
    protocol_discussion: str,
    draft: CausalSpecDraft,
    dataset_summary: DatasetSummaryModel,
) -> str | None:
    selected_column = _selected_negative_control_column(
        protocol_discussion=protocol_discussion,
        dataset_summary=dataset_summary,
    )
    if selected_column is None:
        return None

    selected_key = _normalize_column_key(selected_column)
    roles = _protected_roles_for_column(draft=draft, column_key=selected_key)
    if roles:
        return _build_negative_control_role_conflict_message(
            column=selected_column,
            roles=roles,
        )

    draft_negative_control = (
        str(draft.negative_control_outcome).strip()
        if draft.negative_control_outcome is not None
        else None
    )
    if (
        draft_negative_control is None
        or _normalize_column_key(draft_negative_control) != selected_key
    ):
        return (
            f"{selected_column} was selected as the negative-control outcome in the "
            "protocol, but I could not preserve it safely in the compiled causal draft. "
            "Please confirm whether to use this column only as the negative-control "
            "outcome, choose a different negative-control outcome, or set the "
            "negative-control outcome to null."
        )

    return None


def _selected_negative_control_column(
    *,
    protocol_discussion: str,
    dataset_summary: DatasetSummaryModel,
) -> str | None:
    answer_text = extract_protocol_answer_text(protocol_discussion, "16)")
    if answer_text is None:
        return None

    lowered = answer_text.lower()
    if "unclear" in lowered or re.search(r"\b(null|none|no negative|not available)\b", lowered):
        return None

    column_names = [
        str(profile.name).strip()
        for profile in dataset_summary.profiles
        if str(profile.name).strip()
    ]
    for column in sorted(column_names, key=len, reverse=True):
        if _answer_mentions_column(answer_text, column):
            return column
    return None


def _answer_mentions_column(answer_text: str, column: str) -> bool:
    escaped = re.escape(column)
    return (
        re.search(
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
            answer_text,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _protected_roles_for_column(*, draft: CausalSpecDraft, column_key: str) -> list[str]:
    roles: list[str] = []
    role_columns: list[tuple[str, str]] = [
        (str(draft.treatment_column).strip(), "treatment"),
        (str(draft.outcome_column).strip(), "primary outcome"),
        (str(draft.id_col).strip(), "identifier"),
        *((str(column).strip(), "covariate") for column in draft.covariates),
        *((str(column).strip(), "effect modifier") for column in draft.effect_modifiers),
    ]
    for column, role in role_columns:
        if _normalize_column_key(column) == column_key and role not in roles:
            roles.append(role)
    return roles


def _build_negative_control_role_conflict_message(*, column: str, roles: Sequence[str]) -> str:
    role_text = _format_role_list(roles)
    return (
        f"{column} is currently selected as the negative-control outcome and also as "
        f"{role_text}. In this workflow, a negative-control outcome must be a separate "
        "outcome-like column and cannot also be used as treatment, the primary outcome, "
        "an identifier, a covariate, or an effect modifier. Please choose one role: keep "
        f"{column} in its existing role and skip or select another negative-control "
        f"outcome, or remove {column} from the other role and use it "
        "only as the negative-control outcome."
    )


def _format_role_list(roles: Sequence[str]) -> str:
    if not roles:
        return "another causal role"
    if len(roles) == 1:
        article = "an" if roles[0][0].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"{article} {roles[0]}"
    if len(roles) == 2:
        return f"{roles[0]} and {roles[1]}"
    return f"{', '.join(roles[:-1])}, and {roles[-1]}"


def _normalize_column_key(column: str) -> str:
    return column.strip().casefold()


def _is_affirmative_confirmation(message: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", message.strip().lower()).strip()
    if not normalized:
        return False
    explicit_phrases = (
        "yes",
        "yes i confirm",
        "yes i confirm that",
        "i confirm",
        "confirmed",
        "that is correct",
        "thats correct",
        "that s correct",
        "correct",
        "approved",
        "ok",
        "okay",
    )
    return normalized in explicit_phrases or normalized.startswith("yes ")


def _assistant_asked_unknown_baseline_decision(message: str | None) -> bool:
    if not message:
        return False
    normalized = message.lower()
    mentions_unknown = "unknown" in normalized or "unknown-like" in normalized
    mentions_baseline_scope = any(
        token in normalized
        for token in (
            "baseline",
            "covariate",
            "covariates",
            "effect modifier",
            "effect modifiers",
        )
    )
    mentions_decision = any(
        token in normalized
        for token in ("distinct category", "own category", "kept", "keep", "merged", "handled")
    )
    return mentions_unknown and mentions_baseline_scope and mentions_decision


def summarize_upstream_data_prep_decisions(protocol_discussion: str) -> str | None:
    items: list[str] = []
    for prefix in ("14)", "15)"):
        answer_text = extract_protocol_answer_text(protocol_discussion, prefix)
        if answer_text is None:
            continue
        normalized = answer_text.strip()
        lowered = normalized.lower()
        if "unclear" in lowered:
            continue
        items.append(normalized)
    if not items:
        return None
    return " ".join(items)


def summarize_identifier_choice(
    protocol_discussion: str,
    *,
    suggested_identifier_column: str | None = None,
) -> str | None:
    answer_text = extract_protocol_answer_text(protocol_discussion, "17)")
    if answer_text is None:
        return None

    normalized = answer_text.strip()
    lowered = normalized.lower()
    if "unclear" in lowered:
        return None
    if "auto_id" in lowered:
        return (
            "Identifier handling: no real patient/unit identifier column is being used, "
            "so auto_id will be used."
        )

    if (
        suggested_identifier_column is not None
        and suggested_identifier_column.strip()
        and suggested_identifier_column.strip().lower() in lowered
    ):
        return (
            f"Identifier handling: the likely identifier column is "
            f"{suggested_identifier_column.strip()}. If you confirm this review, that "
            "identifier choice will be accepted unless you correct it."
        )

    details = normalized.split(":", 1)[1].strip() if ":" in normalized else normalized
    return f"Identifier handling: {details}"


def _identifier_column_candidates(
    dataset_summary: DatasetSummaryModel,
    *,
    limit: int = 3,
) -> list[str]:
    strong_matches: list[str] = []
    contextual_matches: list[str] = []
    generic_matches: list[str] = []
    seen: set[str] = set()

    for profile in dataset_summary.profiles:
        column_name = str(profile.name).strip()
        if not column_name or column_name in seen:
            continue
        seen.add(column_name)

        normalized = _normalized_identifier_name(column_name)
        compact = normalized.replace("_", "")
        if compact in _STRONG_IDENTIFIER_KEYS:
            strong_matches.append(column_name)
            continue

        ends_with_identifier_suffix = normalized.endswith("_id") or compact.endswith("id")
        if not ends_with_identifier_suffix:
            continue

        if any(token in normalized for token in _IDENTIFIER_CONTEXT_TOKENS):
            contextual_matches.append(column_name)
            continue
        generic_matches.append(column_name)

    return (strong_matches + contextual_matches + generic_matches)[:limit]


def _normalized_identifier_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
