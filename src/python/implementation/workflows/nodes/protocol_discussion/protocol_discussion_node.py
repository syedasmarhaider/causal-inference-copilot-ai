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
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_summary_blockers import (
    extract_protocol_answer_text,
    scan_protocol_summary_blockers,
    unresolved_summary_blockers,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
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
                } for b in blockers
            ],
            "protocol_discussion": protocol_discussion,
            "dataset_summary": dataset_summary.model_dump_json() if hasattr(dataset_summary, 'model_dump_json') else str(dataset_summary),
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
            str(identifier_column_candidates[0]).strip()
            if identifier_column_candidates
            else None
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
            str(identifier_column_candidates[0]).strip()
            if identifier_column_candidates
            else None
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
            "I drafted the final protocol summary based on the current discussion. "
            f"{preview}"
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
                _ = self._llm_blocker_message(pending_blockers, protocol_discussion, dataset_summary)
                return None
            return draft
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
                    log.exception(
                        "PROTOCOL_DISCUSSION unresolved protocol blockers at confirmation: %s",
                        safe_err(e),
                    )
                    return self._needs_input_result(
                        request=request,
                        payload=payload.model_copy(
                            update={
                                "phase": "DISCUSSING",
                                "pending_dataset_change_request": None,
                                "assistant_message": safe_err(e),
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

        if decision.next_action == "confirm":
            preview_draft = self._compile_preview_causal_spec_draft(
                protocol_discussion=decision.discussion,
                dataset_summary=deps.dataset_summary,
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
                    protocol_discussion=decision.discussion,
                    blockers=blockers,
                )
                if pending_blockers:
                    blocker_message = self._llm_blocker_message(pending_blockers, decision.discussion, deps.dataset_summary)
                    return self._needs_input_result(
                        request=request,
                        payload=payload.model_copy(
                            update={
                                "discussion": decision.discussion,
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
                    protocol_discussion=decision.discussion,
                    dataset_summary_json=summary_string,
                    identifier_column_candidates=identifier_column_candidates,
                )
                review_message = review_summary.assistant_message
            except Exception as e:
                log.exception("PROTOCOL_DISCUSSION review summary failure: %s", safe_err(e))
                review_message = self._fallback_review_summary(
                    decision.discussion,
                    suggested_identifier_column=(
                        identifier_column_candidates[0]
                        if identifier_column_candidates
                        else None
                    ),
                )
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
    answer_text = extract_protocol_answer_text(protocol_discussion, "16)")
    if answer_text is None:
        return None

    normalized = answer_text.strip()
    lowered = normalized.lower()
    if "unclear" in lowered:
        return None
    if "__auto_id__" in lowered:
        return (
            "Identifier handling: no real patient/unit identifier column is being used, "
            "so __auto_id__ will be used."
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
