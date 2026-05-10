from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_causal_draft_prompt,
    get_protocol_discussion_compilation_prompt,
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_response_prompt,
    get_protocol_discussion_status_prompt,
    get_protocol_discussion_template,
    get_protocol_discussion_validation_suggestion_prompt,
    initial_user_message,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
    ProtocolDiscussionStatus,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.utils.utils import safe_err

if TYPE_CHECKING:
    from python.domain.repo.data_repo import DataRepo

log = get_logger(__name__)


class ProtocolDiscussionStatusResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: ProtocolDiscussionStatus


@dataclass(frozen=True)
class ProtocolDiscussionCausalDraftResult:
    draft: Any | None
    validation_issues: list[dict[str, Any]]


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

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, ProtocolDiscussionState):
            raise TypeError(
                f"{self.name}: expected ProtocolDiscussionState, "
                f"got {type(request.node_state).__name__}"
            )

        deps = ProtocolDiscussionDeps.from_request(request)
        payload = request.node_state.payload.model_copy(deep=True)
        dataset_changed = payload.dataset_id is not None and payload.dataset_id != deps.dataset_id
        if dataset_changed or payload.dataset_id is None:
            payload = ProtocolDiscussionPayloadModel(dataset_id=deps.dataset_id)

        latest_user_message: str | None = None
        if request.read_only_messages_history:
            for message in reversed(request.read_only_messages_history):
                if message.role == "user" and message.content.strip():
                    latest_user_message = message.content.strip()
                    break

        if not latest_user_message and not payload.protocol_discussion:
            assistant_message = initial_user_message()
            if dataset_changed:
                assistant_message = (
                    "The active dataset changed, so I reset the protocol discussion. "
                    f"{assistant_message}"
                )
            next_payload = payload.model_copy(
                update={
                    "dataset_id": deps.dataset_id,
                    "protocol_discussion": "",
                    "status": "DISCUSSING",
                    "assistant_message": assistant_message,
                }
            )
            return NodeExecutionResult(
                new_node_state=ProtocolDiscussionState(next_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=assistant_message)],
            )

        if not latest_user_message:
            assistant_message = payload.assistant_message or initial_user_message()
            return NodeExecutionResult(
                new_node_state=ProtocolDiscussionState(payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=assistant_message)],
            )

        recent_messages: Sequence[ChatMessage] | None = (
            list(request.read_only_messages_history[-5:])
            if request.read_only_messages_history
            else None
        )
        previous_protocol_discussion = (
            payload.protocol_discussion or get_protocol_discussion_template()
        )

        try:
            updated_protocol_discussion = self.protocol_discussion_compilation(
                dataset_summary=deps.dataset_summary,
                previous_protocol_discussion=previous_protocol_discussion,
                latest_user_message=latest_user_message,
                recent_messages=recent_messages,
            )
            status = self.protocol_discussion_status(
                dataset_summary=deps.dataset_summary,
                protocol_discussion=updated_protocol_discussion,
                latest_user_message=latest_user_message,
                recent_messages=recent_messages,
            )
            causal_draft_result: ProtocolDiscussionCausalDraftResult | None = None
            if status == "READY":
                causal_draft_result = self.protocol_discussion_causal_draft(
                    protocol_discussion=updated_protocol_discussion,
                    dataset_summary=deps.dataset_summary,
                )
                if causal_draft_result.validation_issues:
                    assistant_message = self.protocol_discussion_validation_suggestion(
                        protocol_discussion=updated_protocol_discussion,
                        causal_draft=causal_draft_result.draft,
                        validation_issues=causal_draft_result.validation_issues,
                        dataset_summary=deps.dataset_summary,
                    )
                    next_payload = payload.model_copy(
                        update={
                            "dataset_id": deps.dataset_id,
                            "protocol_discussion": updated_protocol_discussion,
                            "status": "DISCUSSING",
                            "assistant_message": assistant_message,
                        }
                    )
                    return NodeExecutionResult(
                        new_node_state=ProtocolDiscussionState(next_payload),
                        new_orchestrator_state=request.orchestrator_state,
                        status="PENDING",
                        action="NEEDS_INPUT",
                        response_messages=[
                            ChatMessage(role="assistant", content=assistant_message)
                        ],
                    )

            assistant_message = self.protocol_discussion_response(
                dataset_summary=deps.dataset_summary,
                protocol_discussion=updated_protocol_discussion,
                status=status,
                latest_user_message=latest_user_message,
                recent_messages=recent_messages,
            )
        except Exception as exc:
            log.exception("PROTOCOL_DISCUSSION update failure: %s", safe_err(exc))
            assistant_message = (
                "I could not update the protocol discussion from that message. "
                "Please restate the treatment, outcome, target population, study type, "
                "or time-zero detail you want to set."
            )
            next_payload = payload.model_copy(
                update={
                    "dataset_id": deps.dataset_id,
                    "assistant_message": assistant_message,
                }
            )
            return NodeExecutionResult(
                new_node_state=ProtocolDiscussionState(next_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=assistant_message)],
            )

        next_payload = payload.model_copy(
            update={
                "dataset_id": deps.dataset_id,
                "protocol_discussion": updated_protocol_discussion,
                "status": status,
                "assistant_message": assistant_message,
            }
        )
        if status == "READY":
            request.orchestrator_state.set(
                request.node_state.name(),
                {"causal_spec_draft": causal_draft_result.draft if causal_draft_result else None},
            )
            return NodeExecutionResult(
                new_node_state=ProtocolDiscussionState(next_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="DONE",
                action="NONE",
                response_messages=[ChatMessage(role="assistant", content=assistant_message)],
            )
        return NodeExecutionResult(
            new_node_state=ProtocolDiscussionState(next_payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[ChatMessage(role="assistant", content=assistant_message)],
        )

    def protocol_discussion_compilation(
        self,
        *,
        dataset_summary: DatasetSummaryModel,
        previous_protocol_discussion: str,
        latest_user_message: str,
        recent_messages: Sequence[ChatMessage] | None,
    ) -> str:
        response = self._llm.generate(
            system_prompt=get_protocol_discussion_compilation_prompt(),
            user_prompt=json.dumps(
                {
                    "dataset_summary": dataset_summary.model_dump(mode="json"),
                    "previous_protocol_discussion": previous_protocol_discussion,
                    "latest_user_message": latest_user_message,
                    "recent_messages": [
                        {"role": message.role, "content": message.content}
                        for message in recent_messages or []
                    ],
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="pro", temperature=0.2),
            history=recent_messages,
        )
        return response.content.strip()

    def protocol_discussion_status(
        self,
        *,
        dataset_summary: DatasetSummaryModel,
        protocol_discussion: str,
        latest_user_message: str,
        recent_messages: Sequence[ChatMessage] | None,
    ) -> ProtocolDiscussionStatus:
        result = self._llm.generate_json(
            schema=ProtocolDiscussionStatusResult,
            system_prompt=get_protocol_discussion_status_prompt(),
            user_prompt=json.dumps(
                {
                    "dataset_summary": dataset_summary.model_dump(mode="json"),
                    "protocol_discussion": protocol_discussion,
                    "latest_user_message": latest_user_message,
                    "recent_messages": [
                        {"role": message.role, "content": message.content}
                        for message in recent_messages or []
                    ],
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.0),
            history=recent_messages,
            max_attempts=2,
        )
        return result.status

    def protocol_discussion_response(
        self,
        *,
        dataset_summary: DatasetSummaryModel,
        protocol_discussion: str,
        status: ProtocolDiscussionStatus,
        latest_user_message: str,
        recent_messages: Sequence[ChatMessage] | None,
    ) -> str:
        response = self._llm.generate(
            system_prompt=get_protocol_discussion_response_prompt(),
            user_prompt=json.dumps(
                {
                    "dataset_summary": dataset_summary.model_dump(mode="json"),
                    "protocol_discussion": protocol_discussion,
                    "status": status,
                    "latest_user_message": latest_user_message,
                    "recent_messages": [
                        {"role": message.role, "content": message.content}
                        for message in recent_messages or []
                    ],
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.3),
            history=recent_messages,
        )
        return response.content.strip()

    def protocol_discussion_causal_draft(
        self,
        *,
        protocol_discussion: str,
        dataset_summary: DatasetSummaryModel,
    ) -> ProtocolDiscussionCausalDraftResult:
        from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
            ID_COL_AUTO_FILL,
            CausalSpecDraft,
        )

        try:
            schema = CausalSpecDraft.for_dataset_summary(dataset_summary)
            draft = self._llm.generate_json(
                schema=schema,
                system_prompt=get_protocol_discussion_causal_draft_prompt(),
                user_prompt=json.dumps(
                    {
                        "protocol_discussion": protocol_discussion,
                        "dataset_summary": dataset_summary.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
                config=LLMConfig(model="pro", temperature=0.0),
                history=None,
                max_attempts=2,
            )
        except Exception as exc:
            return ProtocolDiscussionCausalDraftResult(
                draft=None,
                validation_issues=[
                    {
                        "role": "causal_draft",
                        "column": None,
                        "issue": (
                            "The signed-off protocol could not be converted into a valid "
                            f"causal draft: {safe_err(exc)}"
                        ),
                        "suggestions": [
                            "update dataset and make sure the protocol columns exist exactly as named"
                        ],
                    }
                ],
            )

        profiles_by_name = {
            str(profile.name).strip(): profile
            for profile in dataset_summary.profiles
            if str(profile.name).strip()
        }
        validation_issues: list[dict[str, Any]] = []

        treatment_column = str(draft.treatment_column).strip()
        treatment_profile = profiles_by_name.get(treatment_column)
        if treatment_profile is None:
            validation_issues.append(
                {
                    "role": "treatment",
                    "column": treatment_column,
                    "issue": f'Treatment column "{treatment_column}" was not found.',
                    "suggestions": [
                        f"update dataset and create a cleaned binary treatment column from {treatment_column}"
                    ],
                }
            )
        else:
            if treatment_profile.distinct_count != 2:
                validation_issues.append(
                    {
                        "role": "treatment",
                        "column": treatment_column,
                        "issue": (
                            f'Treatment column "{treatment_column}" must be binary but has '
                            f"{treatment_profile.distinct_count} distinct non-missing values."
                        ),
                        "evidence": {
                            "distinct_count": treatment_profile.distinct_count,
                            "inferred_kind": treatment_profile.inferred_kind,
                        },
                        "suggestions": [
                            f"update dataset and create a cleaned binary treatment column from {treatment_column}"
                        ],
                    }
                )
            if treatment_profile.n_missing > 0:
                validation_issues.append(
                    {
                        "role": "treatment",
                        "column": treatment_column,
                        "issue": (
                            f'Treatment column "{treatment_column}" has '
                            f"{treatment_profile.n_missing} missing values."
                        ),
                        "evidence": {
                            "n_missing": treatment_profile.n_missing,
                            "missing_rate": treatment_profile.missing_rate,
                        },
                        "suggestions": [
                            f"update dataset and remove rows where {treatment_column} is missing",
                            f"update dataset and impute missing values in {treatment_column}",
                        ],
                    }
                )

        outcome_column = str(draft.outcome_column).strip()
        outcome_profile = profiles_by_name.get(outcome_column)
        if outcome_profile is None:
            validation_issues.append(
                {
                    "role": "outcome",
                    "column": outcome_column,
                    "issue": f'Outcome column "{outcome_column}" was not found.',
                    "suggestions": [
                        f"update dataset and create a cleaned binary outcome column from {outcome_column}"
                    ],
                }
            )
        else:
            outcome_distinct = outcome_profile.distinct_count
            outcome_is_binary = outcome_distinct == 2
            outcome_is_continuous = (
                outcome_profile.inferred_kind == "NUMERIC"
                and outcome_distinct is not None
                and outcome_distinct > 2
            )
            if not outcome_is_binary and not outcome_is_continuous:
                validation_issues.append(
                    {
                        "role": "outcome",
                        "column": outcome_column,
                        "issue": (
                            f'Outcome column "{outcome_column}" must be binary or numeric '
                            f"continuous, but it has {outcome_distinct} distinct non-missing "
                            f"values and is classified as {outcome_profile.inferred_kind}."
                        ),
                        "evidence": {
                            "distinct_count": outcome_distinct,
                            "inferred_kind": outcome_profile.inferred_kind,
                        },
                        "suggestions": [
                            f"update dataset and create a cleaned binary outcome column from {outcome_column}"
                        ],
                    }
                )
            if outcome_profile.n_missing > 0:
                validation_issues.append(
                    {
                        "role": "outcome",
                        "column": outcome_column,
                        "issue": (
                            f'Outcome column "{outcome_column}" has '
                            f"{outcome_profile.n_missing} missing values."
                        ),
                        "evidence": {
                            "n_missing": outcome_profile.n_missing,
                            "missing_rate": outcome_profile.missing_rate,
                        },
                        "suggestions": [
                            f"update dataset and remove rows where {outcome_column} is missing",
                            f"update dataset and impute missing values in {outcome_column}",
                        ],
                    }
                )

        id_col = str(draft.id_col).strip()
        if id_col and id_col != ID_COL_AUTO_FILL:
            id_profile = profiles_by_name.get(id_col)
            if id_profile is None:
                validation_issues.append(
                    {
                        "role": "id",
                        "column": id_col,
                        "issue": f'ID column "{id_col}" was not found.',
                        "suggestions": [
                            f"update dataset and create a complete ID column from {id_col}"
                        ],
                    }
                )
            elif id_profile.n_missing > 0:
                validation_issues.append(
                    {
                        "role": "id",
                        "column": id_col,
                        "issue": f'ID column "{id_col}" has {id_profile.n_missing} missing values.',
                        "evidence": {
                            "n_missing": id_profile.n_missing,
                            "missing_rate": id_profile.missing_rate,
                        },
                        "suggestions": [
                            f"update dataset and create a complete ID column from {id_col}",
                            f"update dataset and remove rows where {id_col} is missing",
                        ],
                    }
                )

        negative_control_outcome = (
            str(draft.negative_control_outcome).strip()
            if draft.negative_control_outcome is not None
            else None
        )
        if negative_control_outcome:
            negative_profile = profiles_by_name.get(negative_control_outcome)
            if negative_profile is None:
                validation_issues.append(
                    {
                        "role": "negative_control_outcome",
                        "column": negative_control_outcome,
                        "issue": (
                            f'Negative-control outcome column "{negative_control_outcome}" '
                            "was not found."
                        ),
                        "suggestions": [
                            "update dataset and create a cleaned binary negative-control "
                            f"outcome column from {negative_control_outcome}"
                        ],
                    }
                )
            else:
                if negative_profile.distinct_count != 2:
                    validation_issues.append(
                        {
                            "role": "negative_control_outcome",
                            "column": negative_control_outcome,
                            "issue": (
                                "Negative-control outcome column "
                                f'"{negative_control_outcome}" must be binary but has '
                                f"{negative_profile.distinct_count} distinct non-missing values."
                            ),
                            "evidence": {
                                "distinct_count": negative_profile.distinct_count,
                                "inferred_kind": negative_profile.inferred_kind,
                            },
                            "suggestions": [
                                "update dataset and create a cleaned binary negative-control "
                                f"outcome column from {negative_control_outcome}"
                            ],
                        }
                    )
                if negative_profile.n_missing > 0:
                    validation_issues.append(
                        {
                            "role": "negative_control_outcome",
                            "column": negative_control_outcome,
                            "issue": (
                                f'Negative-control outcome column "{negative_control_outcome}" '
                                f"has {negative_profile.n_missing} missing values."
                            ),
                            "evidence": {
                                "n_missing": negative_profile.n_missing,
                                "missing_rate": negative_profile.missing_rate,
                            },
                            "suggestions": [
                                f"update dataset and remove rows where {negative_control_outcome} is missing",
                                f"update dataset and impute missing values in {negative_control_outcome}",
                            ],
                        }
                    )

        return ProtocolDiscussionCausalDraftResult(
            draft=draft,
            validation_issues=validation_issues,
        )

    def protocol_discussion_validation_suggestion(
        self,
        *,
        protocol_discussion: str,
        causal_draft: Any | None,
        validation_issues: list[dict[str, Any]],
        dataset_summary: DatasetSummaryModel,
    ) -> str:
        response = self._llm.generate(
            system_prompt=get_protocol_discussion_validation_suggestion_prompt(),
            user_prompt=json.dumps(
                {
                    "protocol_discussion": protocol_discussion,
                    "causal_draft": (
                        causal_draft.model_dump(mode="json") if causal_draft is not None else None
                    ),
                    "validation_issues": validation_issues,
                    "dataset_summary": dataset_summary.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.2),
            history=None,
        )
        return response.content.strip()


__all__ = [
    "ProtocolDiscussionCausalDraftResult",
    "ProtocolDiscussionNode",
    "ProtocolDiscussionStatusResult",
]
