from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_compile_causal_spec_draft_prompt,
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_response_prompt,
    get_protocol_discussion_update_prompt,
    initial_user_message,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolCausalDraftModel,
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    ID_COL_AUTO_FILL,
    CausalSpecDraft,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    ColumnProfileModel,
    DatasetSummaryModel,
    DatetimeColumnProfileModel,
    NumericColumnProfileModel,
    OtherColumnProfileModel,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)

NextAction = Literal["continue", "confirm"]
ColumnRole = Literal[
    "treatment",
    "outcome",
    "covariate",
    "effect modifier",
    "negative-control outcome",
]


class _DraftDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    draft: ProtocolCausalDraftModel = Field(default_factory=ProtocolCausalDraftModel)
    next_action: NextAction


class _DraftResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


def compile_causal_spec_draft_from_discussion(
    *,
    llm: LLMService,
    protocol_discussion: str,
    dataset_summary: DatasetSummaryModel,
    retry_feedback: str | None = None,
    previous_draft: CausalSpecDraft | None = None,
) -> CausalSpecDraft:
    schema = CausalSpecDraft.for_dataset_summary(dataset_summary)
    user_payload: dict[str, Any] = {
        "protocol_discussion": protocol_discussion,
        "dataset_summary": dataset_summary.model_dump(mode="json"),
    }
    if previous_draft is not None:
        user_payload["previous_draft"] = previous_draft.model_dump(mode="json")
    if retry_feedback is not None:
        user_payload["retry_feedback"] = retry_feedback

    return llm.generate_json(
        schema=schema,
        system_prompt=get_compile_causal_spec_draft_prompt(),
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        config=LLMConfig(model="basic", temperature=0.0),
        history=None,
        max_attempts=2,
    )


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
                f"{self.name}: expected ProtocolDiscussionState, got {type(request.node_state).__name__}"
            )

        deps = ProtocolDiscussionDeps.from_request(request)
        payload = request.node_state.payload.model_copy(deep=True)
        dataset_changed = payload.dataset_id is not None and payload.dataset_id != deps.dataset_id
        if dataset_changed or payload.dataset_id is None:
            payload = ProtocolDiscussionPayloadModel(dataset_id=deps.dataset_id)

        latest_user_message = _latest_user_message(request.read_only_messages_history)
        if not latest_user_message:
            return self._needs_input_result(
                request=request,
                payload=payload.model_copy(
                    update={
                        "assistant_message": _prefix_dataset_reset_message(
                            initial_user_message(),
                            dataset_changed=dataset_changed,
                        )
                    }
                ),
            )

        previous_draft = payload.draft
        history = (
            list(request.read_only_messages_history[-4:])
            if request.read_only_messages_history
            else None
        )

        try:
            decision = self._call_update(
                draft=previous_draft,
                dataset_summary=deps.dataset_summary,
                latest_user_message=latest_user_message,
                history=history,
            )
        except Exception as exc:
            log.exception("PROTOCOL_DISCUSSION draft update failure: %s", safe_err(exc))
            return self._needs_input_result(
                request=request,
                payload=payload.model_copy(
                    update={
                        "assistant_message": (
                            "I could not update the causal draft from that message. "
                            "Please restate the treatment, outcome, population, study type, "
                            "or time zero you want to set."
                        )
                    }
                ),
            )

        draft = decision.draft
        validation_context = _draft_validation_context(
            draft=draft,
            dataset_summary=deps.dataset_summary,
        )
        final_next_action: NextAction = (
            "confirm"
            if decision.next_action == "confirm"
            and not bool(validation_context["has_blocking_issues"])
            else "continue"
        )
        assistant_message = self._call_response(
            previous_draft=previous_draft,
            updated_draft=draft,
            dataset_summary=deps.dataset_summary,
            latest_user_message=latest_user_message,
            requested_next_action=decision.next_action,
            final_next_action=final_next_action,
            validation_context=validation_context,
            dataset_changed=dataset_changed,
            history=history,
        )

        if final_next_action == "confirm":
            causal_spec_draft = _to_causal_spec_draft(draft)
            confirmed_payload = payload.model_copy(
                update={
                    "dataset_id": deps.dataset_id,
                    "draft": draft,
                    "phase": "CONFIRMED",
                    "assistant_message": assistant_message,
                }
            )
            request.orchestrator_state.set(
                request.node_state.name(),
                {"causal_spec_draft": causal_spec_draft},
            )
            return self._done_result(
                request=request,
                payload=confirmed_payload,
                user_message=assistant_message,
            )

        return self._needs_input_result(
            request=request,
            payload=payload.model_copy(
                update={
                    "dataset_id": deps.dataset_id,
                    "draft": draft,
                    "phase": "DISCUSSING",
                    "assistant_message": assistant_message,
                }
            ),
        )

    def _call_update(
        self,
        *,
        draft: ProtocolCausalDraftModel,
        dataset_summary: DatasetSummaryModel,
        latest_user_message: str,
        history: Sequence[ChatMessage] | None,
    ) -> _DraftDecisionModel:
        payload = {
            "current_draft": draft.model_dump(mode="json"),
            "latest_user_message": latest_user_message,
            "dataset_summary": _compact_dataset_summary(dataset_summary),
            "dataset_column_names": _summary_column_names(dataset_summary),
        }
        return self._llm.generate_json(
            schema=_DraftDecisionModel,
            system_prompt=get_protocol_discussion_update_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.4),
            history=history,
            max_attempts=2,
        )

    def _call_response(
        self,
        *,
        previous_draft: ProtocolCausalDraftModel,
        updated_draft: ProtocolCausalDraftModel,
        dataset_summary: DatasetSummaryModel,
        latest_user_message: str,
        requested_next_action: NextAction,
        final_next_action: NextAction,
        validation_context: dict[str, Any],
        dataset_changed: bool,
        history: Sequence[ChatMessage] | None,
    ) -> str:
        payload = {
            "latest_user_message": latest_user_message,
            "previous_draft": previous_draft.model_dump(mode="json"),
            "updated_draft": updated_draft.model_dump(mode="json"),
            "requested_next_action": requested_next_action,
            "final_next_action": final_next_action,
            "dataset_changed": dataset_changed,
            "validation_context": validation_context,
            "selected_column_context": _selected_column_context(
                draft=updated_draft,
                dataset_summary=dataset_summary,
            ),
            "population_context": _population_context(updated_draft.target_population),
            "dataset_summary": _compact_dataset_summary(dataset_summary),
            "dataset_column_names": _summary_column_names(dataset_summary),
        }
        response = self._llm.generate_json(
            schema=_DraftResponseModel,
            system_prompt=get_protocol_discussion_response_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.3),
            history=history,
            max_attempts=2,
        )
        return response.assistant_message

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: ProtocolDiscussionPayloadModel,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ProtocolDiscussionState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[
                ChatMessage(
                    role="assistant", content=payload.assistant_message or initial_user_message()
                )
            ],
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


def _prefix_dataset_reset_message(message: str, *, dataset_changed: bool) -> str:
    if not dataset_changed:
        return message
    return f"The active dataset changed, so I reset the causal draft. {message}"


def _draft_validation_context(
    *,
    draft: ProtocolCausalDraftModel,
    dataset_summary: DatasetSummaryModel,
) -> dict[str, Any]:
    missing_required = _missing_required_fields(draft)
    missing_columns = _missing_selected_columns(
        draft=draft,
        dataset_summary=dataset_summary,
    )
    role_conflicts = _role_conflicts(draft)
    return {
        "has_blocking_issues": bool(missing_required or missing_columns or role_conflicts),
        "missing_required_fields": missing_required,
        "missing_selected_columns": [
            {"role": role, "column": column} for role, column in missing_columns
        ],
        "role_conflicts": role_conflicts,
    }


def _missing_required_fields(draft: ProtocolCausalDraftModel) -> list[str]:
    missing: list[str] = []
    if not draft.treatment_column:
        missing.append("treatment column")
    if not draft.outcome_column:
        missing.append("outcome column")
    if not draft.target_population:
        missing.append("target population")
    if not draft.study_type:
        missing.append("study type")
    if not draft.time_zero:
        missing.append("time zero")
    return missing


def _selected_role_columns(draft: ProtocolCausalDraftModel) -> list[tuple[ColumnRole, str]]:
    selected: list[tuple[ColumnRole, str]] = []
    if draft.treatment_column:
        selected.append(("treatment", draft.treatment_column))
    if draft.outcome_column:
        selected.append(("outcome", draft.outcome_column))
    selected.extend(("covariate", column) for column in draft.covariates)
    selected.extend(("effect modifier", column) for column in draft.effect_modifiers)
    if draft.negative_control_outcome:
        selected.append(("negative-control outcome", draft.negative_control_outcome))
    return selected


def _missing_selected_columns(
    *,
    draft: ProtocolCausalDraftModel,
    dataset_summary: DatasetSummaryModel,
) -> list[tuple[ColumnRole, str]]:
    columns = set(_summary_column_names(dataset_summary))
    return [
        (role, column) for role, column in _selected_role_columns(draft) if column not in columns
    ]


def _role_conflicts(draft: ProtocolCausalDraftModel) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    treatment = _norm(draft.treatment_column)
    outcome = _norm(draft.outcome_column)
    if treatment and outcome and treatment == outcome:
        conflicts.append(
            {
                "type": "treatment_outcome_same_column",
                "columns": [draft.treatment_column, draft.outcome_column],
            }
        )

    covariates = {_norm(column): column for column in draft.covariates if _norm(column)}
    effect_modifiers = {_norm(column): column for column in draft.effect_modifiers if _norm(column)}
    overlap = sorted(set(covariates).intersection(effect_modifiers))
    if overlap:
        conflicts.append(
            {
                "type": "covariate_effect_modifier_overlap",
                "columns": [covariates[column] for column in overlap],
            }
        )

    protected = {key for key in (treatment, outcome) if key}
    protected_overlap = [
        column
        for column in [*draft.covariates, *draft.effect_modifiers]
        if _norm(column) in protected
    ]
    if protected_overlap:
        conflicts.append(
            {
                "type": "treatment_or_outcome_reused_as_adjustment_or_modifier",
                "columns": protected_overlap,
            }
        )

    negative_control = _norm(draft.negative_control_outcome)
    if negative_control:
        other_roles = [
            role
            for role, column in _selected_role_columns(
                draft.model_copy(update={"negative_control_outcome": None})
            )
            if _norm(column) == negative_control
        ]
        if other_roles:
            conflicts.append(
                {
                    "type": "negative_control_outcome_reused_in_other_role",
                    "column": draft.negative_control_outcome,
                    "other_roles": other_roles,
                }
            )
    return conflicts


def _norm(value: str | None) -> str:
    return value.strip().casefold() if value else ""


def _selected_column_context(
    *,
    draft: ProtocolCausalDraftModel,
    dataset_summary: DatasetSummaryModel,
) -> list[dict[str, Any]]:
    profiles_by_name = {str(profile.name): profile for profile in dataset_summary.profiles}
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for role, column in _selected_role_columns(draft):
        key = (role, column)
        if key in seen:
            continue
        seen.add(key)
        profile = profiles_by_name.get(column)
        if profile is None:
            selected.append({"role": role, "column": column, "exists": False})
            continue
        selected.append(
            {
                "role": role,
                "column": column,
                "exists": True,
                "profile": _compact_column_profile(profile),
                "may_need_next_step": _profile_may_need_next_step(
                    profile=profile,
                    role=role,
                ),
            }
        )
    return selected


def _compact_column_profile(profile: ColumnProfileModel) -> dict[str, Any]:
    return {
        "name": profile.name,
        "dtype": profile.dtype,
        "kind": profile.inferred_kind,
        "missing": profile.n_missing,
        "missing_rate": profile.missing_rate,
        "distinct_count": profile.distinct_count,
        "visible_values": _visible_profile_values(profile),
    }


def _visible_profile_values(profile: ColumnProfileModel) -> str:
    if isinstance(profile, NumericColumnProfileModel):
        summary = profile.summary
        if summary.min is not None and summary.max is not None:
            return f"Range {summary.min:g} to {summary.max:g}."
        return ""
    if isinstance(profile, DatetimeColumnProfileModel):
        summary = profile.summary
        if summary.min and summary.max:
            return f"Observed from {summary.min} to {summary.max}."
        return ""
    if isinstance(profile, BooleanColumnProfileModel):
        counts = ", ".join(
            f"{key}: {value}" for key, value in list(profile.summary.counts.items())[:4]
        )
        return f"Values {counts}." if counts else ""
    if isinstance(profile, CategoricalColumnProfileModel):
        values = ", ".join(
            f"{item.value}: {item.count}" for item in profile.summary.top_categories[:4]
        )
        return f"Top values {values}." if values else ""
    if isinstance(profile, OtherColumnProfileModel):
        values = ", ".join(profile.summary.distinct_values_sample[:4])
        return f"Sample values {values}." if values else ""
    return ""


def _profile_may_need_next_step(*, profile: ColumnProfileModel, role: ColumnRole) -> bool:
    if profile.n_missing > 0:
        return True
    if role == "treatment" and profile.distinct_count != 2:
        return True
    if isinstance(profile, CategoricalColumnProfileModel):
        return any(_looks_unknown_like(item.value) for item in profile.summary.top_categories)
    if isinstance(profile, NumericColumnProfileModel) and profile.distinct_count is not None:
        return 2 < profile.distinct_count <= 10
    return False


def _looks_unknown_like(value: str) -> bool:
    normalized = value.strip().casefold()
    return normalized in {
        "",
        "unknown",
        "unk",
        "other",
        "other/unknown",
        "missing",
        "na",
        "n/a",
        "none",
    }


def _population_context(target_population: str | None) -> dict[str, Any]:
    if not target_population:
        return {
            "target_population": None,
            "specific_population": False,
            "possible_filter_command": None,
        }
    normalized = target_population.strip().casefold()
    if normalized in {"all rows", "all patients", "entire dataset", "all observations", "everyone"}:
        return {
            "target_population": target_population,
            "specific_population": False,
            "possible_filter_command": None,
        }
    command = _population_filter_command(target_population)
    return {
        "target_population": target_population,
        "specific_population": True,
        "possible_filter_command": command,
    }


def _population_filter_command(target_population: str) -> str:
    text = target_population.strip()
    comparison = re.search(
        r"([A-Za-z_][A-Za-z0-9_]*\s*(?:==|=|>=|<=|>|<)\s*[^,;.]+)",
        text,
    )
    if comparison:
        condition = comparison.group(1).strip()
        condition = re.sub(r"(?<![<>=!])=(?!=)", "==", condition, count=1)
        return f"update dataset and filter rows where {condition.strip()}"
    return "update dataset and filter rows where <population condition>"


def _to_causal_spec_draft(draft: ProtocolCausalDraftModel) -> CausalSpecDraft:
    # Required-field validation happens before this function is called.
    return CausalSpecDraft(
        id_col=ID_COL_AUTO_FILL,
        treatment_column=str(draft.treatment_column),
        outcome_column=str(draft.outcome_column),
        negative_control_outcome=draft.negative_control_outcome,
        covariates=list(draft.covariates),
        effect_modifiers=list(draft.effect_modifiers),
        target_population=draft.target_population,
        study_type=draft.study_type,
        time_zero=draft.time_zero,
    )


def _summary_column_names(summary: DatasetSummaryModel) -> list[str]:
    return [str(profile.name).strip() for profile in summary.profiles if str(profile.name).strip()]


def _compact_dataset_summary(summary: DatasetSummaryModel) -> dict[str, Any]:
    return {
        "n_rows": summary.n_rows,
        "columns": [
            {
                "name": profile.name,
                "dtype": profile.dtype,
                "kind": profile.inferred_kind,
                "missing": profile.n_missing,
                "missing_rate": profile.missing_rate,
                "distinct_count": profile.distinct_count,
                "visible_values": _visible_profile_values(profile),
            }
            for profile in summary.profiles
        ],
    }


__all__ = [
    "ProtocolDiscussionNode",
    "compile_causal_spec_draft_from_discussion",
]











def _question_to_ask(
    *,
    draft: ProtocolCausalDraftModel,
    dataset_summary: DatasetSummaryModel,
    last_5_messages: Sequence[ChatMessage],
) -> str:
    # Implementation for determining the next question to ask based on the draft,
    # dataset summary, and recent messages.
    pass










def _fill_protocol_causal_draft_model(
    *,
    pre_draft: ProtocolCausalDraftModel,
    dataset_summary: DatasetSummaryModel,
    last_5_messages: Sequence[ChatMessage],) -> ProtocolCausalDraftModel:
    
    