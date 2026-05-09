from __future__ import annotations

import json
import re
from collections.abc import Sequence
from difflib import get_close_matches
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
    get_protocol_discussion_get_node_info,
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

        try:
            decision = self._call_update(
                draft=payload.draft,
                dataset_summary=deps.dataset_summary,
                latest_user_message=latest_user_message,
                history=(
                    list(request.read_only_messages_history[-4:])
                    if request.read_only_messages_history
                    else None
                ),
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
        validation_message = _draft_blocking_message(
            draft=draft,
            dataset_summary=deps.dataset_summary,
        )
        column_notes = _column_structure_notes(
            draft=draft,
            dataset_summary=deps.dataset_summary,
        )
        population_note = _population_guidance(draft.target_population)

        assistant_message = _compose_assistant_message(
            decision.assistant_message,
            validation_message,
            column_notes,
            population_note,
            dataset_changed=dataset_changed,
        )

        if decision.next_action == "confirm" and validation_message is None:
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


def _compose_assistant_message(
    base_message: str,
    validation_message: str | None,
    column_notes: Sequence[str],
    population_note: str | None,
    *,
    dataset_changed: bool,
) -> str:
    parts = [_prefix_dataset_reset_message(base_message.strip(), dataset_changed=dataset_changed)]
    if column_notes:
        parts.append("Column structure:\n" + "\n".join(f"- {note}" for note in column_notes))
    if population_note:
        parts.append(population_note)
    if validation_message:
        parts.append(validation_message)
    return "\n\n".join(part for part in parts if part.strip())


def _draft_blocking_message(
    *,
    draft: ProtocolCausalDraftModel,
    dataset_summary: DatasetSummaryModel,
) -> str | None:
    issues: list[str] = []
    missing_required = _missing_required_fields(draft)
    if missing_required:
        issues.append(
            "Before I can accept the causal draft, I still need: "
            + ", ".join(missing_required)
            + "."
        )

    missing_columns = _missing_selected_columns(
        draft=draft,
        dataset_summary=dataset_summary,
    )
    if missing_columns:
        issues.append(_missing_columns_message(missing_columns, dataset_summary))

    role_conflicts = _role_conflict_messages(draft)
    issues.extend(role_conflicts)

    if not issues:
        return None
    return "\n".join(issues)


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


def _missing_columns_message(
    missing_columns: Sequence[tuple[ColumnRole, str]],
    dataset_summary: DatasetSummaryModel,
) -> str:
    available_columns = _summary_column_names(dataset_summary)
    lines = [
        "I cannot accept the draft because these selected variables are not current dataset columns:"
    ]
    for role, column in missing_columns:
        suggestions = _close_column_suggestions(column, available_columns)
        suggestion_text = (
            f" Close existing columns: {', '.join(suggestions)}." if suggestions else ""
        )
        lines.append(f"- {column} ({role}).{suggestion_text}")
        if suggestions:
            lines.append(
                f"  You can choose an existing column, or say `update dataset and rename {suggestions[0]} to {column}`."
            )
            lines.append(
                f"  If it must be derived, say `update dataset and create {column} from {', '.join(suggestions[:2])} by <rule>`."
            )
        else:
            lines.append(
                f"  You can choose an existing column, or say `update dataset and create {column} from <existing columns> by <rule>`."
            )
    return "\n".join(lines)


def _role_conflict_messages(draft: ProtocolCausalDraftModel) -> list[str]:
    messages: list[str] = []
    treatment = _norm(draft.treatment_column)
    outcome = _norm(draft.outcome_column)
    if treatment and outcome and treatment == outcome:
        messages.append("Treatment and outcome must be different columns.")

    covariates = {_norm(column): column for column in draft.covariates if _norm(column)}
    effect_modifiers = {_norm(column): column for column in draft.effect_modifiers if _norm(column)}
    overlap = sorted(set(covariates).intersection(effect_modifiers))
    if overlap:
        messages.append(
            "Covariates and effect modifiers must not overlap: "
            + ", ".join(covariates[column] for column in overlap)
            + "."
        )

    protected = {key for key in (treatment, outcome) if key}
    protected_overlap = [
        column
        for column in [*draft.covariates, *draft.effect_modifiers]
        if _norm(column) in protected
    ]
    if protected_overlap:
        messages.append(
            "Treatment and outcome columns cannot also be covariates or effect modifiers: "
            + ", ".join(protected_overlap)
            + "."
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
            messages.append(
                f"{draft.negative_control_outcome} cannot be the negative-control outcome "
                f"and also used as {', '.join(other_roles)}."
            )
    return messages


def _norm(value: str | None) -> str:
    return value.strip().casefold() if value else ""


def _close_column_suggestions(
    column: str, available_columns: Sequence[str], *, limit: int = 3
) -> list[str]:
    matches = get_close_matches(column, list(available_columns), n=limit, cutoff=0.55)
    if matches:
        return matches

    normalized = re.sub(r"[^a-z0-9]+", " ", column.casefold()).strip()
    tokens = [token for token in normalized.split() if len(token) >= 3]
    token_matches: list[str] = []
    for candidate in available_columns:
        candidate_normalized = re.sub(r"[^a-z0-9]+", " ", candidate.casefold())
        if any(token in candidate_normalized for token in tokens):
            token_matches.append(candidate)
        if len(token_matches) >= limit:
            break
    return token_matches


def _column_structure_notes(
    *,
    draft: ProtocolCausalDraftModel,
    dataset_summary: DatasetSummaryModel,
) -> list[str]:
    profiles_by_name = {str(profile.name): profile for profile in dataset_summary.profiles}
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()
    for role, column in _selected_role_columns(draft):
        key = (role, column)
        if key in seen:
            continue
        seen.add(key)
        profile = profiles_by_name.get(column)
        if profile is None:
            continue
        notes.append(_column_structure_note(profile=profile, role=role))
    return notes


def _column_structure_note(*, profile: ColumnProfileModel, role: ColumnRole) -> str:
    missing_rate_pct = profile.missing_rate * 100
    pieces = [
        f"{profile.name} as {role}",
        f"{profile.inferred_kind.lower()}",
    ]
    if profile.dtype:
        pieces.append(f"dtype {profile.dtype}")
    pieces.append(f"{profile.n_missing} missing ({missing_rate_pct:.1f}%)")
    if profile.distinct_count is not None:
        pieces.append(f"{profile.distinct_count} distinct")
    structure = ", ".join(pieces)
    visible = _visible_profile_values(profile)
    plausibility = _role_plausibility(profile=profile, role=role)
    note = f"{structure}. {visible} {plausibility}".strip()
    if _profile_may_need_next_step(profile=profile, role=role):
        note += " Don't worry, we can figure this out in the next step."
    return note


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


def _role_plausibility(*, profile: ColumnProfileModel, role: ColumnRole) -> str:
    if role == "treatment":
        if profile.distinct_count == 2:
            return "This looks plausible for a binary treatment column."
        return "This may be risky for treatment because treatment must become binary."
    if role == "outcome":
        if isinstance(profile, NumericColumnProfileModel):
            return "This can be plausible for a binary or continuous outcome depending on the target endpoint."
        return "This can be plausible for an outcome if it represents the endpoint after time zero."
    if role == "covariate":
        return "This is plausible as a covariate if it is measured before time zero."
    if role == "effect modifier":
        return "This is plausible as an effect modifier if it is baseline and clinically relevant for heterogeneity."
    return "This is plausible as a negative-control outcome only if treatment should not affect it."


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


def _population_guidance(target_population: str | None) -> str | None:
    if not target_population:
        return None
    normalized = target_population.strip().casefold()
    if normalized in {"all rows", "all patients", "entire dataset", "all observations", "everyone"}:
        return None
    command = _population_filter_command(target_population)
    return (
        "Target population can stay as draft text. If you want the dataset physically "
        f"filtered first, say `{command}`."
    )


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
    "_column_structure_notes",
    "_draft_blocking_message",
    "_population_guidance",
]
