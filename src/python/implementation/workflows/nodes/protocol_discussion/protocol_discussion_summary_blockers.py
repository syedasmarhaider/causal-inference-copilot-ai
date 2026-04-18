from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    ColumnProfileModel,
    DatasetSummaryModel,
    NumericColumnProfileModel,
)

BlockerRole = Literal["treatment", "outcome", "covariate", "effect_modifier"]
BlockerIssue = Literal[
    "missing_values",
    "treatment_not_binary",
    "outcome_mapping_required",
    "unknown_category_present",
    "coded_categories_need_decision",
]


@dataclass(frozen=True)
class ProtocolSummaryBlocker:
    column: str
    role: BlockerRole
    issue: BlockerIssue
    user_question: str

    @property
    def protocol_question_prefix(self) -> str:
        if self.role in {"treatment", "outcome"}:
            return "14)"
        return "15)"


def scan_protocol_summary_blockers(
    *,
    dataset_summary: DatasetSummaryModel,
    treatment_column: str,
    outcome_column: str,
    covariates: list[str],
    effect_modifiers: list[str],
) -> list[ProtocolSummaryBlocker]:
    profiles_by_name = {str(profile.name): profile for profile in dataset_summary.profiles}
    blockers: list[ProtocolSummaryBlocker] = []
    seen: set[tuple[str, BlockerRole, BlockerIssue]] = set()

    def add(column: str, role: BlockerRole, issue: BlockerIssue, user_question: str) -> None:
        key = (column, role, issue)
        if key in seen:
            return
        seen.add(key)
        blockers.append(
            ProtocolSummaryBlocker(
                column=column,
                role=role,
                issue=issue,
                user_question=user_question,
            )
        )

    treatment_profile = profiles_by_name.get(treatment_column)
    if treatment_profile is not None:
        if treatment_profile.n_missing > 0:
            add(
                treatment_column,
                "treatment",
                "missing_values",
                f"The treatment column '{treatment_column}' has {treatment_profile.n_missing} missing values in the summary. "
                "How should we handle those rows before modeling?",
            )
        if treatment_profile.distinct_count is not None and treatment_profile.distinct_count != 2:
            add(
                treatment_column,
                "treatment",
                "treatment_not_binary",
                f"The treatment column '{treatment_column}' currently shows {treatment_profile.distinct_count} distinct values. "
                "Please state the exact two treatment values to keep and how any unexpected values should be handled.",
            )

    outcome_profile = profiles_by_name.get(outcome_column)
    if outcome_profile is not None:
        if outcome_profile.n_missing > 0:
            add(
                outcome_column,
                "outcome",
                "missing_values",
                f"The outcome column '{outcome_column}' has {outcome_profile.n_missing} missing values in the summary. "
                "How should we handle those rows before modeling?",
            )
        if _needs_explicit_outcome_mapping(outcome_profile):
            add(
                outcome_column,
                "outcome",
                "outcome_mapping_required",
                f"The outcome column '{outcome_column}' looks like a coded endpoint rather than an already finalized binary label. "
                "Please state the exact outcome mapping that should be used before modeling.",
            )

    for column in covariates:
        profile = profiles_by_name.get(column)
        if profile is None:
            continue
        _add_baseline_blockers(blockers_add=add, column=column, role="covariate", profile=profile)

    for column in effect_modifiers:
        profile = profiles_by_name.get(column)
        if profile is None:
            continue
        _add_baseline_blockers(
            blockers_add=add,
            column=column,
            role="effect_modifier",
            profile=profile,
        )

    return blockers


def unresolved_summary_blockers(
    *,
    protocol_discussion: str,
    blockers: list[ProtocolSummaryBlocker],
) -> list[ProtocolSummaryBlocker]:
    unresolved: list[ProtocolSummaryBlocker] = []
    for blocker in blockers:
        relevant_line = _find_protocol_line(protocol_discussion, blocker.protocol_question_prefix)
        if relevant_line is None:
            unresolved.append(blocker)
            continue
        lower_line = relevant_line.lower()
        if "unclear" in lower_line:
            unresolved.append(blocker)
            continue
        if blocker.column.lower() not in lower_line:
            unresolved.append(blocker)
            continue
    return unresolved


def build_summary_blocker_follow_up_message(
    blockers: list[ProtocolSummaryBlocker],
) -> str:
    if not blockers:
        return (
            "I did not find any deterministic summary-based blockers in the selected protocol columns."
        )

    lines = [
        "Before I lock this protocol, I need your decision on a few upstream data-preparation issues that are already visible in the selected columns:",
        "",
    ]
    for blocker in blockers:
        lines.append(f"- {blocker.user_question}")
    lines.extend(
        [
            "",
            "Once you answer these, I will record them in the protocol as explicit cleaning instructions for downstream compilation.",
        ]
    )
    return "\n".join(lines)


def _add_baseline_blockers(
    *,
    blockers_add,
    column: str,
    role: Literal["covariate", "effect_modifier"],
    profile: ColumnProfileModel,
) -> None:
    role_name = "effect modifier" if role == "effect_modifier" else "covariate"
    if profile.n_missing > 0:
        blockers_add(
            column,
            role,
            "missing_values",
            f"The baseline {role_name} '{column}' has {profile.n_missing} missing values in the summary. "
            f"How should it be prepared before modeling? If you want imputation, please state that explicitly.",
        )
    if _has_unknown_like_category(profile):
        blockers_add(
            column,
            role,
            "unknown_category_present",
            f"The baseline {role_name} '{column}' includes an unknown-like category. "
            "Should that category be kept as its own level, merged into another group, or handled in some other explicit way?",
        )
    if _is_low_cardinality_numeric(profile):
        blockers_add(
            column,
            role,
            "coded_categories_need_decision",
            f"The baseline {role_name} '{column}' is numeric but has only {profile.distinct_count} distinct values, "
            "so it may represent coded categories. Should it be treated as coded categories, or should it remain numeric?",
        )


def _needs_explicit_outcome_mapping(profile: ColumnProfileModel) -> bool:
    distinct_count = profile.distinct_count
    if distinct_count is None:
        return False
    if isinstance(profile, BooleanColumnProfileModel):
        return False
    if isinstance(profile, CategoricalColumnProfileModel):
        return distinct_count != 2
    if isinstance(profile, NumericColumnProfileModel):
        return 2 < distinct_count <= 10
    return False


def _has_unknown_like_category(profile: ColumnProfileModel) -> bool:
    values: list[str] = []
    if isinstance(profile, CategoricalColumnProfileModel):
        values = [str(item.value) for item in profile.summary.top_categories]
    elif isinstance(profile, BooleanColumnProfileModel):
        values = list(profile.summary.counts.keys())
    else:
        return False

    return any(_looks_unknown_like(value) for value in values)


def _is_low_cardinality_numeric(profile: ColumnProfileModel) -> bool:
    return (
        isinstance(profile, NumericColumnProfileModel)
        and profile.distinct_count is not None
        and 2 <= profile.distinct_count <= 10
    )


def _looks_unknown_like(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    tokens = {
        "unknown",
        "other",
        "other/unknown",
        "unk",
        "missing",
        "na",
        "n/a",
        "none",
        "not recorded",
        "not known",
    }
    return normalized in tokens


def _find_protocol_line(protocol_discussion: str, prefix: str) -> str | None:
    for line in protocol_discussion.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped
    return None


__all__ = [
    "ProtocolSummaryBlocker",
    "build_summary_blocker_follow_up_message",
    "scan_protocol_summary_blockers",
    "unresolved_summary_blockers",
]
