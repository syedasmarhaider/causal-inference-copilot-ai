from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_questions,
)
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

_QUESTION_LINE_RE = re.compile(r"^(?P<number>\d+)\)")
_ANSWER_LINE_RE = re.compile(r"^A(?P<number>\d+)\)")
_CANONICAL_QUESTION_BY_NUMBER = {
    str(index): question.strip()
    for index, question in enumerate(get_questions(), start=1)
}


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
        answer_text = extract_protocol_answer_text(
            protocol_discussion,
            blocker.protocol_question_prefix,
        )
        if answer_text is None:
            unresolved.append(blocker)
            continue
        lower_answer = answer_text.lower()
        if "unclear" in lower_answer:
            unresolved.append(blocker)
            continue
        if not _answer_resolves_blocker(blocker=blocker, answer_text=lower_answer):
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


def extract_protocol_answer_text(protocol_discussion: str, prefix: str) -> str | None:
    question_number = _question_number_from_prefix(prefix)
    if question_number is None:
        return None

    answer_prefix = f"A{question_number})"
    canonical_question = _CANONICAL_QUESTION_BY_NUMBER.get(question_number)
    lines = [line.strip() for line in protocol_discussion.splitlines()]

    for index, line in enumerate(lines):
        if line.startswith(answer_prefix):
            return _collect_answer_block(
                lines=lines,
                start_index=index,
                question_number=question_number,
                include_start=True,
            )

    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        if canonical_question is not None and line == canonical_question:
            continue
        return _collect_answer_block(
            lines=lines,
            start_index=index,
            question_number=question_number,
            include_start=True,
        )

    if canonical_question is None:
        return None

    for index, line in enumerate(lines):
        if line != canonical_question:
            continue
        return _collect_answer_block(
            lines=lines,
            start_index=index + 1,
            question_number=question_number,
            include_start=False,
        )

    return None


def _answer_resolves_blocker(
    *,
    blocker: ProtocolSummaryBlocker,
    answer_text: str,
) -> bool:
    if blocker.column.lower() in answer_text:
        return True

    if blocker.role == "treatment":
        role_tokens = ("treatment", "exposure", "treated", "control")
    elif blocker.role == "outcome":
        role_tokens = ("outcome", "endpoint")
    else:
        role_tokens = ()

    if blocker.issue == "missing_values":
        return bool(role_tokens) and any(token in answer_text for token in role_tokens) and any(
            token in answer_text for token in ("missing", "drop", "exclude", "imput", "keep")
        )

    if blocker.issue == "outcome_mapping_required":
        return bool(role_tokens) and any(token in answer_text for token in role_tokens) and any(
            token in answer_text
            for token in ("map", "mapping", "binary", "success", "failure")
        )

    if blocker.issue == "treatment_not_binary":
        return bool(role_tokens) and any(token in answer_text for token in role_tokens) and any(
            token in answer_text
            for token in ("binary", "treated", "control", "map", "keep")
        )

    if blocker.issue == "unknown_category_present":
        return blocker.role in {"covariate", "effect_modifier"} and _baseline_answer_resolves_unknown_category(
            answer_text
        )

    if blocker.issue == "coded_categories_need_decision":
        return blocker.role in {"covariate", "effect_modifier"} and any(
            token in answer_text
            for token in (
                "baseline",
                "covariate",
                "covariates",
                "effect modifier",
                "effect modifiers",
            )
        ) and any(token in answer_text for token in ("coded", "categor", "numeric", "remain numeric"))

    return False


def _baseline_answer_resolves_unknown_category(answer_text: str) -> bool:
    baseline_scope = any(
        token in answer_text
        for token in (
            "baseline",
            "covariate",
            "covariates",
            "effect modifier",
            "effect modifiers",
        )
    )
    if not baseline_scope:
        return False

    mentions_unknown = any(
        token in answer_text
        for token in ("unknown", "other/unknown", "unk", "missing", "not known", "not recorded")
    )
    if not mentions_unknown:
        return False

    return any(
        token in answer_text
        for token in (
            "distinct category",
            "distinct level",
            "own category",
            "own level",
            "separate category",
            "separate level",
            "keep",
            "kept",
            "merge",
            "merged",
            "imput",
            "drop",
            "exclude",
        )
    )


def _question_number_from_prefix(prefix: str) -> str | None:
    match = _QUESTION_LINE_RE.match(prefix)
    if match is None:
        return None
    return str(match.group("number"))


def _collect_answer_block(
    *,
    lines: list[str],
    start_index: int,
    question_number: str,
    include_start: bool,
) -> str | None:
    collected: list[str] = []
    if include_start and start_index < len(lines):
        start_line = lines[start_index].strip()
        if start_line:
            collected.append(start_line)

    for line in lines[start_index + (1 if include_start else 0) :]:
        stripped = line.strip()
        if not stripped:
            continue

        question_match = _QUESTION_LINE_RE.match(stripped)
        if question_match is not None and question_match.group("number") != question_number:
            break

        answer_match = _ANSWER_LINE_RE.match(stripped)
        if answer_match is not None and answer_match.group("number") != question_number:
            break

        collected.append(stripped)

    if not collected:
        return None

    normalized = " ".join(collected).strip()
    return normalized or None


__all__ = [
    "ProtocolSummaryBlocker",
    "build_summary_blocker_follow_up_message",
    "extract_protocol_answer_text",
    "scan_protocol_summary_blockers",
    "unresolved_summary_blockers",
]
