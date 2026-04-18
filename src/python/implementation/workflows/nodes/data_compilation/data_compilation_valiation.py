from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from python.domain.models.validation import ValidationIssueModel
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
)


@dataclass(frozen=True)
class DataCompilationValidationResult:
    validation_errors: list[ValidationIssueModel]
    user_suggestion_message: str | None


_REPAIRABLE_MARKERS: tuple[str, ...] = (
    "transform plan",
    "transform preset",
    "encoding preset",
    "mapping/order",
    "map_binary",
    "map_ordinal",
    "num_log1p preset",
    "datetime_epoch_seconds preset",
    "unknown-value handling",
)


def validate_data_compilation(
    candidate_df: pd.DataFrame,
    causal_spec: CausalSpec,
    transform_plan: TransformPlan | None,
) -> DataCompilationValidationResult:
    report = ValidationBackdoorTool().validate(
        causal_spec=causal_spec,
        dataframe=candidate_df,
        transform_plan=transform_plan,
    )
    issues = list(report.issues)
    return DataCompilationValidationResult(
        validation_errors=issues,
        user_suggestion_message=_build_user_suggestion_message(
            causal_spec=causal_spec,
            issues=issues,
        ),
    )


def _build_user_suggestion_message(
    *,
    causal_spec: CausalSpec,
    issues: list[ValidationIssueModel],
) -> str | None:
    fail_issues = [issue for issue in issues if issue.severity == "FAIL"]
    if not fail_issues:
        return None

    repairable_fail_issues = [
        issue for issue in fail_issues if _is_repairable_transform_issue(issue)
    ]
    if len(repairable_fail_issues) != len(fail_issues):
        return None

    lines = [
        "Validation found repairable transformation or encoding issues. These can still be addressed without changing locked column identities or roles.",
        "",
        f"Locked treatment column: {causal_spec.treatment_spec.column}",
        f"Locked outcome column: {causal_spec.outcome_spec.column}",
        f"Locked covariates: {', '.join(causal_spec.covariates) if causal_spec.covariates else 'None'}",
        f"Locked effect modifiers: {', '.join(causal_spec.effect_modifiers) if causal_spec.effect_modifiers else 'None'}",
        "",
        "Repairable validation errors:",
    ]

    for issue in repairable_fail_issues:
        lines.append(f"- {issue.message}")
        if issue.fix_hint:
            lines.append(f"  What to fix: {issue.fix_hint}")

    lines.extend(
        [
            "",
            "Next step: revise the transform encodings or transformation choices while keeping the locked treatment, outcome, covariate, and effect-modifier columns unchanged.",
        ]
    )
    return "\n".join(lines)


def _is_repairable_transform_issue(issue: ValidationIssueModel) -> bool:
    haystack = " ".join(
        part.strip()
        for part in [issue.message, issue.fix_hint or ""]
        if part and part.strip()
    ).lower()
    return any(marker in haystack for marker in _REPAIRABLE_MARKERS)


__all__ = [
    "DataCompilationValidationResult",
    "validate_data_compilation",
]
