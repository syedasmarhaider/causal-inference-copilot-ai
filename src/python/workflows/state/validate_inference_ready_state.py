from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

ValidationSeverity = Literal["FAIL", "WARN"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]


class InferenceReadyValidationIssue(TypedDict):
    rule_id: str
    severity: ValidationSeverity
    message: str
    evidence: Dict[str, Any]
    fix_hint: str | None


class InferenceReadyValidationReport(TypedDict):
    status: ValidationStatus
    issues: List[InferenceReadyValidationIssue]
    metrics: Dict[str, Any]


class InferenceReadyValidationState(TypedDict, total=False):
    report: InferenceReadyValidationReport

