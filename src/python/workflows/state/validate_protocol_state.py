from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

ValidationSeverity = Literal["FAIL", "WARN"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]


class ProtocolValidationIssue(TypedDict):
    rule_id: str
    severity: ValidationSeverity
    message: str
    evidence: Dict[str, Any]
    fix_hint: str | None


class ProtocolValidationReport(TypedDict):
    status: ValidationStatus
    issues: List[ProtocolValidationIssue]
    metrics: Dict[str, Any]


class ProtocolStaticValidationState(TypedDict, total=False):
    report: ProtocolValidationReport
