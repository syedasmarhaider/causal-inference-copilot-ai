from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.models import NonEmptyStr

ValidationSeverity = Literal["WARN", "FAIL"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]

class ValidationIssueModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    severity: ValidationSeverity
    message: NonEmptyStr
    evidence: dict[str, Any] = Field(default_factory=dict) # pyright: ignore[reportUnknownVariableType, reportAssignmentType, reportCallIssue]
    fix_hint: NonEmptyStr | None = None