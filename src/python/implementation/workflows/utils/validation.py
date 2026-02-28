from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.models import NonEmptyStr



ValidationSeverity = Literal["WARN", "FAIL"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]

class ValidationIssueModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    severity: ValidationSeverity
    message: NonEmptyStr
    evidence: Dict[str, Any] = Field(default_factory=dict) # pyright: ignore[reportUnknownVariableType, reportAssignmentType, reportCallIssue]
    fix_hint: Optional[NonEmptyStr] = None