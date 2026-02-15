from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from python.workflows.tools.inference.models.causal_command import CmdType, Issue, Status

@dataclass(frozen=True)
class CausalResult:
    status: Status
    model_id: Optional[UUID] = None
    outputs: Dict[str, Any] = field(default_factory=dict) # pyright: ignore[reportUnknownVariableType]
    issues: List[Issue] = field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    meta: Dict[str, Any] = field(default_factory=dict) # pyright: ignore[reportUnknownVariableType]

@dataclass(frozen=True)
class OutputContract:
    cmd: CmdType
    outputs: Dict[str, str]             # name -> description/type
    errors: List[str]                   # typical failure modes / statuses