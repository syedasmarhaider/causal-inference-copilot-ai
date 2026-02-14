from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

CmdType = Literal["FIT", "EFFECT", "INTERVAL"]

@dataclass(frozen=True)
class CausalCommand:
    cmd: CmdType
    estimator_fqcn: str
    dataset_id: UUID
    inputs: Dict[str, Any]                 # "directory": Y,T,X,W,Z,Xq,...
    options: Dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType] # init kwargs, inference, etc.
    meta: Dict[str, Any] = field(default_factory=dict)     # pyright: ignore[reportUnknownVariableType] # run_id, schema_fingerprint, provenance

Status = Literal["OK", "NEEDS_INPUT", "INVALID", "UNSUPPORTED", "ERROR"]

@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    path: str                     # e.g. "inputs.X" or "options.cv"
    fix_hint: Optional[str] = None
    required: Optional[list[str]] = None   # e.g. ["inputs.X", "inputs.T"]


@dataclass(frozen=True)
class FieldSpec:
    path: str                   # e.g. "inputs.Y" or "options.init.model_y"
    prompt: str                 # what LLM should ask user
    required: bool
    default: Optional[Any] = None
    derived_from_data: bool = False
    notes: Optional[str] = None

@dataclass(frozen=True)
class InputContract:
    cmd: CmdType
    ask_user: List[FieldSpec]           # ONLY these are user questions
    optional_user: List[FieldSpec]
    derived: List[FieldSpec]            # set by profiling (dtype, uniques, etc.)
    defaults: Dict[str, Any]            # server-applied defaults (non-questions)