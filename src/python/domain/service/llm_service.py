from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, Optional, Sequence, TypeAlias, Union

# ===== JSON-like value used for provider-specific "extra" knobs =====
JSONScalar: TypeAlias = Union[str, int, float, bool, None]
JSONValue: TypeAlias = Union[JSONScalar, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
ProviderExtra: TypeAlias = Mapping[str, JSONValue]

def _empty_extra() -> Dict[str, JSONValue]:
    # Typed empty dict so Pylance doesn’t complain about default_factory=dict
    return {}

# ===== Domain types =====
Role = Literal["system", "user", "assistant", "tool"]

@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    name: Optional[str] = None

@dataclass(frozen=True)
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

@dataclass(frozen=True)
class ToolCall:
    id: Optional[str]
    type: str                      # e.g., "function"
    arguments: Mapping[str, JSONValue]

@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Usage = Usage()
    tool_calls: Optional[List[ToolCall]] = None
    raw: Optional[object] = None   # provider-specific payload (opaque)

@dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    extra: ProviderExtra = field(default_factory=_empty_extra)

# ===== Interfaces (ABCs) =====

class LLMService(ABC):
    @abstractmethod
    def generate(self, *, config: LLMConfig, history: Sequence[ChatMessage]) -> LLMResponse:
        ...

class ProviderNotFound(Exception):
    ...
