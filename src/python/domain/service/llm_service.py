from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

# ===== JSON-like value used for provider-specific "extra" knobs =====
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
ProviderExtra: TypeAlias = Mapping[str, JSONValue]



# ===== Domain types =====
Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None



@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    type: str  # e.g., "function"
    arguments: Mapping[str, JSONValue]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None = None
    usage: Usage = Usage()
    tool_calls: list[ToolCall] | None = None
    raw: object | None = None  # provider-specific payload (opaque)


@dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float = 0.5
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None


# ===== Interfaces (ABCs) =====


class LLMService(ABC):
    @abstractmethod
    def generate(self, 
                 *, 
                 config: LLMConfig,
                 system_prompt: str,
                 user_prompt: str,
                 history: Sequence[ChatMessage] | None) -> LLMResponse: ...


class ProviderNotFound(Exception): ...
