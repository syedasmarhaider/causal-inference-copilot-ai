from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, TypedDict, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Role = Literal["user", "assistant","system"]
AvailableModelsKey = Literal["mini","basic", "pro","thinking"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


ProviderExtra = Dict[str, Any]


@dataclass(frozen=True)
class LLMConfig:
    model: AvailableModelsKey = "basic"
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 60000
    stop: Optional[List[str]] = None
    extra: Optional[ProviderExtra] = None

class ToolCall(TypedDict, total=False):
    id: str
    name: str
    args: Dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    raw: Any = None


class LLMService(Protocol):
    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
    ) -> LLMResponse: ...

    def generate_json(
        self,
        *,
        schema: type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
        max_attempts: int = 3,
    ) -> T: ...
