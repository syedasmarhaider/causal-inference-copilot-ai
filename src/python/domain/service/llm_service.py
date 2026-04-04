from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel
from typing_extensions import TypedDict

from python.domain.models.models import ChatMessage


T = TypeVar("T", bound=BaseModel)
AvailableModelsKey = Literal["mini","basic", "pro","thinking"]
ProviderExtra = dict[str, Any]



@dataclass(frozen=True)
class LLMConfig:
    model: AvailableModelsKey = "basic"
    temperature: float | None = 0.2
    top_p: float | None = 0.95
    max_tokens: int | None = 60000
    stop: list[str] | None = None
    extra: ProviderExtra | None = None

class ToolCall(TypedDict, total=False):
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw: Any = None


class LLMService(Protocol):
    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse: ...

    def generate_json(
        self,
        *,
        schema: type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> T: ...
