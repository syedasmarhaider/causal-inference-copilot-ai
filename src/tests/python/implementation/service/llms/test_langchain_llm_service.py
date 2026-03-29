from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from python.domain.service.llm_service import AvailableModelsKey, ChatMessage, LLMConfig
from python.implementation.service.llms.langchain_llm_service import (
    LangChainLLMService,
    ReliabilityPolicy,
)
from python.implementation.service.llms.llm_service_factory import (
    LLMServiceSettings,
    make_llm_service,
)


@dataclass
class _InvokeStub:
    responses: list[Any] = field(default_factory=list)
    errors: list[Exception] = field(default_factory=list)
    calls: list[tuple[list[BaseMessage], dict[str, Any]]] = field(default_factory=list)

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        if self.errors:
            raise self.errors.pop(0)
        if not self.responses:
            raise AssertionError("No fake response configured")
        return self.responses.pop(0)


class _StructuredReply:
    def __init__(self, content: Any) -> None:
        self.content = content


class _PayloadModel(BaseModel):
    answer: str


def _build_service(
    model: _InvokeStub,
    *,
    max_tokens_param_name: str = "max_output_tokens",
    reliability: ReliabilityPolicy | None = None,
) -> LangChainLLMService:
    aliases: tuple[AvailableModelsKey, ...] = ("mini", "basic", "pro", "thinking")
    rel = reliability or ReliabilityPolicy(
        timeout_s=3.0,
        hard_deadline_s=None,
        max_retries=1,
        base_backoff_s=0.001,
        executor_workers=1,
    )
    return LangChainLLMService(
        models={alias: model for alias in aliases},
        model_names={alias: f"fake-{alias}" for alias in aliases},
        max_tokens_param_name=max_tokens_param_name,
        reliability=rel,
    )


def test_generate_builds_messages_and_sanitizes_max_output_tokens() -> None:
    stub = _InvokeStub(responses=[AIMessage(content="  final answer  ")])
    service = _build_service(stub, max_tokens_param_name="max_output_tokens")

    response = service.generate(
        system_prompt="global-system",
        user_prompt="current-request",
        config=LLMConfig(
            model="basic",
            temperature=0.3,
            top_p=0.8,
            max_tokens=64,
            stop=["END"],
            extra={"max_tokens": 999, "foo": "bar"},
        ),
        history=[
            ChatMessage(role="user", content="past-user"),
            ChatMessage(role="assistant", content="past-assistant"),
            ChatMessage(role="system", content="past-system"),
        ],
    )

    assert response.content == "final answer"
    assert len(stub.calls) == 1

    messages, kwargs = stub.calls[0]
    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
        HumanMessage,
    ]
    assert [message.content for message in messages] == [
        "global-system",
        "past-user",
        "past-assistant",
        "past-system",
        "current-request",
    ]
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.8
    assert kwargs["stop"] == ["END"]
    assert kwargs["foo"] == "bar"
    assert kwargs["max_output_tokens"] == 64
    assert "max_tokens" not in kwargs


def test_generate_handles_non_ai_message_return_shape() -> None:
    stub = _InvokeStub(responses=[_StructuredReply(content={"text": "ok-from-structured"})])
    service = _build_service(stub)

    response = service.generate(
        system_prompt=None,
        user_prompt="hello",
        config=LLMConfig(model="basic"),
        history=None,
    )

    assert response.content == "ok-from-structured"


def test_invoke_kwargs_removes_openai_style_token_key_for_gemini_mode() -> None:
    stub = _InvokeStub()
    service = _build_service(stub, max_tokens_param_name="max_output_tokens")

    kwargs = service._invoke_kwargs(  # noqa: SLF001 - testing service internals directly
        LLMConfig(
            model="basic",
            max_tokens=50,
            extra={"max_tokens": 200, "top_k": 40},
        )
    )

    assert kwargs["max_output_tokens"] == 50
    assert kwargs["top_k"] == 40
    assert "max_tokens" not in kwargs


def test_invoke_kwargs_removes_gemini_style_token_key_for_openai_mode() -> None:
    stub = _InvokeStub()
    service = _build_service(stub, max_tokens_param_name="max_tokens")

    kwargs = service._invoke_kwargs(  # noqa: SLF001 - testing service internals directly
        LLMConfig(
            model="basic",
            max_tokens=50,
            extra={"max_output_tokens": 200, "presence_penalty": 0.5},
        )
    )

    assert kwargs["max_tokens"] == 50
    assert kwargs["presence_penalty"] == 0.5
    assert "max_output_tokens" not in kwargs


def test_generate_retries_transient_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _InvokeStub(
        responses=[AIMessage(content="retry-success")],
        errors=[TimeoutError("timed out")],
    )
    service = _build_service(
        stub,
        reliability=ReliabilityPolicy(
            timeout_s=3.0,
            hard_deadline_s=None,
            max_retries=1,
            base_backoff_s=0.001,
            executor_workers=1,
        ),
    )

    attempts: list[int] = []
    monkeypatch.setattr(service, "_sleep_backoff", lambda attempt: attempts.append(attempt))

    response = service.generate(
        system_prompt=None,
        user_prompt="hello",
        config=LLMConfig(model="basic"),
        history=None,
    )

    assert response.content == "retry-success"
    assert attempts == [0]
    assert len(stub.calls) == 2


def test_generate_json_repairs_invalid_output_and_uses_deterministic_retry() -> None:
    stub = _InvokeStub(
        responses=[
            AIMessage(content="not valid json"),
            AIMessage(content='{"answer": "valid"}'),
        ]
    )
    service = _build_service(stub)

    parsed = service.generate_json(
        schema=_PayloadModel,
        system_prompt="json-system",
        user_prompt="return an object",
        config=LLMConfig(model="basic", temperature=0.7, top_p=0.2, max_tokens=80),
        history=[ChatMessage(role="assistant", content="history")],
        max_attempts=2,
    )

    assert parsed.answer == "valid"
    assert len(stub.calls) == 2

    first_messages, first_kwargs = stub.calls[0]
    second_messages, second_kwargs = stub.calls[1]

    assert first_kwargs["temperature"] == 0.7
    assert first_kwargs["top_p"] == 0.2
    assert second_kwargs["temperature"] == 0.0
    assert second_kwargs["top_p"] == 1.0

    assert isinstance(first_messages[1], AIMessage)
    assert isinstance(second_messages[-1], HumanMessage)
    second_prompt = str(second_messages[-1].content).lower()
    assert "previous output did not validate" in second_prompt
    assert "output must be valid json" in second_prompt


def test_generate_json_raises_after_max_attempts() -> None:
    stub = _InvokeStub(
        responses=[
            AIMessage(content="still not json"),
            AIMessage(content="also not json"),
        ]
    )
    service = _build_service(stub)

    with pytest.raises(RuntimeError, match=r"Failed JSON schema=_PayloadModel after 2 attempts"):
        service.generate_json(
            schema=_PayloadModel,
            system_prompt=None,
            user_prompt="return payload",
            config=LLMConfig(model="basic"),
            history=None,
            max_attempts=2,
        )


@pytest.mark.parametrize(
    ("policy", "error_pattern"),
    [
        (ReliabilityPolicy(timeout_s=0), r"timeout_s"),
        (ReliabilityPolicy(timeout_s=2, hard_deadline_s=1), r"hard_deadline_s must be >= timeout_s"),
        (ReliabilityPolicy(max_retries=-1), r"max_retries"),
        (ReliabilityPolicy(executor_workers=0), r"executor_workers"),
    ],
)
def test_constructor_rejects_invalid_reliability(policy: ReliabilityPolicy, error_pattern: str) -> None:
    stub = _InvokeStub()
    aliases: tuple[AvailableModelsKey, ...] = ("mini", "basic", "pro", "thinking")

    with pytest.raises(ValueError, match=error_pattern):
        LangChainLLMService(
            models={alias: stub for alias in aliases},
            model_names={alias: f"fake-{alias}" for alias in aliases},
            max_tokens_param_name="max_output_tokens",
            reliability=policy,
        )


@pytest.mark.integration
def test_generate_with_real_gemini_api_key() -> None:
    if any(
        os.environ.get(key, "").strip().lower() not in {"", "0", "false", "no", "off"}
        for key in ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "TF_BUILD", "JENKINS_URL")
    ):
        pytest.skip("Integration test is skipped in CI/pipeline environments")

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured")

    service = make_llm_service(
        LLMServiceSettings(
            timeout_s=30.0,
            hard_deadline_s=60.0,
            max_retries=0,
            executor_workers=1,
        )
    )

    try:
        response = service.generate(
            system_prompt="Respond with the single word PONG.",
            user_prompt="Ping",
            config=LLMConfig(model="mini", temperature=0.0, top_p=1.0, max_tokens=16),
            history=None,
        )
    finally:
        close = getattr(service, "close", None)
        if callable(close):
            close()

    assert response.content
    assert "pong" in response.content.lower()
