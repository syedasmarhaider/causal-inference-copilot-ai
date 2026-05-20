from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from python.domain.models.models import ChatMessage
from python.domain.service.llm_service import AvailableModelsKey, LLMConfig
from python.implementation.service.llms.litellm_llm_service import (
    LiteLLMService,
    Provider,
    ReliabilityPolicy,
)
from python.implementation.service.llms.llm_service_factory import (
    LLMServiceSettings,
    make_llm_service,
)


@dataclass
class _Message:
    content: Any
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str | None = None


@dataclass
class _Response:
    choices: list[_Choice]


@dataclass
class _DirectContentResponse:
    content: Any


@dataclass
class _CompletionStub:
    responses: list[Any] = field(default_factory=list)
    errors: list[Exception] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        if not self.responses:
            raise AssertionError("No fake response configured")
        return self.responses.pop(0)


class _PayloadModel(BaseModel):
    answer: str


class _StructuredPayloadModel(BaseModel):
    answer: str
    provider: str


def _build_service(
    completion_stub: _CompletionStub,
    *,
    provider: Provider = "vertex_ai",
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
    model_names: dict[AvailableModelsKey, str] | None = None,
    reliability: ReliabilityPolicy | None = None,
) -> LiteLLMService:
    aliases: tuple[AvailableModelsKey, ...] = ("mini", "basic", "pro", "thinking")
    resolved_model_names = model_names or {alias: f"fake-{alias}" for alias in aliases}
    resolved_reliability = reliability or ReliabilityPolicy(
        timeout_s=3.0,
        hard_deadline_s=None,
        max_retries=1,
        base_backoff_s=0.001,
        executor_workers=1,
    )
    return LiteLLMService(
        provider=provider,
        model_names=resolved_model_names,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        reliability=resolved_reliability,
        completion_fn=completion_stub,
    )


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_API_BASE",
        "LLM_API_VERSION",
        "LLM_MODEL_MINI",
        "LLM_MODEL_BASIC",
        "LLM_MODEL_PRO",
        "LLM_MODEL_THINKING",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_AI_STUDIO_API_KEY",
        "AZURE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_API_VERSION",
        "OPENAI_API_BASE",
        "OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def test_generate_builds_messages_and_uses_vertex_ai_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERTEXAI_PROJECT", "project-x")
    stub = _CompletionStub(
        responses=[
            _Response(
                choices=[
                    _Choice(message=_Message(content="  final answer  "), finish_reason="stop")
                ]
            )
        ]
    )
    service = _build_service(stub)

    response = service.generate(
        system_prompt="global-system",
        user_prompt="current-request",
        config=LLMConfig(
            model="basic",
            temperature=0.3,
            top_p=0.8,
            max_tokens=64,
            stop=["END"],
            extra={"foo": "bar"},
        ),
        history=[
            ChatMessage(role="user", content="past-user"),
            ChatMessage(role="assistant", content="past-assistant"),
            ChatMessage(role="system", content="past-system"),
        ],
    )

    assert response.content == "final answer"
    assert len(stub.calls) == 1

    kwargs = stub.calls[0]
    assert kwargs["model"] == "vertex_ai/fake-basic"
    assert kwargs["vertex_project"] == "project-x"
    assert kwargs["vertex_location"] == "global"
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.8
    assert kwargs["stop"] == ["END"]
    assert kwargs["foo"] == "bar"
    assert kwargs["max_tokens"] == 64
    assert kwargs["messages"] == [
        {"role": "system", "content": "global-system"},
        {"role": "user", "content": "past-user"},
        {"role": "assistant", "content": "past-assistant"},
        {"role": "system", "content": "past-system"},
        {"role": "user", "content": "current-request"},
    ]


def test_generate_builds_vertex_ai_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEXAI_PROJECT", "project-x")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="vertex-ok"))])]
    )
    service = _build_service(
        stub,
        provider="vertex_ai",
        model_names={
            "mini": "gemini/fake-mini",
            "basic": "gemini/fake-basic",
            "pro": "gemini/fake-pro",
            "thinking": "gemini/fake-thinking",
        },
    )

    response = service.generate(
        system_prompt=None,
        user_prompt="hello",
        config=LLMConfig(model="basic"),
        history=None,
    )

    assert response.content == "vertex-ok"
    kwargs = stub.calls[0]
    assert kwargs["model"] == "vertex_ai/fake-basic"
    assert kwargs["vertex_project"] == "project-x"
    assert kwargs["vertex_location"] == "us-central1"


def test_generate_builds_google_ai_studio_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="google-ok"))])]
    )
    service = _build_service(
        stub,
        provider="google_ai_studio",
        model_names={
            "mini": "models/gemini-fake-mini",
            "basic": "models/gemini-fake-basic",
            "pro": "models/gemini-fake-pro",
            "thinking": "models/gemini-fake-thinking",
        },
    )

    response = service.generate(
        system_prompt=None,
        user_prompt="hello",
        config=LLMConfig(model="basic"),
        history=None,
    )

    assert response.content == "google-ok"
    kwargs = stub.calls[0]
    assert kwargs["model"] == "gemini/gemini-fake-basic"
    assert kwargs["api_key"] == "gemini-secret"


def test_generate_builds_openai_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="openai-ok"))])]
    )
    service = _build_service(
        stub,
        provider="openai",
        model_names={
            "mini": "gpt-fake-mini",
            "basic": "gpt-fake-basic",
            "pro": "gpt-fake-pro",
            "thinking": "gpt-fake-thinking",
        },
    )

    response = service.generate(
        system_prompt=None,
        user_prompt="hello",
        config=LLMConfig(model="basic"),
        history=None,
    )

    assert response.content == "openai-ok"
    kwargs = stub.calls[0]
    assert kwargs["model"] == "openai/gpt-fake-basic"
    assert kwargs["api_key"] == "openai-secret"


def test_generate_builds_azure_route() -> None:
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="azure-ok"))])]
    )
    service = _build_service(
        stub,
        provider="azure",
        api_key="azure-secret",
        api_base="https://example.openai.azure.com",
        api_version="2024-10-21",
        model_names={
            "mini": "fake-mini-deployment",
            "basic": "fake-basic-deployment",
            "pro": "fake-pro-deployment",
            "thinking": "fake-thinking-deployment",
        },
    )

    response = service.generate(
        system_prompt=None,
        user_prompt="hello",
        config=LLMConfig(model="basic"),
        history=None,
    )

    assert response.content == "azure-ok"
    kwargs = stub.calls[0]
    assert kwargs["model"] == "azure/fake-basic-deployment"
    assert kwargs["api_key"] == "azure-secret"
    assert kwargs["api_base"] == "https://example.openai.azure.com"
    assert kwargs["api_version"] == "2024-10-21"


def test_generate_handles_non_choice_return_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEXAI_PROJECT", "project-x")
    stub = _CompletionStub(
        responses=[_DirectContentResponse(content={"text": "ok-from-structured"})]
    )
    service = _build_service(stub)

    response = service.generate(
        system_prompt=None,
        user_prompt="hello",
        config=LLMConfig(model="basic"),
        history=None,
    )

    assert response.content == "ok-from-structured"


def test_generate_retries_transient_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEXAI_PROJECT", "project-x")
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="retry-success"))])],
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


def test_generate_json_repairs_invalid_output_and_uses_deterministic_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VERTEXAI_PROJECT", "project-x")
    stub = _CompletionStub(
        responses=[
            _Response(choices=[_Choice(message=_Message(content="not valid json"))]),
            _Response(choices=[_Choice(message=_Message(content='{"answer": "valid"}'))]),
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

    first_kwargs = stub.calls[0]
    second_kwargs = stub.calls[1]

    assert first_kwargs["temperature"] == 0.7
    assert first_kwargs["top_p"] == 0.2
    assert second_kwargs["temperature"] == 0.0
    assert second_kwargs["top_p"] == 1.0
    assert first_kwargs["messages"][1] == {"role": "assistant", "content": "history"}
    second_prompt = str(second_kwargs["messages"][-1]["content"]).lower()
    assert "previous output did not validate" in second_prompt
    assert "output must be valid json" in second_prompt


def test_generate_json_raises_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEXAI_PROJECT", "project-x")
    stub = _CompletionStub(
        responses=[
            _Response(choices=[_Choice(message=_Message(content="still not json"))]),
            _Response(choices=[_Choice(message=_Message(content="also not json"))]),
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


def test_vertex_ai_requires_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="unused"))])]
    )
    service = _build_service(stub, provider="vertex_ai")

    with pytest.raises(ValueError, match="Missing Vertex project"):
        service.generate(
            system_prompt=None,
            user_prompt="hello",
            config=LLMConfig(model="basic"),
            history=None,
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
def test_constructor_rejects_invalid_reliability(
    policy: ReliabilityPolicy, error_pattern: str
) -> None:
    stub = _CompletionStub()
    aliases: tuple[AvailableModelsKey, ...] = ("mini", "basic", "pro", "thinking")

    with pytest.raises(ValueError, match=error_pattern):
        LiteLLMService(
            provider="vertex_ai",
            model_names={alias: f"fake-{alias}" for alias in aliases},
            reliability=policy,
            completion_fn=stub,
        )


def test_factory_uses_vertex_ai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        LiteLLMService,
        "_load_completion_fn",
        staticmethod(lambda: _CompletionStub()),
    )

    service = make_llm_service(LLMServiceSettings())

    assert isinstance(service, LiteLLMService)
    assert service._provider == "vertex_ai"  # noqa: SLF001
    service.close()


def test_factory_builds_openai_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "openai-secret")
    monkeypatch.setenv("LLM_MODEL_MINI", "gpt-mini")
    monkeypatch.setenv("LLM_MODEL_BASIC", "gpt-basic")
    monkeypatch.setenv("LLM_MODEL_PRO", "gpt-pro")
    monkeypatch.setenv("LLM_MODEL_THINKING", "gpt-thinking")
    monkeypatch.setattr(
        LiteLLMService,
        "_load_completion_fn",
        staticmethod(lambda: _CompletionStub()),
    )

    settings = LLMServiceSettings.from_env()
    service = make_llm_service(settings)

    assert isinstance(service, LiteLLMService)
    assert service._provider == "openai"  # noqa: SLF001
    assert service._api_key == "openai-secret"  # noqa: SLF001
    assert service._model_names["basic"] == "gpt-basic"  # noqa: SLF001
    service.close()


def test_factory_auto_provider_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "google-ai-studio")
    monkeypatch.setenv("LLM_API_KEY", "google-secret")
    monkeypatch.setenv("LLM_MODEL_MINI", "gemini-mini")
    monkeypatch.setenv("LLM_MODEL_BASIC", "gemini-basic")
    monkeypatch.setenv("LLM_MODEL_PRO", "gemini-pro")
    monkeypatch.setenv("LLM_MODEL_THINKING", "gemini-thinking")
    monkeypatch.setattr(
        LiteLLMService,
        "_load_completion_fn",
        staticmethod(lambda: _CompletionStub()),
    )

    service = make_llm_service(LLMServiceSettings(provider="auto"))

    assert isinstance(service, LiteLLMService)
    assert service._provider == "google_ai_studio"  # noqa: SLF001
    assert service._api_key == "google-secret"  # noqa: SLF001
    assert service._model_names["thinking"] == "gemini-thinking"  # noqa: SLF001
    service.close()


def test_factory_requires_non_vertex_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "openai-secret")

    with pytest.raises(ValueError, match="Missing model env vars"):
        LLMServiceSettings.from_env()


@pytest.mark.integration
def test_generate_json_with_live_vertex_ai() -> None:
    if not _env_truthy("RUN_LIVE_LLM_TESTS"):
        pytest.skip("Set RUN_LIVE_LLM_TESTS=true to enable live LLM tests")
    if not os.environ.get("VERTEXAI_PROJECT", "").strip():
        pytest.skip("No Vertex project configured")

    service = LiteLLMService(
        provider="vertex_ai",
        model_names={
            "mini": "gemini-3.1-flash-lite-preview",
            "basic": "gemini-3-flash-preview",
            "pro": "gemini-3-flash-preview",
            "thinking": "gemini-3.1-pro-preview",
        },
        reliability=ReliabilityPolicy(
            timeout_s=30.0,
            hard_deadline_s=60.0,
            max_retries=0,
            executor_workers=1,
        ),
    )

    try:
        response = service.generate_json(
            schema=_StructuredPayloadModel,
            system_prompt=(
                "Return valid JSON only. The answer field must contain the lowercase word pong. "
                "The provider field must echo the provider name you were asked to return."
            ),
            user_prompt='Return JSON with {"answer":"pong","provider":"vertex_ai"} and nothing else.',
            config=LLMConfig(model="mini", temperature=0.0, top_p=1.0, max_tokens=64),
            history=None,
            max_attempts=2,
        )
    finally:
        service.close()

    assert response.answer == "pong"
    assert response.provider == "vertex_ai"
