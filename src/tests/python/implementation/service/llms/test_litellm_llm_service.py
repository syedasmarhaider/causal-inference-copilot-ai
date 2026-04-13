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
    provider: Provider = "gemini",
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
        reliability=resolved_reliability,
        completion_fn=completion_stub,
    )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def test_generate_builds_messages_and_uses_gemini_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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
    assert kwargs["model"] == "gemini/fake-basic"
    assert kwargs["api_key"] == "test-key"
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


def test_generate_builds_vertex_api_key_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEX_AI_API_KEY", "vertex-key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "project-x")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="vertex-ok"))])]
    )
    service = _build_service(
        stub,
        provider="vertex_api",
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
    assert kwargs["model"] == "fake-basic"
    assert kwargs["custom_llm_provider"] == "gemini"
    assert kwargs["api_key"] == "vertex-key"
    assert kwargs["gemini_api_key"] == "vertex-key"
    assert (
        kwargs["api_base"]
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/project-x/locations/us-central1/publishers/google"
    )


def test_generate_handles_non_choice_return_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
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


def test_vertex_api_requires_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEX_AI_API_KEY", "vertex-key")
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT_ID", raising=False)
    stub = _CompletionStub(
        responses=[_Response(choices=[_Choice(message=_Message(content="unused"))])]
    )
    service = _build_service(stub, provider="vertex_api")

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
            provider="gemini",
            model_names={alias: f"fake-{alias}" for alias in aliases},
            reliability=policy,
            completion_fn=stub,
        )


def test_factory_prefers_gemini_keys_over_vertex_in_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LITELLM_PROVIDER", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("VERTEX_AI_API_KEY", "vertex-key")
    monkeypatch.setattr(
        LiteLLMService,
        "_load_completion_fn",
        staticmethod(lambda: _CompletionStub()),
    )

    service = make_llm_service(LLMServiceSettings())

    assert isinstance(service, LiteLLMService)
    assert service._provider == "gemini"  # noqa: SLF001
    service.close()


def test_factory_auto_detects_vertex_api_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("VERTEX_AI_API_KEY", "vertex-key")
    monkeypatch.setattr(
        LiteLLMService,
        "_load_completion_fn",
        staticmethod(lambda: _CompletionStub()),
    )

    service = make_llm_service(LLMServiceSettings())

    assert isinstance(service, LiteLLMService)
    assert service._provider == "vertex_api"  # noqa: SLF001
    service.close()


def test_factory_maps_legacy_vertex_ai_provider_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_PROVIDER", "vertex_ai")
    monkeypatch.setattr(
        LiteLLMService,
        "_load_completion_fn",
        staticmethod(lambda: _CompletionStub()),
    )

    service = make_llm_service(LLMServiceSettings())

    assert isinstance(service, LiteLLMService)
    assert service._provider == "vertex_api"  # noqa: SLF001
    service.close()


@pytest.mark.integration
def test_generate_with_live_gemini_key() -> None:
    if not _env_truthy("RUN_LIVE_LLM_TESTS"):
        pytest.skip("Set RUN_LIVE_LLM_TESTS=true to enable live LLM tests")
    if not any(os.environ.get(key, "").strip() for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY")):
        pytest.skip("No Gemini API key configured")

    service = LiteLLMService(
        provider="gemini",
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
        response = service.generate(
            system_prompt="Respond with the single word PONG.",
            user_prompt="Ping",
            config=LLMConfig(model="mini", temperature=0.0, top_p=1.0, max_tokens=16),
            history=None,
        )
    finally:
        service.close()

    assert response.content
    assert "pong" in response.content.lower()


@pytest.mark.integration
def test_generate_json_with_live_vertex_api_key() -> None:
    if not _env_truthy("RUN_LIVE_LLM_TESTS"):
        pytest.skip("Set RUN_LIVE_LLM_TESTS=true to enable live LLM tests")
    if not os.environ.get("VERTEX_AI_API_KEY", "").strip():
        pytest.skip("No Vertex API key configured")
    if not any(
        os.environ.get(key, "").strip()
        for key in ("VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT_ID")
    ):
        pytest.skip("No Vertex project configured")

    service = LiteLLMService(
        provider="vertex_api",
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
            user_prompt='Return JSON with {"answer":"pong","provider":"vertex_api"} and nothing else.',
            config=LLMConfig(model="mini", temperature=0.0, top_p=1.0, max_tokens=64),
            history=None,
            max_attempts=2,
        )
    finally:
        service.close()

    assert response.answer == "pong"
    assert response.provider == "vertex_api"
