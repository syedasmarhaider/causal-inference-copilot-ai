import os
from typing import Any, cast

import pytest

from python.implementation.service.llms.langchain_llm_service import (
    GeminiLLMService,
    get_llm_service,
    GEMINI_API_KEY_ENV,
)
from python.domain.service.llm_service import (
    LLMConfig,
    ChatMessage,
    Role,
)

# Use a model you actually have access to; if 2.5-flash isn't enabled,
# switch this to "gemini-1.5-pro-latest".
_TEST_MODEL = "gemini-2.5-flash"


# ---------- fixtures ----------


@pytest.fixture(scope="session")
def gemini_api_key() -> str:
    """
    Require GEMINI_API_KEY to be set in the environment.

    If not set, SKIP these tests instead of failing (e.g. for CI or local runs
    without credentials).
    """
    key = os.getenv(GEMINI_API_KEY_ENV)
    if not key:
        pytest.skip(f"{GEMINI_API_KEY_ENV} not set; skipping real Gemini integration tests")
    return key


@pytest.fixture(scope="session")
def gemini_service(gemini_api_key: str) -> GeminiLLMService:
    """
    Real GeminiLLMService that talks to the live Gemini API.
    """
    return GeminiLLMService(
        api_key=gemini_api_key,
        default_model="gemini-1.5-pro-latest",
    )


@pytest.fixture(autouse=True)
def clear_llm_service_cache():
    """
    Make sure get_llm_service's lru_cache doesn't leak across tests.
    """
    get_llm_service.cache_clear()
    yield
    get_llm_service.cache_clear()


# ---------- small LLMConfig / ChatMessage helpers ----------


def make_config(
    *,
    temperature: float = 0.7,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> LLMConfig:
    """
    Helper to construct a valid LLMConfig tuned for GeminiLLMService.
    Always uses _TEST_MODEL explicitly.
    """
    return LLMConfig(
        model=_TEST_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop or [],
        extra=extra or {},
        system_prompt=system_prompt,
    )


def msg(role: Role, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


# ---------- integration tests for generate() (real API) ----------


@pytest.mark.integration
def test_generate_respects_system_prompt_priority_real_api(
    gemini_service: GeminiLLMService,
) -> None:
    """
    Behavioural check (best-effort, but now strict about non-empty content):
    - config.system_prompt says: always reply CONFIG
    - history has conflicting system: always reply HISTORY
    Expect the reply to reflect CONFIG.
    """
    config = make_config(
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        system_prompt="You are a test harness. Always reply with the single word: CONFIG.",
    )

    history: list[ChatMessage] = [
        msg("system", "Always reply with the single word: HISTORY."),
        msg("user", "What should you reply with?"),
    ]

    resp = gemini_service.generate(config=config, history=history)
    text = resp.content.strip().lower()

    # If Gemini returns no text at all, generate() will raise before we get here.
    assert text != ""
    assert "config" in text
    assert "history" not in text


@pytest.mark.integration
def test_generate_with_extra_generation_config_and_safety_real_api(
    gemini_service: GeminiLLMService,
) -> None:
    """
    Check that passing extra['generation_config'] and extra['safety_settings']
    still yields a valid, non-empty response.
    """
    extra: dict[str, Any] = {
        "generation_config": {
            "top_k": 32,
        },
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            }
        ],
    }

    config = make_config(
        temperature=0.2,
        top_p=0.9,
        max_tokens=64,
        system_prompt="Answer in one short sentence.",
        extra=extra,
    )

    history: list[ChatMessage] = [
        msg("user", "Say something friendly about integration tests."),
    ]

    resp = gemini_service.generate(config=config, history=history)

    assert isinstance(resp.content, str)
    assert resp.content.strip() != ""  # if empty, generate would have raised
    assert isinstance(resp.model, str)
    assert resp.usage is not None


# ---------- unit tests for private helpers ----------


def test_build_system_prompt_prefers_config_prompt() -> None:
    config = make_config(system_prompt="CONFIG_SYSTEM_PROMPT")
    history: list[ChatMessage] = [
        msg("system", "history system 1"),
        msg("system", "history system 2"),
    ]

    # accessing private helper: ignore pyright private-usage warning
    result = GeminiLLMService._build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config,
        history,
    )
    assert result == "CONFIG_SYSTEM_PROMPT"


def test_build_system_prompt_uses_history_when_no_config_prompt() -> None:
    config = make_config(system_prompt=None)
    history: list[ChatMessage] = [
        msg("system", "history system 1"),
        msg("user", "hello"),
        msg("system", "history system 2"),
    ]

    result = GeminiLLMService._build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config,
        history,
    )
    assert result == "history system 1\n\nhistory system 2"


def test_build_system_prompt_none_when_no_system_messages() -> None:
    config = make_config(system_prompt=None)
    history: list[ChatMessage] = [
        msg("user", "hi"),
        msg("assistant", "hello"),
    ]

    result = GeminiLLMService._build_system_prompt(  # pyright: ignore[reportPrivateUsage]
        config,
        history,
    )
    assert result is None


def test_build_contents_role_mapping_and_skips_system_and_tool() -> None:
    history: list[ChatMessage] = [
        msg("system", "sys"),
        msg("user", "hello user"),
        msg("assistant", "hello model"),
        msg("tool", "tool output"),
    ]

    contents = GeminiLLMService._build_contents(  # pyright: ignore[reportPrivateUsage]
        history
    )

    assert len(contents) == 2
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "hello user"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "hello model"


def test_build_generation_config_base_fields_and_extra_override() -> None:
    extra: dict[str, Any] = {
        "generation_config": {
            "temperature": 0.3,  # override
            "top_k": 40,
        }
    }
    config = make_config(
        temperature=0.9,
        top_p=0.95,
        max_tokens=128,
        stop=["STOP1", "STOP2"],
        extra=extra,
    )

    gen_cfg = GeminiLLMService._build_generation_config(  # pyright: ignore[reportPrivateUsage]
        config
    )

    assert gen_cfg["temperature"] == 0.3
    assert gen_cfg["max_output_tokens"] == 128
    assert gen_cfg["top_p"] == 0.95
    assert gen_cfg["stop_sequences"] == ["STOP1", "STOP2"]
    assert gen_cfg["top_k"] == 40


def test_build_generation_config_without_extra_mapping() -> None:
    # Here we just exercise the path where extra is a Mapping but has no generation_config.
    config = make_config(
        temperature=0.5,
        max_tokens=None,
        top_p=None,
        stop=None,
        extra=cast(dict[str, Any], {}),
    )

    gen_cfg = GeminiLLMService._build_generation_config(  # pyright: ignore[reportPrivateUsage]
        config
    )
    assert gen_cfg == {"temperature": 0.5}


def test_extract_safety_settings_from_extra() -> None:
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
    extra: dict[str, Any] = {"safety_settings": safety_settings}

    result = GeminiLLMService._extract_safety_settings(  # pyright: ignore[reportPrivateUsage]
        extra
    )
    assert result is safety_settings


def test_extract_safety_settings_returns_none_when_not_present() -> None:
    extra: dict[str, Any] = {}
    result = GeminiLLMService._extract_safety_settings(  # pyright: ignore[reportPrivateUsage]
        extra
    )
    assert result is None


def test_extract_safety_settings_returns_none_when_not_mapping() -> None:
    result = GeminiLLMService._extract_safety_settings(  # pyright: ignore[reportPrivateUsage]
        cast(Any, None)
    )
    assert result is None


# ---------- get_llm_service (factory + cache, real API) ----------


@pytest.mark.integration
def test_get_llm_service_uses_env_and_caches_instance(gemini_api_key: str) -> None:
    os.environ[GEMINI_API_KEY_ENV] = gemini_api_key
    get_llm_service.cache_clear()

    s1 = get_llm_service()
    s2 = get_llm_service()

    # Just test wiring + caching here
    assert isinstance(s1, GeminiLLMService)
    assert s1 is s2  # cached instance