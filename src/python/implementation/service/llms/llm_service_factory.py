from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from python.domain.service.llm_service import AvailableModelsKey, LLMService
from python.implementation.service.llms.litellm_llm_service import (
    MAX_TIMEOUT_S,
    LiteLLMService,
    Provider,
    ReliabilityPolicy,
)

ProviderSetting = Literal["auto", "vertex_ai", "google_ai_studio", "openai", "azure"]
_REQUIRED_MODEL_KEYS: tuple[AvailableModelsKey, ...] = ("mini", "basic", "pro", "thinking")
_DEFAULT_PROVIDER: Provider = "vertex_ai"

_MODEL_ENV_NAMES: Mapping[AvailableModelsKey, tuple[str, ...]] = {
    "mini": ("LLM_MODEL_MINI", "LLM_MINI_MODEL"),
    "basic": ("LLM_MODEL_BASIC", "LLM_BASIC_MODEL"),
    "pro": ("LLM_MODEL_PRO", "LLM_PRO_MODEL"),
    "thinking": ("LLM_MODEL_THINKING", "LLM_THINKING_MODEL"),
}


DEFAULT_VERTEX_MODEL_MAP: Mapping[AvailableModelsKey, str] = {
    "mini": "gemini-3.1-flash-lite-preview",
    "basic": "gemini-3-flash-preview",
    "pro": "gemini-3-flash-preview",
    "thinking": "gemini-3.1-pro-preview",
}


@dataclass(frozen=True)
class LLMServiceSettings:
    backend: Literal["litellm"] = "litellm"
    provider: ProviderSetting = "vertex_ai"
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    model_map: Mapping[AvailableModelsKey, str] = field(
        default_factory=lambda: dict(DEFAULT_VERTEX_MODEL_MAP)
    )
    timeout_s: float = 300.0
    hard_deadline_s: float | None = 300.0
    max_retries: int = 2
    executor_workers: int = 4

    @classmethod
    def from_env(cls) -> LLMServiceSettings:
        provider = _provider_from_env()
        return cls(
            provider=provider,
            api_key=_api_key_from_env(provider),
            api_base=_optional_env("LLM_API_BASE", "AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT"),
            api_version=_optional_env("LLM_API_VERSION", "AZURE_API_VERSION", "OPENAI_API_VERSION"),
            model_map=_model_map_from_env(provider),
        )


def make_llm_service(settings: LLMServiceSettings) -> LLMService:
    settings = _resolve_auto_settings(settings)
    _validate_settings(settings)

    if settings.backend != "litellm":
        raise ValueError(f"Unsupported backend: {settings.backend}")

    return LiteLLMService(
        provider=_resolve_provider(settings.provider),
        model_names=settings.model_map,
        api_key=settings.api_key,
        api_base=settings.api_base,
        api_version=settings.api_version,
        reliability=ReliabilityPolicy(
            timeout_s=settings.timeout_s,
            hard_deadline_s=settings.hard_deadline_s,
            max_retries=settings.max_retries,
            executor_workers=settings.executor_workers,
        ),
    )


def _resolve_provider(provider: ProviderSetting) -> Provider:
    normalized = _normalize_provider(provider)
    if normalized == "auto":
        return _DEFAULT_PROVIDER
    return normalized


def _resolve_auto_settings(settings: LLMServiceSettings) -> LLMServiceSettings:
    if _normalize_provider(settings.provider) != "auto":
        return settings

    env_settings = LLMServiceSettings.from_env()
    return LLMServiceSettings(
        backend=settings.backend,
        provider=env_settings.provider,
        api_key=settings.api_key or env_settings.api_key,
        api_base=settings.api_base or env_settings.api_base,
        api_version=settings.api_version or env_settings.api_version,
        model_map=env_settings.model_map,
        timeout_s=settings.timeout_s,
        hard_deadline_s=settings.hard_deadline_s,
        max_retries=settings.max_retries,
        executor_workers=settings.executor_workers,
    )


def _provider_from_env() -> Provider:
    return _resolve_provider(os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER))


def _normalize_provider(provider: str) -> ProviderSetting:
    normalized = provider.strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"vertex", "vertex_ai", "google_vertex_ai"}:
        return "vertex_ai"
    if normalized in {"google_ai_studio", "ai_studio", "google", "gemini"}:
        return "google_ai_studio"
    if normalized == "openai":
        return "openai"
    if normalized in {"azure", "azure_openai"}:
        return "azure"
    raise ValueError(
        "Unsupported LLM provider. Expected one of: "
        "vertex_ai, google_ai_studio, openai, azure."
    )


def _model_map_from_env(provider: Provider) -> Mapping[AvailableModelsKey, str]:
    if provider == "vertex_ai":
        defaults = dict(DEFAULT_VERTEX_MODEL_MAP)
        defaults.update(_configured_model_overrides())
        return defaults

    model_map = _configured_model_overrides()
    missing = [alias for alias in _REQUIRED_MODEL_KEYS if alias not in model_map]
    if missing:
        expected = ", ".join(_MODEL_ENV_NAMES[alias][0] for alias in missing)
        raise ValueError(
            f"Missing model env vars for LLM_PROVIDER={provider}: {expected}."
        )
    return model_map


def _configured_model_overrides() -> dict[AvailableModelsKey, str]:
    model_map: dict[AvailableModelsKey, str] = {}
    for alias, env_names in _MODEL_ENV_NAMES.items():
        value = _optional_env(*env_names)
        if value is not None:
            model_map[alias] = value
    return model_map


def _api_key_from_env(provider: Provider) -> str | None:
    if provider == "vertex_ai":
        return None
    if provider == "google_ai_studio":
        return _optional_env(
            "LLM_API_KEY",
            "GOOGLE_AI_STUDIO_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        )
    if provider == "openai":
        return _optional_env("LLM_API_KEY", "OPENAI_API_KEY")
    if provider == "azure":
        return _optional_env("LLM_API_KEY", "AZURE_API_KEY", "AZURE_OPENAI_API_KEY")
    raise ValueError(f"Unsupported provider: {provider}")


def _optional_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _validate_settings(settings: LLMServiceSettings) -> None:
    _resolve_provider(settings.provider)

    if settings.timeout_s <= 0 or settings.timeout_s > MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout_s must be in (0, {MAX_TIMEOUT_S}] seconds. Got {settings.timeout_s}."
        )

    if settings.hard_deadline_s is not None and (
        settings.hard_deadline_s <= 0 or settings.hard_deadline_s > MAX_TIMEOUT_S
    ):
        raise ValueError(
            "hard_deadline_s must be in "
            f"(0, {MAX_TIMEOUT_S}] seconds or None. Got {settings.hard_deadline_s}."
        )

    if settings.hard_deadline_s is not None and settings.hard_deadline_s < settings.timeout_s:
        raise ValueError("hard_deadline_s must be >= timeout_s (or None).")

    if settings.max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    if settings.executor_workers < 1:
        raise ValueError("executor_workers must be >= 1")

    provided_keys = set(settings.model_map.keys())
    required_keys = set(_REQUIRED_MODEL_KEYS)
    if provided_keys != required_keys:
        missing = sorted(required_keys - provided_keys)
        extra = sorted(provided_keys - required_keys)
        raise ValueError(
            f"model_map must contain exactly {sorted(required_keys)}. Missing={missing}, extra={extra}"
        )

    for alias in _REQUIRED_MODEL_KEYS:
        concrete_model_name = settings.model_map[alias]
        if not concrete_model_name.strip():
            raise ValueError(f"Concrete model name for alias '{alias}' must be a non-empty string.")
