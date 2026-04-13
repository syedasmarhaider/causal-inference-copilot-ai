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

ProviderSetting = Literal["auto", "gemini", "vertex_api", "vertex_ai"]
_REQUIRED_MODEL_KEYS: tuple[AvailableModelsKey, ...] = ("mini", "basic", "pro", "thinking")


DEFAULT_GEMINI_MODEL_MAP: Mapping[AvailableModelsKey, str] = {
    "mini": "gemini-3.1-flash-lite-preview",
    "basic": "gemini-3-flash-preview",
    "pro": "gemini-3-flash-preview",
    "thinking": "gemini-3.1-pro-preview",
}


@dataclass(frozen=True)
class LLMServiceSettings:
    backend: Literal["litellm"] = "litellm"
    provider: ProviderSetting = "auto"
    model_map: Mapping[AvailableModelsKey, str] = field(
        default_factory=lambda: dict(DEFAULT_GEMINI_MODEL_MAP)
    )
    timeout_s: float = 300.0
    hard_deadline_s: float | None = 300.0
    max_retries: int = 2
    executor_workers: int = 4


def make_llm_service(settings: LLMServiceSettings) -> LLMService:
    _validate_settings(settings)

    if settings.backend != "litellm":
        raise ValueError(f"Unsupported backend: {settings.backend}")

    return LiteLLMService(
        provider=_resolve_provider(settings.provider),
        model_names=settings.model_map,
        reliability=ReliabilityPolicy(
            timeout_s=settings.timeout_s,
            hard_deadline_s=settings.hard_deadline_s,
            max_retries=settings.max_retries,
            executor_workers=settings.executor_workers,
        ),
    )


def _resolve_provider(provider: ProviderSetting) -> Provider:
    if provider != "auto":
        return _normalize_provider(provider)

    env_provider = os.environ.get("LITELLM_PROVIDER", "").strip().lower()
    if env_provider:
        return _normalize_provider(env_provider)

    if _has_any_env_value("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        return "gemini"

    if _has_any_env_value("VERTEX_AI_API_KEY"):
        return "vertex_api"

    return "gemini"


def _normalize_provider(provider: str) -> Provider:
    if provider == "gemini":
        return "gemini"
    if provider in {"vertex_api", "vertex_ai"}:
        return "vertex_api"
    raise ValueError(
        "Unsupported provider. Expected one of: auto, gemini, vertex_api, vertex_ai."
    )


def _has_any_env_value(*env_names: str) -> bool:
    return any(os.environ.get(name, "").strip() for name in env_names)


def _validate_settings(settings: LLMServiceSettings) -> None:
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
