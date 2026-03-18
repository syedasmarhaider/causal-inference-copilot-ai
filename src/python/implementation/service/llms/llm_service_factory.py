from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from python.domain.service.llm_service import AvailableModelsKey, LLMService
from python.implementation.service.llms.langchain_llm_service import (
    MAX_TIMEOUT_S,
    LangChainLLMService,
    ReliabilityPolicy,
)

Provider = Literal["gemini"]


DEFAULT_GEMINI_MODEL_MAP: Mapping[AvailableModelsKey, str] = {
    "mini": "gemini-3.1-flash-lite-preview",
    "basic": "gemini-2.5-flash",
    "pro": "gemini-3-flash-preview",
    "thinking": "gemini-3.1-pro-preview",
}


@dataclass(frozen=True)
class LLMServiceSettings:
    """
    Composition-root settings.

    `config.model` at call time must be one of:
      - "mini"
      - "basic"
      - "pro"
      - "thinking"

    This object maps those aliases to concrete provider model names.
    """
    backend: Literal["langchain"] = "langchain"
    provider: Provider = "gemini"

    model_map: Mapping[AvailableModelsKey, str] = field(
        default_factory=lambda: dict(DEFAULT_GEMINI_MODEL_MAP)
    )

    timeout_s: float = 300.0
    hard_deadline_s: Optional[float] = 300.0
    max_retries: int = 2
    executor_workers: int = 4


def make_llm_service(settings: LLMServiceSettings) -> LLMService:
    _validate_settings(settings)

    if settings.backend != "langchain":
        raise ValueError(f"Unsupported backend: {settings.backend}")

    alias_models, alias_model_names = _build_chat_models(
        provider=settings.provider,
        model_map=settings.model_map,
        timeout_s=settings.timeout_s,
    )

    return LangChainLLMService(
        models=alias_models,
        model_names=alias_model_names,
        max_tokens_param_name="max_output_tokens",  # Gemini-specific
        reliability=ReliabilityPolicy(
            timeout_s=settings.timeout_s,
            hard_deadline_s=settings.hard_deadline_s,
            max_retries=settings.max_retries,
            executor_workers=settings.executor_workers,
        ),
    )


def _build_chat_models(
    *,
    provider: Provider,
    model_map: Mapping[AvailableModelsKey, str],
    timeout_s: float,
) -> tuple[Dict[AvailableModelsKey, BaseChatModel], Dict[AvailableModelsKey, str]]:
    """
    Build one LangChain chat model per unique concrete provider model name,
    then attach aliases to those instances.

    This deduplicates instances when multiple aliases point to the same concrete model.
    """
    if provider != "gemini":
        raise ValueError(f"Unsupported provider: {provider}")

    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set GEMINI_API_KEY.")

    unique_models: Dict[str, BaseChatModel] = {}
    alias_models: Dict[AvailableModelsKey, BaseChatModel] = {}
    alias_model_names: Dict[AvailableModelsKey, str] = {}

    for alias, concrete_model_name in model_map.items():
        name = concrete_model_name.strip()
        if not name:
            raise ValueError(f"Concrete model name for alias '{alias}' must be non-empty.")

        model = unique_models.get(name)
        if model is None:
            model = ChatGoogleGenerativeAI(
                model=name,
                api_key=api_key,
                timeout=timeout_s,
                max_retries=0,  # important: service owns retries
            )
            unique_models[name] = model

        alias_models[alias] = model
        alias_model_names[alias] = name

    return alias_models, alias_model_names


def _validate_settings(settings: LLMServiceSettings) -> None:
    if settings.timeout_s <= 0 or settings.timeout_s > MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S}] seconds. Got {settings.timeout_s}.")

    if settings.hard_deadline_s is not None and (
        settings.hard_deadline_s <= 0 or settings.hard_deadline_s > MAX_TIMEOUT_S
    ):
        raise ValueError(
            f"hard_deadline_s must be in (0, {MAX_TIMEOUT_S}] seconds or None. Got {settings.hard_deadline_s}."
        )

    if settings.hard_deadline_s is not None and settings.hard_deadline_s < settings.timeout_s:
        raise ValueError("hard_deadline_s must be >= timeout_s (or None).")

    if settings.max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    if settings.executor_workers < 1:
        raise ValueError("executor_workers must be >= 1")

    required_keys = {"mini", "basic", "pro", "thinking"}
    provided_keys = set(settings.model_map.keys())
    if provided_keys != required_keys:
        missing = sorted(required_keys - provided_keys)
        extra = sorted(provided_keys - required_keys) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportOperatorIssue]
        raise ValueError(
            f"model_map must contain exactly {sorted(required_keys)}. Missing={missing}, extra={extra}"
        )

    for alias, concrete_model_name in settings.model_map.items():
        if not isinstance(concrete_model_name, str) or not concrete_model_name.strip(): # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"Concrete model name for alias '{alias}' must be a non-empty string.")