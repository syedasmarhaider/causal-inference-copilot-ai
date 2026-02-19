from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from python.domain.service.llm_service import LLMService
from python.implementation.service.llms.langchain_llm_service import (
    LangChainLLMService,
    ReliabilityPolicy,
)

# NOTE: keep this max cap centralized
MAX_TIMEOUT_S: float = 300.0  # 5 minutes


Provider = Literal["openai", "gemini"]


@dataclass(frozen=True)
class LLMServiceSettings:
    """
    Composition-root settings.

    - backend: "langchain" keeps the domain clean
    - provider: the underlying model provider
    """
    backend: Literal["langchain"] = "langchain"

    provider: Provider = "openai"
    model: str = "gpt-4o-mini"

    timeout_s: float = 300.0
    hard_deadline_s: Optional[float] = 300.0
    max_retries: int = 2


def make_llm_service(settings: LLMServiceSettings) -> LLMService:
    _validate_timeouts(settings.timeout_s, settings.hard_deadline_s)

    if settings.backend != "langchain":
        raise ValueError(f"Unsupported backend: {settings.backend}")

    chat_model = _build_chat_model(
        provider=settings.provider,
        model=settings.model,
        timeout_s=settings.timeout_s,
        max_retries=settings.max_retries,
    )

    return LangChainLLMService(
        model=chat_model,
        reliability=ReliabilityPolicy(
            timeout_s=settings.timeout_s,
            hard_deadline_s=settings.hard_deadline_s,
            max_retries=settings.max_retries,
        ),
    )


def _build_chat_model(*, provider: Provider, model: str, timeout_s: float, max_retries: int) -> BaseChatModel:
    """
    Build a LangChain chat model instance.

    Supports OpenAI and Gemini through LangChain chat providers.
    """
    if provider == "openai":
        # pip install langchain-openai
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY.")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    if provider == "gemini":
        # pip install langchain-google-genai
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY.")

        return ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    # Exhaustive for Literal, but keep runtime guard
    raise ValueError(f"Unsupported provider: {provider}")


def _validate_timeouts(timeout_s: float, hard_deadline_s: Optional[float]) -> None:
    if timeout_s <= 0 or timeout_s > MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S}] seconds. Got {timeout_s}.")
    if hard_deadline_s is not None and (hard_deadline_s <= 0 or hard_deadline_s > MAX_TIMEOUT_S):
        raise ValueError(f"hard_deadline_s must be in (0, {MAX_TIMEOUT_S}] seconds or None. Got {hard_deadline_s}.")
    if hard_deadline_s is not None and hard_deadline_s < timeout_s:
        raise ValueError("hard_deadline_s must be >= timeout_s (or None).")
    if max_retries := 0:
        # placeholder to avoid unused-style warnings if you later add more checks
        _ = max_retries
