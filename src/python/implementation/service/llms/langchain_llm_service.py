from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from python.domain.service.llm_service import AvailableModelsKey, ChatMessage, LLMConfig, LLMResponse, LLMService

T = TypeVar("T", bound=BaseModel)
from langfuse import observe

MAX_TIMEOUT_S: float = 300.0  # 5 minutes


@dataclass(frozen=True)
class ReliabilityPolicy:
    """
    - timeout_s: soft timeout at provider/client level
    - hard_deadline_s: hard cutoff enforced by our own executor future timeout
    - max_retries: retry count for transient failures
    - base_backoff_s: exponential backoff base
    - executor_workers: parallel invoke capacity for a shared service instance
    """
    timeout_s: float = 60.0
    hard_deadline_s: float | None = 90.0
    max_retries: int = 2
    base_backoff_s: float = 1.5
    executor_workers: int = 4


class LangChainLLMService(LLMService):
    """
    LLMService backed by LangChain chat models.

    Key design points:
    - `config.model` is an alias key: mini/basic/pro/thinking
    - alias -> concrete provider model is resolved by the composition root
    - this service does NOT detect provider dynamically
    - token kwarg name is injected (`max_output_tokens` for Gemini, `max_tokens` for OpenAI)
    - hard deadline is enforced with ThreadPoolExecutor
    """

    def __init__(
        self,
        *,
        models: Mapping[AvailableModelsKey, BaseChatModel],
        model_names: Mapping[AvailableModelsKey, str],
        max_tokens_param_name: str,
        reliability: ReliabilityPolicy = ReliabilityPolicy(),
    ) -> None:
        self._rel = self._validate_rel(reliability)

        self._models: Dict[AvailableModelsKey, BaseChatModel] = dict(models)
        self._model_names: Dict[AvailableModelsKey, str] = dict(model_names)
        self._max_tokens_param_name = max_tokens_param_name
        self._executor = ThreadPoolExecutor(max_workers=self._rel.executor_workers)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
    
    @observe()
    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
    ) -> LLMResponse:
        messages = self._to_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
        )
        ai = self._invoke_with_retries(messages=messages, config=config)
        return self._to_domain_response(
            ai=ai,
            fallback_model=self._resolve_concrete_model_name(config.model),
        )
    @observe()
    def generate_json(
        self,
        *,
        schema: Type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
        max_attempts: int = 3,
    ) -> T:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        
        parser = PydanticOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()

        messages = self._to_messages(
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\n{format_instructions}",
            history=history,
        )

        last_error_text: str | None = None
        last_output_text: str | None = None

        for attempt in range(1, max_attempts + 1):
            attempt_config = config if attempt == 1 else self._deterministic_config(config)

            ai = self._invoke_with_retries(messages=messages, config=attempt_config)
            text = self._content_to_str(ai.content).strip() # pyright: ignore[reportUnknownMemberType]
            last_output_text = text

            try:
                return parser.parse(text)
            except (ValidationError, OutputParserException, ValueError) as e:
                last_error_text = self._format_validation_error(e)

                if attempt >= max_attempts:
                    break

                repair_prompt = (
                    "Your previous output did not validate.\n"
                    "Fix it.\n\n"
                    "Rules:\n"
                    "1) Output MUST be valid JSON\n"
                    "2) Output MUST match the schema exactly\n"
                    "3) Output ONLY JSON (no markdown, no explanations)\n\n"
                    f"Validation error:\n{last_error_text}\n\n"
                    f"{format_instructions}\n\n"
                    f"Invalid output:\n{last_output_text}\n"
                )
                messages = self._to_messages(
                    system_prompt=system_prompt,
                    user_prompt=repair_prompt,
                    history=None,
                )

        raise RuntimeError(
            f"Failed JSON schema={schema.__name__} after {max_attempts} attempts. "
            f"Last error: {last_error_text or 'unknown'}"
        )

    # ---------------- internals ----------------
    
    @staticmethod
    def _validate_rel(rel: ReliabilityPolicy) -> ReliabilityPolicy:
        if rel.timeout_s <= 0 or rel.timeout_s > MAX_TIMEOUT_S:
            raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S}] seconds. Got {rel.timeout_s}.")
        if rel.hard_deadline_s is not None and (rel.hard_deadline_s <= 0 or rel.hard_deadline_s > MAX_TIMEOUT_S):
            raise ValueError(
                f"hard_deadline_s must be in (0, {MAX_TIMEOUT_S}] seconds or None. Got {rel.hard_deadline_s}."
            )
        if rel.hard_deadline_s is not None and rel.hard_deadline_s < rel.timeout_s:
            raise ValueError("hard_deadline_s must be >= timeout_s (or None).")
        if rel.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if rel.executor_workers < 1:
            raise ValueError("executor_workers must be >= 1")
        return rel

    @staticmethod
    def _to_messages(
        *,
        system_prompt: str | None,
        user_prompt: str,
        history: Optional[Sequence[ChatMessage]],
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if history:
            for item in history:
                if item.role == "user":
                    messages.append(HumanMessage(content=item.content))
                elif item.role == "assistant":
                    messages.append(AIMessage(content=item.content))
                elif item.role == "system":
                    messages.append(SystemMessage(content=item.content))

        messages.append(HumanMessage(content=user_prompt))
        return messages

    def _resolve_model_key(self, model: str) -> AvailableModelsKey:
        if model in self._models:
            return  model

        allowed = ", ".join(sorted(self._models.keys()))
        raise ValueError(
            f"Unsupported model alias '{model}'. Expected one of: {allowed}. "
            "Pass the alias key, not the concrete provider model name."
        )

    def _resolve_concrete_model_name(self, model: str) -> str:
        key = self._resolve_model_key(model)
        return self._model_names[key]

    def _get_model(self, model: str) -> BaseChatModel:
        key = self._resolve_model_key(model)
        return self._models[key]

    def _invoke_with_retries(self, *, messages: List[BaseMessage], config: LLMConfig) -> AIMessage:
        last_exc: Exception | None = None

        for attempt in range(self._rel.max_retries + 1):
            try:
                out = self._invoke_once(messages=messages, config=config)

                if isinstance(out, AIMessage):
                    return out

                content = getattr(out, "content", out)
                return AIMessage(content=self._content_to_str(content))
            except Exception as e:
                last_exc = e
                if attempt >= self._rel.max_retries or not self._is_retriable(e):
                    raise
                self._sleep_backoff(attempt)

        assert last_exc is not None
        raise last_exc

    def _invoke_once(self, *, messages: List[BaseMessage], config: LLMConfig) -> Any:
        model = self._get_model(config.model)
        kwargs = self._invoke_kwargs(config)

        if self._rel.hard_deadline_s is None:
            return model.invoke(messages, **kwargs)

        return self._call_with_deadline(
            model=model,
            messages=messages,
            kwargs=kwargs,
            deadline_s=self._rel.hard_deadline_s,
        )

    def _invoke_kwargs(self, config: LLMConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}

        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.stop:
            kwargs["stop"] = list(config.stop)

        if config.max_tokens is not None:
            kwargs[self._max_tokens_param_name] = config.max_tokens

        if config.extra:
            kwargs.update(dict(config.extra))

        # sanitize conflicting token params
        if self._max_tokens_param_name == "max_output_tokens":
            kwargs.pop("max_tokens", None)
        elif self._max_tokens_param_name == "max_tokens":
            kwargs.pop("max_output_tokens", None)

        return kwargs

    def _call_with_deadline(
        self,
        *,
        model: BaseChatModel,
        messages: List[BaseMessage],
        kwargs: Dict[str, Any],
        deadline_s: float,
    ) -> Any:
        future = self._executor.submit(model.invoke, messages, **kwargs)
        try:
            return future.result(timeout=deadline_s)
        except FutureTimeoutError as e:
            future.cancel()
            raise TimeoutError(f"LLM call exceeded hard deadline {deadline_s}s") from e

    def _sleep_backoff(self, attempt: int) -> None:
        sleep_s = self._rel.base_backoff_s * (2 ** attempt) * (0.8 + 0.4 * random.random())
        time.sleep(sleep_s)

    @staticmethod
    def _deterministic_config(cfg: LLMConfig) -> LLMConfig:
        return LLMConfig(
            model=cfg.model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            extra=cfg.extra,
        )

    @staticmethod
    def _format_validation_error(e: Exception) -> str:
        if isinstance(e, ValidationError):
            try:
                return json.dumps(e.errors(), indent=2)
            except Exception:
                return str(e)
        return str(e)

    def _is_retriable(self, e: Exception) -> bool:
        if isinstance(e, TimeoutError):
            return True

        message = str(e).lower()
        transient_markers = [
            "429",
            "rate limit",
            "too many requests",
            "503",
            "service unavailable",
            "502",
            "bad gateway",
            "timeout",
            "timed out",
            "connection reset",
            "connection error",
            "temporarily unavailable",
            "deadline",
            "unavailable",
        ]
        return any(marker in message for marker in transient_markers)

    @staticmethod
    def _content_to_str(content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    parts.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False, default=str))
                else:
                    parts.append(str(item))
            return "".join(parts)

        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
            return json.dumps(content, ensure_ascii=False, default=str)

        return str(content)


    def _to_domain_response(self, *, ai: AIMessage, fallback_model: str | None) -> LLMResponse:
        content = self._content_to_str(ai.content).strip()
        return LLMResponse(
            content=content,
            finish_reason=None,
            raw=ai,
        )