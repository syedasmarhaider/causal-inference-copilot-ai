from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from python.domain.models.models import ChatMessage
from python.domain.service.llm_service import (
    AvailableModelsKey,
    LLMConfig,
    LLMResponse,
    LLMService,
    ToolCall,
)

# LiteLLM reads this during import, before service configuration exists.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_OPTIONAL_PROVIDER_WARNING_SNIPPETS = (
    "could not pre-load bedrock-runtime response stream shape",
    "could not pre-load sagemaker-runtime response stream shape",
)


class _LiteLLMOptionalProviderWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(snippet in message for snippet in _OPTIONAL_PROVIDER_WARNING_SNIPPETS)


logging.getLogger("LiteLLM").addFilter(_LiteLLMOptionalProviderWarningFilter())

Provider = Literal["vertex_ai"]
CompletionFn = Callable[..., Any]
T = TypeVar("T", bound=BaseModel)
F = TypeVar("F", bound=Callable[..., Any])

MAX_TIMEOUT_S = 600.0
_REQUIRED_MODEL_KEYS: tuple[AvailableModelsKey, ...] = ("mini", "basic", "pro", "thinking")
_DEFAULT_VERTEX_LOCATION = "global"


@dataclass(frozen=True)
class ReliabilityPolicy:
    timeout_s: float = 300.0
    hard_deadline_s: float | None = 300.0
    max_retries: int = 2
    base_backoff_s: float = 0.5
    executor_workers: int = 4


class LiteLLMService(LLMService):
    def __init__(
        self,
        *,
        provider: Provider,
        model_names: Mapping[AvailableModelsKey, str],
        reliability: ReliabilityPolicy | None = None,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self._provider = provider
        self._model_names = self._validate_model_names(model_names)
        self._reliability = reliability or ReliabilityPolicy()
        self._validate_reliability(self._reliability)
        self._completion = completion_fn or self._load_completion_fn()
        self._executor = ThreadPoolExecutor(max_workers=self._reliability.executor_workers)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse:
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
        )
        return self._run_with_retries(messages=messages, config=config)

    def generate_json(
        self,
        *,
        schema: type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> T:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        schema_json = json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)
        current_prompt = self._build_json_user_prompt(
            user_prompt=user_prompt,
            schema_json=schema_json,
        )

        for attempt in range(max_attempts):
            attempt_config = config
            if attempt > 0:
                attempt_config = replace(config, temperature=0.0, top_p=1.0)

            response = self.generate(
                system_prompt=system_prompt,
                user_prompt=current_prompt,
                config=attempt_config,
                history=history,
            )

            try:
                return schema.model_validate_json(self._extract_json_payload(response.content))
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Failed JSON schema={schema.__name__} after {max_attempts} attempts"
                    ) from exc

                current_prompt = self._build_json_retry_prompt(
                    user_prompt=user_prompt,
                    schema_json=schema_json,
                    previous_output=response.content,
                    error=exc,
                )

        raise AssertionError("unreachable")

    @staticmethod
    def _load_completion_fn() -> CompletionFn:
        from litellm import completion

        return completion

    @staticmethod
    def _validate_model_names(
        model_names: Mapping[AvailableModelsKey, str],
    ) -> dict[AvailableModelsKey, str]:
        provided_keys = set(model_names.keys())
        required_keys = set(_REQUIRED_MODEL_KEYS)
        if provided_keys != required_keys:
            missing = sorted(required_keys - provided_keys)
            extra = sorted(provided_keys - required_keys)
            raise ValueError(
                f"model_names must contain exactly {sorted(required_keys)}. "
                f"Missing={missing}, extra={extra}"
            )

        normalized: dict[AvailableModelsKey, str] = {}
        for alias in _REQUIRED_MODEL_KEYS:
            name = model_names[alias].strip()
            if not name:
                raise ValueError(f"Concrete model name for alias '{alias}' must be non-empty.")
            normalized[alias] = name
        return normalized

    @staticmethod
    def _validate_reliability(policy: ReliabilityPolicy) -> None:
        if policy.timeout_s <= 0 or policy.timeout_s > MAX_TIMEOUT_S:
            raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S}].")
        if policy.hard_deadline_s is not None and (
            policy.hard_deadline_s <= 0 or policy.hard_deadline_s > MAX_TIMEOUT_S
        ):
            raise ValueError(f"hard_deadline_s must be in (0, {MAX_TIMEOUT_S}] or None.")
        if policy.hard_deadline_s is not None and policy.hard_deadline_s < policy.timeout_s:
            raise ValueError("hard_deadline_s must be >= timeout_s")
        if policy.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if policy.base_backoff_s < 0:
            raise ValueError("base_backoff_s must be >= 0")
        if policy.executor_workers < 1:
            raise ValueError("executor_workers must be >= 1")

    def _run_with_retries(
        self,
        *,
        messages: list[dict[str, str]],
        config: LLMConfig,
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(self._reliability.max_retries + 1):
            try:
                return self._call_completion(messages=messages, config=config)
            except Exception as exc:
                last_error = exc
                if attempt == self._reliability.max_retries:
                    raise
                self._sleep_backoff(attempt)

        if last_error is None:
            raise AssertionError("unreachable")
        raise last_error

    def _call_completion(
        self,
        *,
        messages: list[dict[str, str]],
        config: LLMConfig,
    ) -> LLMResponse:
        kwargs = self._build_completion_kwargs(messages=messages, config=config)

        if self._reliability.hard_deadline_s is None:
            raw_response = self._completion(**kwargs)
        else:
            future = self._executor.submit(self._completion, **kwargs)
            try:
                raw_response = future.result(timeout=self._reliability.hard_deadline_s)
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    f"LLM hard deadline exceeded after {self._reliability.hard_deadline_s} seconds."
                ) from exc

        return self._to_llm_response(raw_response)

    def _build_completion_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        config: LLMConfig,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "timeout": self._reliability.timeout_s,
            "num_retries": 0,
        }

        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        if config.stop is not None:
            kwargs["stop"] = config.stop

        kwargs.update(self._provider_kwargs(config.model))

        if config.extra:
            kwargs.update(dict(config.extra))

        return kwargs

    def _provider_kwargs(self, model_alias: AvailableModelsKey) -> dict[str, Any]:
        model_name = self._model_names[model_alias]

        return {
            "model": self._normalize_vertex_model_name(model_name),
            "vertex_project": self._require_vertex_project(),
            "vertex_location": self._resolve_vertex_location(),
        }

    @staticmethod
    def _normalize_vertex_model_name(model_name: str) -> str:
        prefixes = ("gemini/", "vertex_ai/", "models/")
        normalized = model_name
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        return f"vertex_ai/{normalized}"

    @staticmethod
    def _require_vertex_project() -> str:
        value = os.environ.get("VERTEXAI_PROJECT", "").strip()
        if value:
            return value
        raise ValueError("Missing Vertex project. Set VERTEXAI_PROJECT.")

    @staticmethod
    def _resolve_vertex_location() -> str:
        value = os.environ.get("VERTEXAI_LOCATION", "").strip()
        if value:
            return value
        return _DEFAULT_VERTEX_LOCATION

    @staticmethod
    def _build_messages(
        *,
        system_prompt: str | None,
        user_prompt: str,
        history: Sequence[ChatMessage] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for message in history:
                messages.append({"role": message.role, "content": message.content})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    @staticmethod
    def _build_json_user_prompt(*, user_prompt: str, schema_json: str) -> str:
        return (
            f"{user_prompt}\n\n"
            "Output must be valid JSON matching this schema exactly:\n"
            f"{schema_json}"
        )

    @staticmethod
    def _build_json_retry_prompt(
        *,
        user_prompt: str,
        schema_json: str,
        previous_output: str,
        error: Exception,
    ) -> str:
        return (
            f"{user_prompt}\n\n"
            "Previous output did not validate against the required schema.\n"
            f"Validation error: {error}\n\n"
            "Previous output:\n"
            f"{previous_output}\n\n"
            "Output must be valid JSON matching this schema exactly:\n"
            f"{schema_json}"
        )

    @staticmethod
    def _extract_json_payload(content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()

        if stripped.startswith(("{", "[")):
            return stripped

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1]

        return stripped

    @staticmethod
    def _to_llm_response(raw_response: Any) -> LLMResponse:
        choices = getattr(raw_response, "choices", None)
        if isinstance(choices, Sequence) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            content = LiteLLMService._stringify_content(getattr(message, "content", ""))
            finish_reason = getattr(first_choice, "finish_reason", None)
            tool_calls = LiteLLMService._normalize_tool_calls(getattr(message, "tool_calls", None))
            return LLMResponse(
                content=content.strip(),
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                raw=raw_response,
            )

        content = LiteLLMService._stringify_content(getattr(raw_response, "content", raw_response))
        return LLMResponse(content=content.strip(), raw=raw_response)

    @staticmethod
    def _normalize_tool_calls(raw_tool_calls: Any) -> list[ToolCall] | None:
        if not isinstance(raw_tool_calls, Sequence) or isinstance(raw_tool_calls, (str, bytes)):
            return None

        normalized: list[ToolCall] = []
        for tool_call in raw_tool_calls:
            if isinstance(tool_call, Mapping):
                raw_args = tool_call.get("args")
                normalized.append(
                    ToolCall(
                        id=str(tool_call.get("id", "")),
                        name=str(tool_call.get("name", "")),
                        args=LiteLLMService._normalize_tool_args(raw_args),
                    )
                )
                continue

            function = getattr(tool_call, "function", None)
            raw_args = getattr(function, "arguments", None)
            normalized.append(
                ToolCall(
                    id=str(getattr(tool_call, "id", "")),
                    name=str(getattr(function, "name", "")),
                    args=LiteLLMService._normalize_tool_args(raw_args),
                )
            )

        return normalized or None

    @staticmethod
    def _normalize_tool_args(raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, Mapping):
            return dict(raw_args)
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"raw": raw_args}
        return {}

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, Mapping):
            text = content.get("text")
            if isinstance(text, str):
                return text
            return json.dumps(dict(content), ensure_ascii=False)
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _sleep_backoff(self, attempt: int) -> None:
        delay_s = self._reliability.base_backoff_s * (2**attempt)
        if delay_s > 0:
            time.sleep(delay_s)
