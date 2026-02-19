from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from python.domain.service.llm_service import (
    ChatMessage,
    LLMConfig,
    LLMResponse,
    LLMService,
)

T = TypeVar("T", bound=BaseModel)

MAX_TIMEOUT_S: float = 300.0  # 5 minutes


@dataclass(frozen=True)
class ReliabilityPolicy:
    timeout_s: float = 300.0
    hard_deadline_s: float | None = 300.0
    max_retries: int = 2
    base_backoff_s: float = 1.5


class LangChainLLMService(LLMService):
    """
    Provider-agnostic LLMService backed by a LangChain BaseChatModel.

    Pylance-safe:
      - AIMessage.content normalization (str | list[str|dict] -> str)
      - response_metadata/tool_calls typed access
    """

    def __init__(self, *, model: BaseChatModel, reliability: ReliabilityPolicy = ReliabilityPolicy()) -> None:
        self._model = model
        self._rel = self._validate_rel(reliability)

    def close(self) -> None:
        return

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
    ) -> LLMResponse:
        msgs = self._to_messages(system_prompt=system_prompt, user_prompt=user_prompt, history=history)
        ai = self._invoke_with_retries(messages=msgs, config=config)
        return self._to_domain_response(ai=ai, fallback_model=config.model)

    def generate_json(
        self,
        *,
        schema: Type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
        max_attempts: int = 2,
    ) -> T:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        parser = PydanticOutputParser(pydantic_object=schema)
        fmt = parser.get_format_instructions()

        # Attempt 1 includes history. Repair attempts drop history and force determinism.
        prompt = f"{user_prompt}\n\n{fmt}"
        msgs: List[BaseMessage] = self._to_messages(system_prompt=system_prompt, user_prompt=prompt, history=history)

        last_err: Exception | None = None
        last_text: str | None = None

        for attempt in range(1, max_attempts + 1):
            attempt_cfg = config if attempt == 1 else self._deterministic_config(config)

            ai = self._invoke_with_retries(messages=msgs, config=attempt_cfg)
            text = self._content_to_str(ai.content).strip()
            last_text = text

            try:
                return parser.parse(text)
            except (ValidationError, OutputParserException, ValueError) as e:
                last_err = e
                if attempt >= max_attempts:
                    break

                repair_prompt = (
                    "Your previous output did not validate against the required schema.\n"
                    "Fix it.\n\n"
                    "Rules:\n"
                    "1) Output MUST be valid JSON\n"
                    "2) Output MUST match the schema exactly\n"
                    "3) Output ONLY JSON. No markdown, no explanations.\n\n"
                    f"{fmt}\n\n"
                    f"Invalid output:\n{last_text}\n"
                )
                msgs = self._to_messages(system_prompt=system_prompt, user_prompt=repair_prompt, history=None)

        assert last_err is not None
        raise RuntimeError(
            f"Failed to produce valid JSON for schema={schema.__name__} after {max_attempts} attempts. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        ) from last_err

    # ---------------- internals ----------------

    @staticmethod
    def _validate_rel(rel: ReliabilityPolicy) -> ReliabilityPolicy:
        if rel.timeout_s <= 0 or rel.timeout_s > MAX_TIMEOUT_S:
            raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S}] seconds. Got {rel.timeout_s}.")
        if rel.hard_deadline_s is not None and (rel.hard_deadline_s <= 0 or rel.hard_deadline_s > MAX_TIMEOUT_S):
            raise ValueError(f"hard_deadline_s must be in (0, {MAX_TIMEOUT_S}] seconds or None. Got {rel.hard_deadline_s}.")
        if rel.hard_deadline_s is not None and rel.hard_deadline_s < rel.timeout_s:
            raise ValueError("hard_deadline_s should be >= timeout_s (or None).")
        if rel.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        return rel

    @staticmethod
    def _to_messages(
        *,
        system_prompt: str | None,
        user_prompt: str,
        history: Optional[Sequence[ChatMessage]],
    ) -> List[BaseMessage]:
        msgs: List[BaseMessage] = []
        if system_prompt:
            msgs.append(SystemMessage(content=system_prompt))
        if history:
            for h in history:
                if h.role == "user":
                    msgs.append(HumanMessage(content=h.content))
                elif h.role == "assistant":
                    msgs.append(AIMessage(content=h.content))
        msgs.append(HumanMessage(content=user_prompt))
        return msgs

    def _invoke_with_retries(self, *, messages: List[BaseMessage], config: LLMConfig) -> AIMessage:
        last_exc: Exception | None = None
        for attempt in range(self._rel.max_retries + 1):
            try:
                out = self._invoke_once(messages=messages, config=config)
                if isinstance(out, AIMessage):
                    return out
                # Defensive: normalize any BaseMessage to AIMessage
                content = getattr(out, "content", out)
                return AIMessage(content=self._content_to_str(content))
            except Exception as e:
                last_exc = e
                if attempt >= self._rel.max_retries:
                    raise
                self._sleep_backoff(attempt)
        assert last_exc is not None
        raise last_exc

    def _invoke_once(self, *, messages: List[BaseMessage], config: LLMConfig) -> Any:
        kwargs = self._invoke_kwargs(config)

        if self._rel.hard_deadline_s is None:
            return self._model.invoke(messages, **kwargs)

        return self._call_with_deadline(
            model=self._model,
            messages=messages,
            kwargs=kwargs,
            deadline_s=self._rel.hard_deadline_s,
        )

    @staticmethod
    def _invoke_kwargs(config: LLMConfig) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.stop:
            kwargs["stop"] = config.stop
        if config.max_tokens is not None:
            # Some providers use max_tokens, Gemini often uses max_output_tokens
            kwargs["max_tokens"] = config.max_tokens
            kwargs["max_output_tokens"] = config.max_tokens
        if config.extra:
            kwargs.update(config.extra)
        return kwargs

    @staticmethod
    def _call_with_deadline(*, model: BaseChatModel, messages: List[BaseMessage], kwargs: Dict[str, Any], deadline_s: float) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(model.invoke, messages, **kwargs)
        try:
            return future.result(timeout=deadline_s)
        except FutureTimeoutError as e:
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"LLM call exceeded hard deadline {deadline_s}s") from e
        finally:
            executor.shutdown(wait=False, cancel_futures=False)

    def _sleep_backoff(self, attempt: int) -> None:
        sleep_s = self._rel.base_backoff_s * (2**attempt) * (0.8 + 0.4 * random.random())
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

    # --------- Pylance-safe normalization + extraction ---------

    @staticmethod
    def _content_to_str(content: Any) -> str:
        """
        LangChain AIMessage.content may be:
          - str
          - list[str | dict[...]]  (multi-part content)
        This normalizes to a single string deterministically.
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    # Common patterns: {"text": "..."} or {"type": "text", "text": "..."}
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False, default=str))
                    continue

                parts.append(str(item))
            return "".join(parts)

        if isinstance(content, dict):
            txt2 = content.get("text")
            if isinstance(txt2, str):
                return txt2
            return json.dumps(content, ensure_ascii=False, default=str)

        return str(content)


    @staticmethod
    def _extract_model(ai: AIMessage, fallback: str | None) -> str:
        md_any = getattr(ai, "response_metadata", None)
        md: Mapping[str, Any] = md_any if isinstance(md_any, dict) else {}

        m = md.get("model") or md.get("model_name") or md.get("model_version")
        if isinstance(m, str) and m.strip():
            return m
        return fallback or "unknown"


    def _to_domain_response(self, *, ai: AIMessage, fallback_model: str | None) -> LLMResponse:
        content_str = self._content_to_str(ai.content).strip()

        return LLMResponse(
            content=content_str,
            model=self._extract_model(ai, fallback=fallback_model),
            finish_reason=None,
            raw=ai,
        )
