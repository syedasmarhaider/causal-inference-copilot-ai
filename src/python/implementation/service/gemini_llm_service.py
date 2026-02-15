from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import  Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from python.domain.service.llm_service import (
    LLMService,
    LLMConfig,
    ChatMessage,
    LLMResponse,
    Usage,
    ToolCall,
    ProviderExtra,
)

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


class GeminiLLMService(LLMService):
    """
    LLMService implementation backed by google-genai (python-genai).

    Hard requirements for reliability:
      - set HttpOptions.timeout (ms) to avoid indefinite hangs
      - always set max_output_tokens to bound model work
    """

    def __init__(
        self,
        *,
        default_model: str = "gemini-2.5-flash",

        timeout_ms: int = 220_000,        # 120s read timeout (ms)
        hard_deadline_s: float | None = None,  # e.g. 130.0 to kill zombied calls
    ) -> None:
        api_key = os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self._default_model = default_model
        self._hard_deadline_s = hard_deadline_s

        http_options = types.HttpOptions(
            timeout=timeout_ms,  # milliseconds
        )

        # Create one client for the lifetime of this service (close on shutdown).
        self._client = genai.Client(api_key=api_key, http_options=http_options)

    def close(self) -> None:
        # Call this from your app shutdown hook (FastAPI lifespan, etc.)
        self._client.close()

    # ------------- public API (from LLMService) -------------

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None = None,
    ) -> LLMResponse:
        logging.warning("-----GeminiLLMService(generate) called-----")

        model_name = config.model or self._default_model
        contents = self._build_contents(history=history, user_prompt=user_prompt)

        gen_cfg = self._build_generation_config(system_prompt=system_prompt, cfg=config)

        max_retries = 2
        base_backoff_s = 1.5
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                if self._hard_deadline_s is None:
                    resp = self._client.models.generate_content( # pyright: ignore[reportUnknownMemberType]
                        model=model_name,
                        contents=contents,
                        config=gen_cfg,
                    )
                else:
                    # Guard against the “socket zombie” hang class.
                    resp = self._call_with_hard_deadline(
                        model=model_name,
                        contents=contents,
                        gen_cfg=gen_cfg,
                        deadline_s=self._hard_deadline_s,
                    )

                logging.warning("-----GeminiLLMService(generate) responded-----")
                return self._to_llm_response(resp, fallback_model=model_name)

            except (genai_errors.ClientError, genai_errors.ServerError) as e:
                # e.code is HTTP status; 429/503 commonly transient
                last_exc = e
                retryable = getattr(e, "code", None) in (429, 500, 502, 503, 504)
                if retryable and attempt < max_retries:
                    self._sleep_backoff(base_backoff_s, attempt)
                    continue
                raise

            except (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.TransportError) as e:
                last_exc = e
                if attempt < max_retries:
                    self._sleep_backoff(base_backoff_s, attempt)
                    continue
                raise

            except Exception as e:
                logging.exception("Gemini unrecoverable error")
                raise

        assert last_exc is not None
        raise last_exc

    # ------------- helpers -------------

    @staticmethod
    def _sleep_backoff(base_s: float, attempt: int) -> None:
        # exponential backoff + jitter
        sleep_s = base_s * (2**attempt) * (0.8 + 0.4 * random.random())
        time.sleep(sleep_s)

    @staticmethod
    def _build_contents(
        *,
        history: Sequence[ChatMessage] | None,
        user_prompt: str,
    ) -> list[types.Content]:
        contents: list[types.Content] = []

        if history:
            for msg in history:
                if msg.role == "user":
                    contents.append(
                        types.UserContent(parts=[types.Part.from_text(text=msg.content)])
                    )
                elif msg.role == "assistant":
                    contents.append(
                        types.ModelContent(parts=[types.Part.from_text(text=msg.content)])
                    )
                else:
                    # system/tool are handled elsewhere or ignored
                    continue

        contents.append(types.UserContent(parts=[types.Part.from_text(text=user_prompt)]))
        return contents

    @staticmethod
    def _extract_safety_settings(extra: ProviderExtra) -> Any:
        return extra.get("safety_settings")

    def _build_generation_config(
        self,
        *,
        system_prompt: str | None,
        cfg: LLMConfig,
    ) -> types.GenerateContentConfig:
        max_out = cfg.max_tokens if cfg.max_tokens is not None else 30000

        return types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_output_tokens=max_out,
            stop_sequences=cfg.stop or None,
        )

    def _call_with_hard_deadline(
        self,
        *,
        model: str,
        contents: list[types.Content],
        gen_cfg: types.GenerateContentConfig,
        deadline_s: float,
    ) -> Any:
        # NOTE: This pattern exists because of the documented “hang forever” bug class.
        # If the SDK thread stalls at socket-level, future.result(timeout=...) returns,
        # but the worker thread may remain stuck. Use sparingly (e.g., only in prod).
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._client.models.generate_content, # pyright: ignore[reportUnknownArgumentType] # type: ignore
            model=model,
            contents=contents,
            config=gen_cfg,
        )
        try:
            return future.result(timeout=deadline_s)
        except FutureTimeoutError as e:
            # Don't wait for shutdown (can deadlock if thread is stuck)
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"Gemini call exceeded hard deadline {deadline_s}s") from e
        finally:
            # If completed normally, release resources
            executor.shutdown(wait=False, cancel_futures=False)

    @staticmethod
    def _to_llm_response(response: Any, fallback_model: str) -> LLMResponse:
        # response.text is the normal path; still guard for missing text.
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            # stitch parts if needed
            texts: list[str] = []
            parts = getattr(response, "parts", None)
            if isinstance(parts, list):
                for p in parts: # pyright: ignore[reportUnknownVariableType]
                    t = getattr(p, "text", None) # pyright: ignore[reportUnknownArgumentType]
                    if isinstance(t, str):
                        texts.append(t)
            text = "".join(texts)

        if not text.strip():
            raise RuntimeError("Gemini returned no usable text content.")

        usage_meta = getattr(response, "usage_metadata", None)
        usage = Usage()
        if usage_meta is not None:
            usage = Usage(
                prompt_tokens=getattr(usage_meta, "prompt_token_count", None),
                completion_tokens=getattr(usage_meta, "candidates_token_count", None),
                total_tokens=getattr(usage_meta, "total_token_count", None),
            )

        model_name = (
            getattr(response, "model_version", None)
            or getattr(response, "model_name", None)
            or fallback_model
        )

        tool_calls: list[ToolCall] | None = None

        return LLMResponse(
            content=text,
            model=str(model_name),
            finish_reason=None,
            usage=usage,
            tool_calls=tool_calls,
            raw=response,
        )
