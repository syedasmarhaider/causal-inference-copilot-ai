from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
import os

import google.generativeai as _genai

from python.domain.service.llm_service import (
    LLMService,
    LLMConfig,
    ChatMessage,
    LLMResponse,
    Usage,
    ToolCall,
    ProviderExtra,
)

# Treat the SDK as Any so Pyright doesn’t complain about unknown members
genai = cast(Any, _genai)

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


class GeminiLLMService(LLMService):
    """
    LLMService implementation backed by Google Gemini.

    - Maps your ChatMessage[] to Gemini `contents`.
    - Uses LLMConfig for generation_config.
    - Returns LLMResponse with usage and raw payload.
    """

    def __init__(
        self,
        *,
        default_model: str = "gemini-1.5-pro-latest",
    ) -> None:
        api_key = os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self._default_model = default_model

    # ------------- public API (from LLMService) -------------

    def generate(
        self,
        *,
        config: LLMConfig,
        history: Sequence[ChatMessage],
    ) -> LLMResponse:
        model_name = config.model or self._default_model

        system_instruction = self._build_system_prompt(config, history)
        contents = self._build_contents(history)

        gen_model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction,
        )

        generation_config = self._build_generation_config(config)
        safety_settings = self._extract_safety_settings(config.extra)

        response = gen_model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        return self._to_llm_response(response, model_name)

    # ------------- helpers -------------

    @staticmethod
    def _build_system_prompt(
        config: LLMConfig,
        history: Sequence[ChatMessage],
    ) -> str | None:
        """
        Gemini supports a single system_instruction string.
        - Prefer config.system_prompt if provided.
        - Otherwise, concatenate all system messages in history.
        """
        if config.system_prompt:
            return config.system_prompt

        system_parts = [m.content for m in history if m.role == "system"]
        if not system_parts:
            return None
        return "\n\n".join(system_parts)

    @staticmethod
    def _build_contents(history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """
        Map domain ChatMessage list -> Gemini 'contents' list.

        Gemini roles: "user" | "model" | "system" | "tool"
        For now:
          - domain 'user'      -> Gemini 'user'
          - domain 'assistant' -> Gemini 'model'
          - domain 'system'    -> folded into system_instruction (skipped here)
          - domain 'tool'      -> ignored for now (tool-calls can be added later)
        """
        contents: list[dict[str, Any]] = []

        for msg in history:
            if msg.role == "system":
                continue
            if msg.role == "user":
                role = "user"
            elif msg.role == "assistant":
                role = "model"
            else:
                # 'tool' or unknown roles: skip for now
                continue

            contents.append(
                {
                    "role": role,
                    "parts": [{"text": msg.content}],
                }
            )

        return contents

    @staticmethod
    def _build_generation_config(config: LLMConfig) -> dict[str, Any]:
        """
        Map LLMConfig to Gemini generation_config dict.
        Provider extras:
          - config.extra["generation_config"] (Mapping) is shallow-merged on top.
        """
        gen_config: dict[str, Any] = {
            "temperature": config.temperature,
        }

        if config.max_tokens is not None:
            gen_config["max_output_tokens"] = config.max_tokens
        if config.top_p is not None:
            gen_config["top_p"] = config.top_p
        if config.stop:
            gen_config["stop_sequences"] = config.stop

        extra = config.extra
        if isinstance(extra, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            gc_extra = extra.get("generation_config")
            if isinstance(gc_extra, Mapping):
                gen_config.update(gc_extra)

        return gen_config

    @staticmethod
    def _extract_safety_settings(extra: ProviderExtra) -> Any:
        """
        Optionally pull safety_settings from config.extra["safety_settings"].
        Kept as Any because Gemini's safety schema is provider-specific.
        """
        if not isinstance(extra, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            return None
        return extra.get("safety_settings")

    @staticmethod
    def _to_llm_response(response: Any, fallback_model: str) -> LLMResponse:
        """
        Convert Gemini SDK response -> domain LLMResponse.

        If Gemini returns no usable text at all, we RAISE instead of silently
        returning an empty string, so downstream workflows can treat it as an error.
        """
        # --- robust text extraction ---
        content_text: str = ""

        # First try the SDK convenience accessor.
        try:
            raw_text = getattr(response, "text", None)
            if isinstance(raw_text, str):
                content_text = raw_text
        except Exception:
            # If response.text explodes (no parts / MAX_TOKENS / safety), ignore here;
            # we'll try to stitch from parts below or raise later.
            pass

        # If still empty, try to stitch from the first candidate's parts.
        if not content_text:
            candidates = getattr(response, "candidates", None)
            if candidates and isinstance(candidates, list):
                first = candidates[0] # pyright: ignore[reportUnknownVariableType]
                content = getattr(first, "content", None) # pyright: ignore[reportUnknownArgumentType]
                parts = getattr(content, "parts", None) or [] # pyright: ignore[reportUnknownVariableType]
                texts: list[str] = []
                for part in parts: # pyright: ignore[reportUnknownVariableType]
                    t = getattr(part, "text", None) # pyright: ignore[reportUnknownArgumentType]
                    if isinstance(t, str):
                        texts.append(t)
                if texts:
                    content_text = "".join(texts)

        # Extract finish_reason for error reporting / debugging.
        finish_reason: Any = None
        candidates = getattr(response, "candidates", None)
        if candidates and isinstance(candidates, list):
            first = candidates[0]  # pyright: ignore[reportUnknownVariableType]
            finish_reason = getattr(
                first, # pyright: ignore[reportUnknownArgumentType]
                "finish_reason",
                None,
            )  # pyright: ignore[reportUnknownArgumentType]

        # If we STILL have no text at this point, treat it as a hard error.
        if not content_text.strip():
            raise RuntimeError(
                f"Gemini returned no content (finish_reason={finish_reason!r}). "
                "This is treated as an error so workflows don't silently continue with empty output."
            )

        model_name: str = (
            getattr(response, "model_version", None)
            or getattr(response, "model_name", None)
            or fallback_model
        )

        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            usage = Usage(
                prompt_tokens=getattr(usage_meta, "prompt_token_count", None),
                completion_tokens=getattr(
                    usage_meta,
                    "candidates_token_count",
                    None,
                ),
                total_tokens=getattr(usage_meta, "total_token_count", None),
            )
        else:
            usage = Usage()

        # Tool calls: left as None for now (add parsing when you enable tools)
        tool_calls: list[ToolCall] | None = None

        return LLMResponse(
            content=content_text,
            model=model_name,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
            tool_calls=tool_calls,
            raw=response,
        )