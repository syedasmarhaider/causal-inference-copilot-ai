from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, cast

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.state.conversation_state import ConversationState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI

JSONDict = Dict[str, Any]


def _role(m: BaseMessage) -> str:
    t = getattr(m, "type", None)  # "human" | "ai" | "system" | "tool"
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(str(t), "user")


def _truncate(v: Any, limit: int) -> str | None:
    if v is None:
        return None
    s = str(v)
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def _safe_dict(obj: Any) -> JSONDict:
    return cast(JSONDict, obj) if isinstance(obj, dict) else {}


def _schema_preview(raw_schema: Any, *, max_cols: int = 40) -> JSONDict:
    """
    raw_schema shape: {"columns": [{"name": str, "dtype": str}, ...]}
    """
    if not isinstance(raw_schema, dict):
        return {"n_columns": None, "columns": []}

    cols = raw_schema.get("columns")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(cols, list):
        return {"n_columns": None, "columns": []}

    cleaned: List[JSONDict] = []
    for c in cols:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(c, dict):
            name = c.get("name")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            dtype = c.get("dtype")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if isinstance(name, str) and name:
                cleaned.append(
                    {"name": name, "dtype": str(dtype) if dtype is not None else None} # pyright: ignore[reportUnknownArgumentType]
                )  # pyright: ignore[reportUnknownArgumentType]

    n_total = len(cleaned)
    preview = cleaned[:max_cols]
    return {"n_columns": n_total, "columns": preview, "truncated": n_total > max_cols}


def build_node_message_with_llm(
    llm: LLMService,
    *,
    state: ConversationState,
    system_prompt: str,
    intent: str,
    model_name: str = DEFAULT_MODEL_GEMNI,
    temperature: float = 0.4,
    history_window: int = 10,
    fallback: str,
) -> str:
    """
    Node-level presenter.

    Contract:
      - Call ONLY when node is about to PRESENT.
      - Returns exactly one user-facing string.
      - Never raises; returns `fallback` on any error.

    Prompting strategy (as you requested):
      - system: node-specific instructions (system_prompt)
      - tail: last N conversation messages (optional)
      - user: "You are a causal inference copilot. Here is the node's draft message. Here is the current state JSON.
              Follow system instructions and write the final user-facing message."
    """
    try:
        control = _safe_dict(state.get("control"))
        dataset = _safe_dict(state.get("dataset"))
        metadata = _safe_dict(state.get("metadata"))

        last_error = control.get("last_error")
        if isinstance(last_error, dict):
            last_error = {  # pyright: ignore[reportUnknownVariableType]
                "code": last_error.get("code"),  # pyright: ignore[reportUnknownMemberType]
                "detail": _truncate(last_error.get("detail"), 1200),  # pyright: ignore[reportUnknownMemberType]
            }

        # Give LLM stable, compact state.
        payload: JSONDict = {
            "intent": intent,
            "control": {
                "stage": control.get("stage"),
                "status": control.get("status"),
                "post_action": control.get("post_action"),
                "pending_stage": control.get("pending_stage"),
                "node_message_len": len(str(control.get("node_message") or "")),
                "last_error": last_error,
            },
            "dataset": {
                "path": dataset.get("path"),
                "id": str(dataset.get("id")) if dataset.get("id") is not None else None,
                "load_error": dataset.get("load_error"),
                "summary": dataset.get("summary"),
                "schema_preview": _schema_preview(dataset.get("raw_schema")),
            },
            "metadata": {
                "draft": metadata.get("draft"),
                "final_design": metadata.get("final_design"),
                "validation_report": metadata.get("validation_report"),
                "warnings": metadata.get("warnings"),
            },
            # Important: give the node's deterministic “draft” to improve reliability.
            "node_default_message": _truncate(fallback, 1500),
        }

        prior: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))
        tail = list(prior)[-history_window:] if isinstance(prior, list) else []

        history: List[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

        for m in tail:
            history.append(
                ChatMessage(
                    role=cast(Any, _role(m)),
                    content=str(getattr(m, "content", "")),
                )
            )

        user_payload = json.dumps(payload, ensure_ascii=False)

        user_prompt = (
            "You are a causal inference copilot.\n"
            "A workflow node is about to PRESENT a message to the user.\n\n"
            "Node's draft/default message (may be empty or generic):\n"
            f"{fallback.strip()}\n\n"
            "Current conversation state (JSON):\n"
            f"{user_payload}\n\n"
            "Follow the system instructions and write EXACTLY ONE final user-facing message."
        )

        history.append(ChatMessage(role="user", content=user_prompt))

        cfg = LLMConfig(model=model_name, temperature=temperature)
        resp = llm.generate(config=cfg, history=history)
        txt = resp.content.strip()

        return txt if txt else fallback
    except Exception:
        return fallback
