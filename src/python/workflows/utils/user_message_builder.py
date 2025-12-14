# pyright: ignore[reportUnknownVariableType]
from __future__ import annotations

from typing import Any, Dict, List, Sequence, cast
import json

from langchain_core.messages import AIMessage, BaseMessage

from python.domain.service.llm_service import LLMService, LLMConfig, ChatMessage
from python.workflows.state.conversation_state import ConversationState

JSONDict = Dict[str, Any]


def _role(m: BaseMessage) -> str:
    t = getattr(m, "type", None)  # "human" | "ai" | "system" | "tool"
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(str(t), "user")


def _truncate(s: Any, *, limit: int) -> str | None:
    if s is None:
        return None
    txt = str(s)
    if len(txt) <= limit:
        return txt
    return txt[:limit] + "…"


def _schema_preview(raw_schema: Any, *, max_cols: int = 40) -> JSONDict:
    """
    raw_schema shape (your loader): {"columns": [{"name": str, "dtype": str}, ...]}
    Returns a compact preview safe to feed into the presenter LLM.
    """
    if not isinstance(raw_schema, dict):
        return {"n_columns": None, "columns": []}

    cols = raw_schema.get("columns") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(cols, list):
        return {"n_columns": None, "columns": []}

    cleaned: List[JSONDict] = []
    for c in cols: # type: ignore
        if isinstance(c, dict):
            name = c.get("name") # type: ignore
            dtype = c.get("dtype") # type: ignore
            if isinstance(name, str) and name:
                cleaned.append({"name": name, "dtype": str(dtype) if dtype is not None else None}) # type: ignore

    n_total = len(cleaned)
    preview = cleaned[:max_cols]
    return {"n_columns": n_total, "columns": preview, "truncated": n_total > max_cols}


def _compact_proposed_design(design: Any) -> JSONDict | None:
    """
    Keep only the bits that help the user decide, avoid dumping huge JSON.
    """
    if not isinstance(design, dict):
        return None

    def take_list(key: str, n: int) -> List[str]:
        v = design.get(key) # type: ignore
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for x in v[:n]: # type: ignore
            s = str(x).strip() # type: ignore
            if s:
                out.append(s)
        return out

    return {
        "dataset_summary": _truncate(design.get("dataset_summary"), limit=800) or "", # type: ignore
        "treatment_candidates": take_list("treatment_candidates", 8),
        "outcome_candidates": take_list("outcome_candidates", 8),
        "controls_candidates": take_list("controls_candidates", 12),
        "effect_modifier_candidates": take_list("effect_modifier_candidates", 12),
        "effect_examples": take_list("effect_examples", 5),
        "questions_for_user": take_list("questions_for_user", 6),
    }


def _compact_final_design(final_design: Any) -> JSONDict | None:
    if not isinstance(final_design, dict):
        return None

    def get_col(path: List[str]) -> str | None:
        cur: Any = final_design # pyright: ignore[reportUnknownVariableType]
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k) # type: ignore
        return cur if isinstance(cur, str) and cur else None

    def get_cols(path: List[str]) -> List[str]:
        cur: Any = final_design # type: ignore
        for k in path:
            if not isinstance(cur, dict):
                return []
            cur = cur.get(k) # type: ignore
        if not isinstance(cur, list):
            return []
        out: List[str] = []
        for x in cur: # type: ignore
            s = str(x).strip() # type: ignore
            if s:
                out.append(s)
        return out

    return {
        "treatment": get_col(["treatment", "column"]),
        "outcome": get_col(["outcome", "column"]),
        "controls": get_cols(["controls", "columns"]),
        "effect_modifiers": get_cols(["effect_modifiers", "columns"]),
        "causal_question": _truncate(final_design.get("causal_question"), limit=500), # pyright: ignore[reportUnknownMemberType]
        "accepted_by_user": bool(final_design.get("accepted_by_user", False)), # pyright: ignore[reportUnknownArgumentType] # pyright: ignore[reportUnknownMemberType] # type: ignore
    }


def build_user_message_with_llm(
    llm: LLMService,
    state: ConversationState,
    *,
    model_name: str = "gemini-1.5-flash",
    history_window: int = 12,
    max_error_detail_chars: int = 1200,
) -> AIMessage:
    """
    Presenter layer:
      - Takes recent convo history + compact internal state
      - Produces exactly ONE user-facing message
    """
    control = cast(JSONDict, state.get("control", {}))
    dataset = cast(JSONDict, state.get("dataset", {}))
    metadata = cast(JSONDict, state.get("metadata", {}))

    last_error = control.get("last_error")
    if isinstance(last_error, dict):
        # keep error compact; details can explode
        last_error = { # pyright: ignore[reportUnknownVariableType]
            "code": last_error.get("code"), # pyright: ignore[reportUnknownMemberType]
            "detail": _truncate(last_error.get("detail"), limit=max_error_detail_chars), # pyright: ignore[reportUnknownMemberType]
        }

    payload: JSONDict = {
        "control": {
            "stage": control.get("stage"),
            "status": control.get("status"),
            "outcome": control.get("outcome"),
            "need": control.get("need"),
            "interrupt_type": control.get("interrupt_type"),
            "node_message": _truncate(control.get("node_message"), limit=700),
            "last_error": last_error,
        },
        "dataset": {
            "path": dataset.get("path"),
            "id": str(dataset.get("id")) if dataset.get("id") is not None else None,
            "summary": dataset.get("summary"),
            "load_error": dataset.get("load_error"),
            "schema_preview": _schema_preview(dataset.get("raw_schema")),
        },
        "metadata": {
            "treatment_hint": metadata.get("treatment_hint"),
            "outcome_hint": metadata.get("outcome_hint"),
            "controls_hint": metadata.get("controls_hint"),
            "effect_modifiers_hint": metadata.get("effect_modifiers_hint"),
            "proposed_design": _compact_proposed_design(metadata.get("proposed_design")),
            "final_design": _compact_final_design(metadata.get("final_design")),
            "user_accepts": metadata.get("user_accepts"),
        },
    }

    prior: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))
    tail = list(prior)[-history_window:] if isinstance(prior, list) else []

    system_prompt = (
        "You are the user-facing voice of a causal inference copilot.\n"
        "You receive (a) a short chat history and (b) a compact internal state snapshot.\n\n"
        "Write EXACTLY ONE message to the user.\n"
        "- Be brief, concrete, and actionable.\n"
        "- Do NOT expose raw JSON or internal field names.\n"
        "- If there is an error: explain it simply + give the next step.\n"
        "- If the system needs user input, ask only the minimum necessary questions.\n\n"
        "Guidance by 'need':\n"
        "- DATASET_PATH: ask for a CSV path.\n"
        "- TREATMENT_OUTCOME: ask for treatment (T) and outcome (Y) column names.\n"
        "- CONFIRM_METADATA: ask for T, Y, and ALSO controls/confounders (W). Optionally ask effect modifiers (X).\n"
        "- ESTIMATOR_CHOICE / FIT_CONFIRMATION / EFFECT_PLAN_CONFIRMATION: ask that specific decision.\n\n"
        "If schema_preview.columns is present, you may show a short list of column names to help the user copy-paste.\n"
    )

    history: List[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

    for m in tail:
        history.append(ChatMessage(role=cast(Any, _role(m)), content=str(getattr(m, "content", ""))))

    user_prompt = (
        "Internal state snapshot (JSON):\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )
    history.append(ChatMessage(role="user", content=user_prompt))

    config = LLMConfig(model=model_name, temperature=0.2, max_tokens=520)

    try:
        resp = llm.generate(config=config, history=history)
        text = resp.content.strip()
        # small guard: never return empty
        if not text:
            text = str(control.get("node_message") or "Tell me what you want to do next.")
        return AIMessage(content=text)
    except Exception:
        fallback = control.get("node_message") or "Tell me what you want to do next."
        return AIMessage(content=str(fallback))
