from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any,  Dict,  Sequence, Tuple, cast

from langchain_core.messages import AIMessage, BaseMessage

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.control_state import ACTION, NEED_STAGE, ControlState, Stage, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

JSONDictLocal = Dict[str, Any]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


# -----------------------------
# LLM prompts (delegate everything user-facing)
# -----------------------------

GET_FILE_EXTRACT_PROMPT = """
You extract a LOCAL CSV file path from the user's message.

Return ONLY one JSON object with EXACTLY:
{ "dataset_path": string | null }

Rules:
- If the user did not provide a local path, return null.
- Trim surrounding quotes/spaces.
- Only return a path that ends with ".csv" (case-insensitive), otherwise null.
- Do NOT include any other keys. No markdown.
""".strip()

GET_FILE_MESSAGE_PROMPT = """
You are the GET_FILE node of a causal inference copilot.

You receive a compact internal snapshot as JSON (includes validation results).
Write EXACTLY ONE message to the user.

Rules:
- Be comprehensive and actionable.
- Do NOT reveal internal JSON, field names, or implementation details.
- If validation.ok == true: confirm the path is accepted and say you'll load it next.
- If validation.ok == false:
  - If validation.code == "NO_PATH": ask the user to paste a local CSV path that exists.
  - Otherwise: explain the problem simply (based on validation.code) and ask for a valid existing CSV path again.
- When asking for a path, include 2–3 examples:
  - /path/to/data.csv
  - ./data/my.csv
  - C:\\data\\file.csv

Output ONLY the message text. No markdown fences.
""".strip()


# -----------------------------
# Minimal state helpers
# -----------------------------

def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    ds = state.get("dataset", {})
    return  ds 


def _append_final_ai_message(state: ConversationState, content: str) -> None:
    """
    Append ONLY the final user-facing message. Never append LLM prompt/system payloads.
    """
    msgs = state.get("messages")
    msgs.append(
        AIMessage(
            content=content,
            additional_kwargs={"source": "node", "stage": "GET_FILE"},
        )
    )


def _last_new_human_message(
    messages: Sequence[BaseMessage],
    last_idx_seen: int,
) -> Tuple[int, str] | None:
    start = max(-1, int(last_idx_seen))
    for i in range(len(messages) - 1, start, -1):
        m = messages[i]
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "") or "").strip()
            if txt:
                return i, txt
    return None


# -----------------------------
# JSON parsing (strict)
# -----------------------------

def _parse_json_object_strict(text: str) -> JSONDictLocal:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty LLM response")

    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root must be an object")
    return cast(JSONDictLocal, obj)


# -----------------------------
# File path normalization + validation (deterministic)
# -----------------------------

def _normalize_path(p: str) -> str:
    p0 = (p or "").strip().strip('"').strip("'").strip()
    p1 = os.path.expandvars(p0)
    path = Path(p1).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        path = path.resolve(strict=False)
    except Exception:
        pass
    return str(path)


def _validate_csv_path(p: str) -> Tuple[bool, JSONDict | None]:
    try:
        path = Path(p)
    except Exception as e:
        return False, {"code": "INVALID_PATH", "detail": str(e)}

    if not str(p).lower().endswith(".csv"):
        return False, {"code": "NOT_CSV", "detail": f"Path does not end with .csv: {p!r}"}

    if not path.exists():
        return False, {"code": "PATH_NOT_FOUND", "detail": f"File not found: {p!r}"}

    if not path.is_file():
        return False, {"code": "NOT_A_FILE", "detail": f"Path is not a file: {p!r}"}

    try:
        with path.open("rb") as f:
            _ = f.read(256)
    except Exception as e:
        return False, {"code": "NOT_READABLE", "detail": str(e)}

    try:
        if path.stat().st_size == 0:
            return False, {"code": "EMPTY_FILE", "detail": f"CSV file is empty: {p!r}"}
    except Exception:
        # non-fatal; keep going
        pass

    return True, None


# -----------------------------
# LLM calls (strict, no fallback)
# -----------------------------

def _llm_extract_csv_path_strict(
    llm: LLMService,
    *,
    model_name: str,
    user_text: str,
) -> str | None:
    config: LLMConfig = LLMConfig(
        model=model_name,
        temperature=0.0,
    )
    history: Sequence[ChatMessage] = [
        ChatMessage(role="system", content=GET_FILE_EXTRACT_PROMPT),
        ChatMessage(role="user", content=user_text),
    ]
    resp = llm.generate(config=config, history=history)
    obj = _parse_json_object_strict(cast(Any, resp).content)
    if set(obj.keys()) != {"dataset_path"}:
        raise ValueError("Extractor must return exactly: { 'dataset_path': ... }")

    p = obj.get("dataset_path")
    if p is None:
        return None
    if isinstance(p, str):
        p2 = p.strip().strip('"').strip("'").strip()
        if p2 and p2.lower().endswith(".csv"):
            return p2
        return None
    raise ValueError("Extractor returned non-string/non-null dataset_path")


def _llm_write_node_message_strict(
    llm: LLMService,
    *,
    model_name: str,
    snapshot: JSONDict,
) -> str:
    config: LLMConfig = LLMConfig(
        model=model_name,
        temperature=0.0,
    )
    history: Sequence[ChatMessage] = [
        ChatMessage(role="system", content=GET_FILE_MESSAGE_PROMPT),
        ChatMessage(role="user", content=json.dumps(snapshot, ensure_ascii=False)),
    ]
    resp = llm.generate(config=config, history=history)
    if not resp.content:
        raise ValueError("Node message LLM returned empty message")
    return resp.content


# -----------------------------
# Control builder (new schema)
# -----------------------------

def _mk_control(
    *,
    current_stage: Stage,
    current_stage_status: Status,
    action_required: ACTION,
    post_failure_suggested_stage: NEED_STAGE | None,
    node_message: str,
) -> ControlState:
    return cast(
        ControlState,
        {
            "current_stage": current_stage,
            "current_stage_status": current_stage_status,
            "action_required": action_required,
            "post_failure_suggested_stage": post_failure_suggested_stage,
            "node_message": node_message,
        },
    )


# -----------------------------
# Node (delegate nearly everything to LLM)
# -----------------------------

def make_get_file_node(
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
    append_ai_message: bool = True,
) -> CallableNodeFunc:
    """
    GET_FILE node:
    - LLM extracts candidate dataset_path from the newest unprocessed human message (no regex heuristics).
    - Node normalizes + validates path deterministically (exists/readable/etc).
    - LLM writes the user-facing message from validation outcome.
    - No fallbacks: if LLM is down/invalid => mark ABORTED and raise.
    """

    def node(state: ConversationState) -> ConversationState:
        _ = _require_control(state)  # must exist
        dataset_in = _as_dataset(state)

        messages = cast(Sequence[BaseMessage], state.get("messages", []))

        last_idx_seen = dataset_in.get("get_file_last_user_msg_idx", -1)
        last_idx_seen = last_idx_seen if isinstance(last_idx_seen, int) else -1

        newest = _last_new_human_message(messages, last_idx_seen)
        user_idx, user_text = newest if newest else (-1, "")

        # 1) LLM extracts path (or null). If no new message, treat as no path.
        try:
            extracted_path = None
            if user_text:
                extracted_path = _llm_extract_csv_path_strict(
                    llm,
                    model_name=model_name,
                    user_text=user_text,
                )
        except Exception as e:
            state["control"] = _mk_control(
                current_stage="GET_FILE",
                current_stage_status="ABORTED",
                action_required="NEEDS_INPUT",
                post_failure_suggested_stage="GET_FILE",
                node_message="",
            )
            raise ValueError(f"GET_FILE: LLM path extraction failed: {e}") from e

        # 2) Validate (deterministic)
        normalized_path: str | None = None
        ok = False
        err: JSONDict | None = None

        if extracted_path is None:
            ok = False
            err = {"code": "NO_PATH", "detail": "No CSV path found in user message."}
        else:
            normalized_path = _normalize_path(extracted_path)
            ok, err = _validate_csv_path(normalized_path)

        # 3) Update dataset + control routing (deterministic)
        newest_seen = last_idx_seen
        if user_idx >= 0:
            newest_seen = max(newest_seen, user_idx)

        if ok:
            dataset_out: DatasetState = {
                **dataset_in,
                "path": cast(str, normalized_path),
                "load_error": None,
                "get_file_last_user_msg_idx": newest_seen,
            }
            base_state: ConversationState = {
                **state,
                "dataset": dataset_out,
                "control": _mk_control(
                    current_stage="LOAD_DATASET",
                    current_stage_status="PENDING",
                    action_required="NONE",
                    post_failure_suggested_stage=None,
                    node_message="",
                ),
            }
        else:
            dataset_out2: DatasetState = {
                **dataset_in,
                "path": normalized_path or dataset_in.get("path"),
                "load_error": (err.get("code") if isinstance(err, dict) else "PATH_INVALID"),
                "get_file_last_user_msg_idx": newest_seen,
            }
            # Keep details internal for the LLM message (prompt says “if an error code exists”)
            if isinstance(err, dict) and err.get("detail") is not None:
                dataset_out2 = cast(DatasetState, {**dataset_out2, "load_error_detail": str(err["detail"])})

            base_state = {
                **state,
                "dataset": dataset_out2,
                "control": _mk_control(
                    current_stage="GET_FILE",
                    current_stage_status="PENDING",
                    action_required="NEEDS_INPUT",
                    post_failure_suggested_stage=None,
                    node_message="",
                ),
            }

        # 4) LLM writes user-facing node message (strict, no fallback)
        snapshot: JSONDict = {
            "node": "GET_FILE",
            "user_message_present": bool(user_text),
            "extracted_path": extracted_path,
            "normalized_path": normalized_path,
            "validation": {
                "ok": ok,
                "code": (err.get("code") if isinstance(err, dict) else None),
                "detail": (err.get("detail") if isinstance(err, dict) else None),
            },
        }

        try:
            node_msg = _llm_write_node_message_strict(
                llm,
                model_name=model_name,
                snapshot=snapshot,
            )
        except Exception as e:
            base_state["control"] = _mk_control(
                current_stage="GET_FILE",
                current_stage_status="ABORTED",
                action_required="NEEDS_INPUT",
                post_failure_suggested_stage="GET_FILE",
                node_message="",
            )
            raise ValueError(f"GET_FILE: LLM message generation failed: {e}") from e

        # commit message
        c0 =  base_state["control"]
        base_state["control"] = cast(ControlState, {**c0, "node_message": node_msg})

        if append_ai_message:
            _append_final_ai_message(base_state, node_msg)

        return base_state

    return node
