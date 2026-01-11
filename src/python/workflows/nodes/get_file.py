from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Sequence, Tuple, cast
from uuid import UUID

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import LLMService
from python.workflows.utils.node_llm_message import build_node_message_with_llm
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ACTION, NEED_STAGE, ControlState, Stage, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

JSONValue = Any
JSONDictLocal = Dict[str, JSONValue]

_CSV_PATH_RE = re.compile(
    r"""(?xi)
    (?:^|[\s:=])
    (?P<q>["']?)
    (?P<p>
        (?:[a-zA-Z]:[\\/]|/|~\/|\.\.?\/)   # windows drive, /, ~/, ./, ../
        [^\n\r"']*?\.csv                  # anything up to .csv
    )
    (?P=q)
    (?:$|[\s,.;])
    """
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

GET_FILE_PROMPT = (
    "You are the GET_FILE node of a causal inference copilot.\n"
    "You receive a compact internal state snapshot as JSON.\n\n"
    "Write EXACTLY ONE message to the user.\n"
    "- Be direct, and actionable.\n"
    "- Do NOT reveal internal JSON, field names, or implementation details.\n"
    "- If intent == 'ASK_PATH': ask the user to paste a local CSV path that exists.\n"
    "- If intent == 'PATH_INVALID': explain the problem simply (if an error code exists) and ask again.\n"
    "- If intent == 'PATH_ACCEPTED': confirm the path is accepted and say you'll load it next.\n\n"
    "When asking for a path, include 2-3 examples:\n"
    "- /path/to/data.csv\n"
    "- ./data/my.csv\n"
    "- C:\\\\data\\\\file.csv\n"
)


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


def _iter_new_humans(messages: Sequence[BaseMessage], last_idx_seen: int) -> Iterator[Tuple[int, str]]:
    start = max(-1, int(last_idx_seen))
    for i in range(start + 1, len(messages)):
        m = messages[i]
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "")).strip()
            if txt:
                yield i, txt


def _extract_json_object(text: str) -> JSONDictLocal:
    s = (text or "").strip()

    m = _JSON_FENCE_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return cast(JSONDictLocal, obj)
        except Exception:
            pass

    try:
        obj2 = json.loads(s)
        if isinstance(obj2, dict):
            return cast(JSONDictLocal, obj2)
    except Exception:
        pass

    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj3, _end = dec.raw_decode(s[i:])
            if isinstance(obj3, dict):
                return cast(JSONDictLocal, obj3)
        except Exception:
            continue

    raise ValueError("No valid JSON object found.")


def _regex_extract_csv_path(user_text: str) -> str | None:
    m = _CSV_PATH_RE.search(user_text or "")
    if not m:
        return None
    p = (m.group("p") or "").strip()
    if not p:
        return None
    return p if p.lower().endswith(".csv") else None


def _llm_extract_csv_path(
    llm: LLMService,
    *,
    user_text: str,
    model_name: str,
) -> tuple[str | None, JSONDict | None]:
    sys = (
        "Extract a local CSV file path from the user's message.\n"
        "Return ONLY one JSON object with EXACTLY:\n"
        '{ "dataset_path": string | null }\n'
        "Rules:\n"
        "- If the user did not provide a path, return null.\n"
        "- Trim surrounding quotes/spaces.\n"
        "- Only return a path that ends with .csv (case-insensitive). Otherwise null.\n"
        "No markdown. No extra keys."
    )
    try:
        resp = llm.generate(
            config={"model": model_name, "temperature": 0.0},  # type: ignore[arg-type]
            history=[
                {"role": "system", "content": sys},  # type: ignore[list-item]
                {"role": "user", "content": user_text},  # type: ignore[list-item]
            ],
        )
        obj = _extract_json_object(resp.content)
        p = obj.get("dataset_path")
        if isinstance(p, str):
            p2 = p.strip().strip('"').strip("'").strip()
            if p2 and p2.lower().endswith(".csv"):
                return p2, None
        return None, None
    except Exception as e:
        return None, {"code": "LLM_PATH_PARSE_FAILED", "detail": str(e)}


def _normalize_path(p: str) -> str:
    p0 = (p or "").strip().strip('"').strip("'").strip()
    p1 = os.path.expandvars(p0)
    path = Path(p1).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path)
    try:
        path = path.resolve(strict=False)
    except Exception:
        pass
    return str(path)


def _validate_csv_path(p: str) -> tuple[bool, JSONDict | None]:
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
        pass

    return True, None


def make_get_file_node(
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
) -> Callable[[ConversationState], ConversationState]:
    """
    GET_FILE stage:
      - parse path (regex first, optional LLM fallback)
      - normalize + validate
      - when it needs to PRESENT, generate the user message via LLM and store it in control.node_message
    """

    def node(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)

        conversation_id: UUID = control_in["conversation_id"]
        stage: Stage = control_in["stage"]  # expected "GET_FILE"

        def mk_control(
            *,
            status: Status,
            post_action: ACTION,
            post_failure_suggested_stage: NEED_STAGE | None,
            node_message: str,
            last_error: JSONDict | None,
            pending_stage: Stage | None = None,
        ) -> ControlState:
            out: ControlState = cast(
                ControlState,
                {
                    **control_in,
                    "conversation_id": conversation_id,
                    "stage": stage,
                    "status": status,
                    "post_action": post_action,
                    "post_failure_suggested_stage": post_failure_suggested_stage,
                    "last_error": last_error,
                    "node_message": node_message,
                },
            )
            out["pending_stage"] = pending_stage
            return out

        messages: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))

        last_idx_seen = dataset_in.get("get_file_last_user_msg_idx", -1)
        last_idx_seen = last_idx_seen if isinstance(last_idx_seen, int) else -1

        # Helper: finalize state with an LLM-written node_message
        def finalize_with_llm(
            *,
            state_base: ConversationState,
            intent: str,
            fallback: str,
        ) -> ConversationState:
            msg = build_node_message_with_llm(
                llm,
                state=state_base,
                system_prompt=GET_FILE_PROMPT,
                intent=intent,
                model_name=model_name,
                temperature=0.4,
                history_window=10,
                fallback=fallback,
            )
            c0 = cast(ControlState, state_base["control"]) # pyright: ignore[reportUnnecessaryCast]
            c1: ControlState = {**c0, "node_message": msg}
            return {**state_base, "control": c1}

        # If we already have a path and there are no new human messages, validate and proceed.
        existing_path = dataset_in.get("path")
        any_new_human = any(True for _i, _txt in _iter_new_humans(messages, last_idx_seen))

        if isinstance(existing_path, str) and existing_path.strip() and not any_new_human:
            normalized = _normalize_path(existing_path)
            ok, err = _validate_csv_path(normalized)
            if ok:
                base_state: ConversationState = {
                    **state,
                    "dataset": {**dataset_in, "path": normalized, "load_error": None},
                    "control": mk_control(
                        status="DONE",
                        post_action="PRESENT",
                        post_failure_suggested_stage=None,
                        last_error=None,
                        node_message="",
                        pending_stage="LOAD_DATASET",
                    ),
                }
                return finalize_with_llm(
                    state_base=base_state,
                    intent="PATH_ACCEPTED",
                    fallback="✅ CSV path accepted. I’ll load the dataset next.",
                )

            base_state2: ConversationState = {
                **state,
                "dataset": {
                    **dataset_in,
                    "path": normalized,
                    "load_error": (err.get("code") if isinstance(err, dict) else "INVALID_PATH"),
                },
                "control": mk_control(
                    status="PENDING",
                    post_action="PRESENT_AND_USER_INPUT",
                    post_failure_suggested_stage=None,
                    last_error=err,
                    node_message="",
                    pending_stage=None,
                ),
            }
            return finalize_with_llm(
                state_base=base_state2,
                intent="PATH_INVALID",
                fallback=(
                    "⚠️ I can’t read that CSV path. Please paste a valid existing path ending with .csv.\n"
                    "Examples: /path/to/data.csv, ./data/my.csv, C:\\data\\file.csv"
                ),
            )

        # Consume new human messages; choose the first valid path encountered.
        last_error: JSONDict | None = None
        newest_idx_seen = last_idx_seen

        candidate_path: str | None = None
        candidate_err: JSONDict | None = None

        for idx, user_text in _iter_new_humans(messages, last_idx_seen):
            newest_idx_seen = max(newest_idx_seen, idx)

            parsed = _regex_extract_csv_path(user_text)

            llm_err: JSONDict | None = None
            if parsed is None:
                if ".csv" in user_text.lower() or "csv" in user_text.lower() or "/" in user_text or "\\" in user_text:
                    parsed, llm_err = _llm_extract_csv_path(llm, user_text=user_text, model_name=model_name)

            if parsed is None:
                if llm_err:
                    last_error = llm_err
                continue

            normalized = _normalize_path(parsed)
            ok, err = _validate_csv_path(normalized)
            if ok:
                candidate_path = normalized
                candidate_err = None
                break
            else:
                candidate_path = normalized
                candidate_err = err
                last_error = err

        if candidate_path and candidate_err is None:
            dataset_out: DatasetState = {
                **dataset_in,
                "path": candidate_path,
                "load_error": None,
                "get_file_last_user_msg_idx": newest_idx_seen,
            }
            base_state3: ConversationState = {
                **state,
                "dataset": dataset_out,
                "control": mk_control(
                    status="DONE",
                    post_action="PRESENT",
                    post_failure_suggested_stage=None,
                    last_error=None,
                    node_message="",
                    pending_stage="LOAD_DATASET",
                ),
            }
            return finalize_with_llm(
                state_base=base_state3,
                intent="PATH_ACCEPTED",
                fallback="✅ CSV path accepted. I’ll load the dataset next.",
            )

        # Still no valid path: ask user.
        dataset_out2: DatasetState = {
            **dataset_in,
            "path": candidate_path or dataset_in.get("path"),
            "load_error": (candidate_err.get("code") if isinstance(candidate_err, dict) else dataset_in.get("load_error")),
            "get_file_last_user_msg_idx": newest_idx_seen,
        }

        base_state4: ConversationState = {
            **state,
            "dataset": dataset_out2,
            "control": mk_control(
                status="PENDING",
                post_action="PRESENT_AND_USER_INPUT",
                post_failure_suggested_stage=None,
                last_error=last_error,
                node_message="",
                pending_stage=None,
            ),
        }
        return finalize_with_llm(
            state_base=base_state4,
            intent="ASK_PATH",
            fallback=(
                "Paste the full path to your CSV file (must exist and end with .csv).\n"
                "Examples:\n"
                "- /path/to/data.csv\n"
                "- ./data/my.csv\n"
                "- C:\\data\\file.csv"
            ),
        )

    return node
