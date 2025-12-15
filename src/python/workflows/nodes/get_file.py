from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, cast
import json
import re
from pathlib import Path

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import LLMService, LLMConfig, ChatMessage
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Need, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

JSONValue = Any
JSONDictLocal = Dict[str, JSONValue]

_CSV_PATH_RE = re.compile(
    r"""(?xi)
    (?:^|[\s:=])
    (?P<q>["']?)
    (?P<p>
        (?:[a-zA-Z]:\\|/|~\/)          # drive:\ or / or ~/
        [^\n\r"']*?\.csv               # anything up to .csv
    )
    (?P=q)
    (?:$|[\s,.;])
    """
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


def _last_user_text(messages: Sequence[BaseMessage]) -> str | None:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "")).strip()
            return txt or None
    return None


def _extract_json_object(text: str) -> JSONDictLocal:
    s = text.strip()

    m = _JSON_FENCE_RE.search(s)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict):
            return cast(JSONDictLocal, obj)

    try:
        obj2 = json.loads(s)
        if isinstance(obj2, dict):
            return cast(JSONDictLocal, obj2)
    except Exception:
        pass

    m2 = _JSON_OBJ_RE.search(s)
    if m2:
        obj3 = json.loads(m2.group(1))
        if isinstance(obj3, dict):
            return cast(JSONDictLocal, obj3)

    raise ValueError("LLM did not return a valid JSON object.")


def _regex_extract_csv_path(user_text: str) -> str | None:
    m = _CSV_PATH_RE.search(user_text)
    if not m:
        return None
    path = m.group("p").strip()
    return path if path.lower().endswith(".csv") else None


def _llm_extract_csv_path(
    llm: LLMService,
    *,
    user_text: str,
    model_name: str,
) -> tuple[str | None, JSONDict | None]:
    sys = (
        "You extract a local CSV file path from a user message.\n"
        "Return ONLY one JSON object with EXACTLY:\n"
        '{ "dataset_path": string | null }\n'
        "Rules:\n"
        "- If user did not provide a path, return null.\n"
        "- Preserve the path exactly (except trim surrounding quotes/spaces).\n"
        "- Only return paths that end with .csv (case-insensitive). Otherwise null.\n"
        "No markdown. No extra keys."
    )
    history = [
        ChatMessage(role="system", content=sys),
        ChatMessage(role="user", content=user_text),
    ]
    cfg = LLMConfig(model=model_name, temperature=1.0)

    try:
        resp = llm.generate(config=cfg, history=history)
        obj = _extract_json_object(resp.content)
        p = obj.get("dataset_path")
        if isinstance(p, str):
            p2 = p.strip().strip('"').strip("'").strip()
            if p2.lower().endswith(".csv") and p2:
                return p2, None
        return None, None
    except Exception as e:
        return None, {"code": "LLM_PATH_PARSE_FAILED", "detail": str(e)}


def _normalize_path(p: str) -> str:
    # Expand ~ and environment variables, keep absolute if possible
    # (Path.expanduser does ~, expandvars is os.path.expandvars)
    import os
    p2 = os.path.expandvars(p.strip())
    return str(Path(p2).expanduser())


def _validate_csv_path(p: str) -> tuple[bool, JSONDict | None]:
    """
    Returns (ok, error_json).
    Error_json includes a 'code' + 'detail' for UI.
    """
    try:
        path = Path(p)
    except Exception as e:
        return False, {"code": "INVALID_PATH", "detail": str(e)}

    if not p.lower().endswith(".csv"):
        return False, {"code": "NOT_CSV", "detail": f"Path does not end with .csv: {p!r}"}

    if not path.exists():
        return False, {"code": "PATH_NOT_FOUND", "detail": f"File not found: {p!r}"}

    if not path.is_file():
        return False, {"code": "NOT_A_FILE", "detail": f"Path is not a file: {p!r}"}

    # Readability check (simple): try opening a few bytes
    try:
        with path.open("rb") as f:
            _ = f.read(256)
    except Exception as e:
        return False, {"code": "NOT_READABLE", "detail": str(e)}

    # Optional: size sanity (0 bytes likely wrong)
    try:
        if path.stat().st_size == 0:
            return False, {"code": "EMPTY_FILE", "detail": f"CSV file is empty: {p!r}"}
    except Exception:
        # ignore stat errors, we already know we can read it
        pass

    return True, None


def make_get_file_node(
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
) -> Callable[[ConversationState], ConversationState]:
    """
    GET_FILE stage (path owner).

    Responsibilities:
      - Extract a CSV path from user input (regex-first, LLM fallback)
      - Normalize (~, env vars)
      - Validate that the file exists and is readable
      - On success: set dataset.path and return status DONE, need NONE
      - On "user must fix input": status PENDING/RETRYABLE_ERROR, need PRESENT_AND_USER_INPUT
      - On fatal/unrecoverable: status ABORTED (router/orchestrator should bounce back to GET_FILE)
    """

    def get_file(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)

        conversation_id = control_in["conversation_id"]
        stage = control_in["stage"]  # should be "GET_FILE"

        def mk_control(
            *,
            status: Status,
            need: Need,
            node_message: str,
            last_error: JSONDict | None,
        ) -> ControlState:
            # keep any extra control keys you might add later
            return cast(
                ControlState,
                {
                    **control_in,
                    "conversation_id": conversation_id,
                    "stage": stage,
                    "status": status,
                    "need": need,
                    "last_error": last_error,
                    "node_message": node_message,
                },
            )

        # If a path already exists in state, validate it here too (you wanted GET_FILE to own this).
        existing_path = dataset_in.get("path")
        if isinstance(existing_path, str) and existing_path.strip():
            normalized = _normalize_path(existing_path)
            ok, err = _validate_csv_path(normalized)
            if ok:
                return {
                    **state,
                    "dataset": {**dataset_in, "path": normalized},
                    "control": mk_control(
                        status="DONE",
                        need="PRESENT",          
                        last_error=None,
                        node_message="File path is valid",
                    ),
                }
            # Existing path is invalid -> ask user again
            return {
                **state,
                "dataset": {**dataset_in, "path": normalized, "load_error": err.get("code") if isinstance(err, dict) else "INVALID_PATH"},
                "control": mk_control(
                    status="RETRYABLE_ERROR",
                    need="PRESENT_AND_USER_INPUT",
                    last_error=err,
                    node_message=(
                        "The current dataset path is invalid/unreadable.\n"
                        "Please paste a valid existing CSV path."
                    ),
                ),
            }

        # No path: ask user
        prior_msgs = cast(Sequence[BaseMessage], state.get("messages", []))
        user_text = _last_user_text(prior_msgs)
        if not user_text:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    need="PRESENT_AND_USER_INPUT",
                    last_error=None,
                    node_message=(
                        "Paste the full path to your CSV file (must exist and end with .csv).\n"
                        "Example: /path/to/data.csv"
                    ),
                ),
            }

        # Parse user message -> path
        parsed_path = _regex_extract_csv_path(user_text)
        llm_err: JSONDict | None = None
        if parsed_path is None:
            parsed_path, llm_err = _llm_extract_csv_path(
                llm,
                user_text=user_text,
                model_name=model_name,
            )

        if parsed_path is None:
            # Not fatal: user can try again
            return {
                **state,
                "control": mk_control(
                    status="RETRYABLE_ERROR" if llm_err else "PENDING",
                    need="PRESENT_AND_USER_INPUT",
                    last_error=llm_err,
                    node_message=(
                        "I couldn’t detect a valid .csv path.\n"
                        "Please paste the full absolute path to an existing CSV file."
                    ),
                ),
            }

        normalized = _normalize_path(parsed_path)
        ok, err = _validate_csv_path(normalized)
        if not ok:
            # Still not fatal: user can fix the path
            return {
                **state,
                "dataset": {**dataset_in, "path": normalized, "load_error": err.get("code") if isinstance(err, dict) else "INVALID_PATH"},
                "control": mk_control(
                    status="RETRYABLE_ERROR",
                    need="PRESENT_AND_USER_INPUT",
                    last_error=err,
                    node_message=(
                        "That path doesn’t point to a readable CSV file.\n"
                        "Please paste a valid existing CSV path."
                    ),
                ),
            }

        # Success: store the path; LOAD_DATASET will do parsing/loading
        dataset_out: DatasetState = {**dataset_in, "path": normalized, "load_error": None}

        return {
            **state,
            "dataset": dataset_out,
            "control": mk_control(
                status="DONE",
                need="PRESENT",     
                last_error=None, 
                node_message="File path is valid",
            ),
        }

    return get_file
