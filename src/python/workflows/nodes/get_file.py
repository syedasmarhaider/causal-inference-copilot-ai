from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator,  Sequence, Tuple, cast
from uuid import UUID

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ACTION, NEED_STAGE, ControlState, Stage, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

JSONValue = Any
JSONDictLocal = Dict[str, JSONValue]

# Accept absolute and common relative paths (Unix + Windows).
# Examples:
#   /a/b/data.csv
#   ~/data.csv
#   ./data.csv
#   ../data.csv
#   C:\data\file.csv
#   C:/data/file.csv
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


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


def _iter_new_humans(messages: Sequence[BaseMessage], last_idx_seen: int) -> Iterator[Tuple[int, str]]:
    """
    Processes all new human messages since last_idx_seen, in order.
    This avoids losing instructions if multiple user inputs arrive before the node runs.
    """
    start = max(-1, int(last_idx_seen))
    for i in range(start + 1, len(messages)):
        m = messages[i]
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "")).strip()
            if txt:
                yield i, txt


def _extract_json_object(text: str) -> JSONDictLocal:
    """
    Robust JSON object extraction:
      - fenced blocks
      - pure JSON
      - substring decode (first valid object found)
    """
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
    """
    LLM fallback only when regex fails. Keep it deterministic.
    """
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
    cfg = LLMConfig(model=model_name, temperature=0.0)
    try:
        resp = llm.generate(
            config=cfg,
            history=[
                ChatMessage(role="system", content=sys),
                ChatMessage(role="user", content=user_text),
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
    """
    Normalize user input into a stable absolute-ish path:
      - expand env vars + ~
      - if relative, resolve against cwd
      - resolve(strict=False) so it still produces an absolute path even if missing
    """
    p0 = (p or "").strip().strip('"').strip("'").strip()
    p1 = os.path.expandvars(p0)
    path = Path(p1).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path)
    # strict=False is important: we want deterministic normalization even when missing.
    try:
        path = path.resolve(strict=False)
    except Exception:
        # best-effort; return what we have
        pass
    return str(path)


def _validate_csv_path(p: str) -> tuple[bool, JSONDict | None]:
    """
    Returns (ok, error_json).
    Error_json includes a 'code' + 'detail' for UI/runtime.
    """
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

    # Readability check: try reading a small prefix
    try:
        with path.open("rb") as f:
            _ = f.read(256)
    except Exception as e:
        return False, {"code": "NOT_READABLE", "detail": str(e)}

    # Basic sanity
    try:
        if path.stat().st_size == 0:
            return False, {"code": "EMPTY_FILE", "detail": f"CSV file is empty: {p!r}"}
    except Exception:
        # If stat fails but open/read worked, we still accept.
        pass

    return True, None


def make_get_file_node(
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
) -> Callable[[ConversationState], ConversationState]:
    """
    GET_FILE stage (dataset path owner).

    Responsibilities:
      - Extract a CSV path from user input (regex-first, LLM fallback)
      - Normalize path (~, env vars, relative -> absolute)
      - Validate file exists and is readable
      - Persist: dataset.path + dataset.get_file_last_user_msg_idx
      - ControlState:
          - DONE + post_action=PRESENT + pending_stage=LOAD_DATASET on success
          - PENDING + post_action=PRESENT_AND_USER_INPUT on user action needed
          - ABORTED only for truly fatal internal/state corruption
    """

    def node(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)

        conversation_id: UUID = control_in["conversation_id"]
        stage: Stage = control_in["stage"]  # should be "GET_FILE"

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

        # Track which user messages this node already consumed (prevents re-parsing old inputs).
        last_idx_seen = dataset_in.get("get_file_last_user_msg_idx", -1)
        last_idx_seen = last_idx_seen if isinstance(last_idx_seen, int) else -1 # pyright: ignore[reportUnnecessaryIsInstance]

        # If we already have a path and there are no new human messages, validate and proceed.
        existing_path = dataset_in.get("path")
        any_new_human = any(True for _i, _txt in _iter_new_humans(messages, last_idx_seen))

        if isinstance(existing_path, str) and existing_path.strip() and not any_new_human:
            normalized = _normalize_path(existing_path)
            ok, err = _validate_csv_path(normalized)
            if ok:
                return {
                    **state,
                    "dataset": {**dataset_in, "path": normalized, "load_error": None},
                    "control": mk_control(
                        status="DONE",
                        post_action="PRESENT",
                        post_failure_suggested_stage=None,
                        last_error=None,
                        node_message="✅ CSV path is valid.",
                        pending_stage="LOAD_DATASET",
                    ),
                }
            return {
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
                    node_message=(
                        "⚠️ The current dataset path is invalid/unreadable.\n"
                        "Please paste a valid existing CSV path (absolute or relative).\n"
                        "Examples:\n"
                        "- /path/to/data.csv\n"
                        "- ./data/my.csv\n"
                        "- C:\\data\\file.csv"
                    ),
                ),
            }

        # Consume new human messages; choose the last valid path we find.
        last_error: JSONDict | None = None
        newest_idx_seen = last_idx_seen

        candidate_path: str | None = None
        candidate_err: JSONDict | None = None

        for idx, user_text in _iter_new_humans(messages, last_idx_seen):
            newest_idx_seen = max(newest_idx_seen, idx)

            # Fast path: regex extraction
            parsed = _regex_extract_csv_path(user_text)

            # Only call LLM if regex failed AND the message plausibly contains a path hint
            llm_err: JSONDict | None = None
            if parsed is None:
                if ".csv" in user_text.lower() or "csv" in user_text.lower() or "/" in user_text or "\\" in user_text:
                    parsed, llm_err = _llm_extract_csv_path(llm, user_text=user_text, model_name=model_name)

            if parsed is None:
                # Keep the most informative parse error (if any), but continue
                if llm_err:
                    last_error = llm_err
                continue

            normalized = _normalize_path(parsed)
            ok, err = _validate_csv_path(normalized)
            if ok:
                candidate_path = normalized
                candidate_err = None
                break  # first valid in-order is enough
            else:
                candidate_path = normalized
                candidate_err = err
                last_error = err

        # If we found a valid candidate: persist and advance
        if candidate_path and candidate_err is None:
            dataset_out: DatasetState = {
                **dataset_in,
                "path": candidate_path,
                "load_error": None,
                "get_file_last_user_msg_idx": newest_idx_seen,
            }
            return {
                **state,
                "dataset": dataset_out,
                "control": mk_control(
                    status="DONE",
                    post_action="PRESENT",
                    post_failure_suggested_stage=None,
                    last_error=None,
                    node_message="✅ CSV path accepted.",
                    pending_stage="LOAD_DATASET",
                ),
            }

        # No valid path yet: prompt user again (non-fatal)
        dataset_out2: DatasetState = {
            **dataset_in,
            "path": candidate_path or dataset_in.get("path"),
            "load_error": (candidate_err.get("code") if isinstance(candidate_err, dict) else dataset_in.get("load_error")),
            "get_file_last_user_msg_idx": newest_idx_seen,
        }

        return {
            **state,
            "dataset": dataset_out2,
            "control": mk_control(
                status="PENDING",
                post_action="PRESENT_AND_USER_INPUT",
                post_failure_suggested_stage=None,
                last_error=last_error,
                node_message=(
                    "Paste the full path to your CSV file (must exist and end with .csv).\n"
                    "Examples:\n"
                    "- /path/to/data.csv\n"
                    "- ./data/my.csv\n"
                    "- C:\\data\\file.csv"
                ),
            ),
        }

    return node
