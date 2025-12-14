from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, cast
import json
import re

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import LLMService, LLMConfig, ChatMessage
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Need, Outcome, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import JSONDict

JSONValue = Any
JSONDictLocal = Dict[str, JSONValue]

# crude but effective: Unix + Windows-ish CSV paths (quoted or unquoted)
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
    return cast(ControlState, state["control"])  # type: ignore # invariant


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {})) # type: ignore


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
    # minimal guard: must end with .csv
    return path if path.lower().endswith(".csv") else None


def _llm_extract_csv_path(
    llm: LLMService,
    *,
    user_text: str,
    model_name: str,
) -> tuple[str | None, JSONDict | None]:
    """
    Returns: (path_or_none, error_or_none)
    """
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
    cfg = LLMConfig(model=model_name, temperature=0.0, max_tokens=200)

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


def make_get_file_node(
    llm: LLMService,
    *,
    model_name: str = "gemini-1.5-flash",
) -> Callable[[ConversationState], ConversationState]:
    """
    GET_FILE node.

    - If dataset.path already exists -> outcome DONE.
    - Else: if no user message yet -> NEEDS_INPUT(DATASET_PATH) with a welcome prompt.
    - Else: parse the latest user message into dataset.path (regex-first, LLM fallback).
      * If parse succeeds -> outcome DONE.
      * Else -> NEEDS_INPUT(DATASET_PATH).
    Router should send:
      GET_FILE (DONE) -> LOAD_DATASET
      GET_FILE (NEEDS_INPUT) -> PRESENT -> END
    """

    def get_file(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)

        conversation_id = control_in["conversation_id"]
        stage = control_in["stage"]  # router-owned

        def mk_control(
            *,
            status: Status,
            outcome: Outcome,
            need: Need,
            node_message: str,
            last_error: JSONDict | None,
        ) -> ControlState:
            return {
                "conversation_id": conversation_id,
                "status": status,
                "stage": stage,
                "outcome": outcome,
                "need": need,
                "interrupt_type": None,
                "last_error": last_error,
                "node_message": node_message,
            }

        # already have a path -> nothing to do
        existing_path = dataset_in.get("path")
        if isinstance(existing_path, str) and existing_path.strip():
            return {
                **state,
                "control": mk_control(
                    status="OK",
                    outcome="DONE",
                    need="NONE",
                    last_error=None,
                    node_message=f"Dataset path is set. Next: load `{existing_path}`.",
                ),
            }

        # need user input
        prior_msgs = cast(Sequence[BaseMessage], state.get("messages", []))
        user_text = _last_user_text(prior_msgs)
        if not user_text:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="NEEDS_INPUT",
                    need="DATASET_PATH",
                    last_error=None,
                    node_message=(
                        "Welcome. Paste the full path to your CSV file (ending in .csv). "
                        "Example: /path/to/data.csv"
                    ),
                ),
            }

        # parse: regex first
        parsed_path = _regex_extract_csv_path(user_text)

        llm_err: JSONDict | None = None
        if parsed_path is None:
            parsed_path, llm_err = _llm_extract_csv_path(
                llm,
                user_text=user_text,
                model_name=model_name,
            )

        if parsed_path is None:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="NEEDS_INPUT",
                    need="DATASET_PATH",
                    last_error=llm_err,
                    node_message=(
                        "I couldn’t detect a valid .csv path. "
                        "Please paste the full path (ending with .csv)."
                    ),
                ),
            }

        dataset_out: DatasetState = {
            **dataset_in,
            "path": parsed_path,
        }

        return {
            **state,
            "dataset": dataset_out,
            "control": mk_control(
                status="OK",
                outcome="DONE",
                need="NONE",
                last_error=llm_err,  # keep if LLM helped but had non-fatal issue; you can also drop it
                node_message=f"Got the dataset path. Next: load `{parsed_path}`.",
            ),
        }

    return get_file
