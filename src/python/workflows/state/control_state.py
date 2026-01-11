from __future__ import annotations

from typing import Literal, NotRequired, TypedDict
from uuid import UUID

from python.workflows.utils.types import JSONDict

Stage = Literal[
    "GET_FILE",
    "LOAD_DATASET",
    "PROPOSE_METADATA",
    "CONFIRM_METADATA",
    "DONE",
]

Status = Literal[
    "PENDING",
    "DONE",
    "RETRYABLE_ERROR",
    "ABORTED",
]

ACTION = Literal[
    "NONE",
    "PRESENT",
    "NEEDS_INPUT",
    "PRESENT_AND_USER_INPUT",
]

NEED_STAGE = Stage


class ControlState(TypedDict):
    """
    Control plane for orchestration.

    Rules of thumb:
    - Nodes write node_message + post_action.
    - The runner/PRESENT layer flushes node_message to messages and then clears it.
    - awaiting_user is a latch set only when we PRESENT_AND_USER_INPUT.
    """
    conversation_id: UUID
    stage: Stage
    status: Status
    post_action: ACTION

    post_failure_suggested_stage: NEED_STAGE | None
    last_error: JSONDict | None
    node_message: str

    pending_stage: NotRequired[Stage | None]
    awaiting_user: NotRequired[bool]


def new_control_state(conversation_id: UUID) -> ControlState:
    """
    Single canonical initializer so every entry path produces a type-correct ControlState.
    """
    return {
        "conversation_id": conversation_id,
        "stage": "GET_FILE",
        "status": "PENDING",
        "post_action": "NONE",
        "post_failure_suggested_stage": None,
        "last_error": None,
        "node_message": "",
        "pending_stage": None,
        "awaiting_user": False,
    }


def control_log_line(c: ControlState) -> str:
    """
    Stable one-liner for logs. Avoids dumping huge last_error/node_message.

    Example:
      control{cid=..., stage=GET_FILE, status=PENDING, action=NONE,
              pending=None, suggested=None, awaiting=False, node_len=0, err=none}
    """
    cid = str(c.get("conversation_id"))
    stage = str(c.get("stage"))
    status = str(c.get("status"))
    action = str(c.get("post_action"))

    pending = c.get("pending_stage", None)
    suggested = c.get("post_failure_suggested_stage", None)
    awaiting = bool(c.get("awaiting_user", False))

    node_msg = c.get("node_message", "")
    node_len = len(node_msg) if isinstance(node_msg, str) else 0 # pyright: ignore[reportUnnecessaryIsInstance]

    err = c.get("last_error")
    if isinstance(err, dict):
        code = err.get("code")
        err_tag = f"code={code}" if isinstance(code, str) and code else "dict"
    elif err is None:
        err_tag = "none"
    else:
        err_tag = type(err).__name__

    return (
        "control{"
        f"cid={cid}, "
        f"stage={stage}, "
        f"status={status}, "
        f"action={action}, "
        f"pending={pending}, "
        f"suggested={suggested}, "
        f"awaiting={awaiting}, "
        f"node_len={node_len}, "
        f"err={err_tag}"
        "}"
    )
