from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, cast
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from python.domain.models.workflow_response import WorkflowResponse
from python.domain.repo.conversation_repo import ConversationRepo
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import (
    ACTION,
    ControlState,
    Stage,
    Status,
    control_log_line,
    new_control_state,
)
from python.workflows.state.dataset_state import DatasetState, empty_dataset_state
from python.workflows.state.metadata_state import DraftDesign, MetadataState, empty_metadata_state
from python.workflows.utils.types import JSONDict

log = logging.getLogger(__name__)

NodeFn = Callable[[ConversationState], ConversationState]
EnhancerFn = Callable[[ConversationState], str]

# Treat ONLY these as "placeholder defaults" eligible for enhancement replacement.
_PLACEHOLDER_DEFAULTS = {"", "(no message)"}


# =============================================================================
# Canonical state init + normalization
# =============================================================================
def new_conversation_state(conversation_id: UUID) -> ConversationState:
    return {
        "control": new_control_state(conversation_id),
        "dataset": empty_dataset_state(),
        "metadata": empty_metadata_state(),
        "messages": [],
    }


def _as_dict(obj: object) -> dict[str, object]:
    """Turn mapping-ish object into dict[str, object] to avoid 'unknown member type' on .get()."""
    if isinstance(obj, dict):
        return cast(dict[str, object], obj)
    if hasattr(obj, "items"):
        try:
            d = dict(obj)  # type: ignore[arg-type]
            return cast(dict[str, object], d)
        except Exception:
            return {}
    return {}


def _as_list(obj: object) -> list[object]:
    return cast(list[object], obj) if isinstance(obj, list) else []


def _coerce_jsondict(obj: object) -> JSONDict | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return cast(JSONDict, obj)
    if hasattr(obj, "items"):
        try:
            return cast(JSONDict, dict(obj))  # type: ignore[arg-type]
        except Exception:
            return None
    return None


def _coerce_optional_str(obj: object) -> str | None:
    return obj if isinstance(obj, str) else None


def _coerce_optional_uuid(obj: object) -> UUID | None:
    if isinstance(obj, UUID):
        return obj
    if isinstance(obj, str):
        try:
            return UUID(obj)
        except Exception:
            return None
    return None


def _coerce_str_list(obj: object) -> list[str]:
    items = _as_list(obj)
    out: list[str] = []
    for x in items:
        if isinstance(x, str):
            out.append(x)
    return out


def _coerce_messages(obj: object) -> list[BaseMessage]:
    """
    Repo should store BaseMessage objects.
    If it stores dict-like messages: {"type": "human"/"ai", "content": "..."},
    we coerce them.
    """
    items = _as_list(obj)
    out: list[BaseMessage] = []

    for item in items:
        if isinstance(item, BaseMessage):
            out.append(item)
            continue

        msg = _as_dict(item)
        if not msg:
            continue

        t = msg.get("type")
        content = msg.get("content", "")

        if t == "human":
            out.append(HumanMessage(content=str(content)))
        elif t == "ai":
            out.append(AIMessage(content=str(content)))

    return out


def _normalize_loaded_state(*, loaded: object | None, conversation_id: UUID) -> ConversationState:
    """
    Ensure a shape-complete ConversationState, even if repo returns None/partial/untyped dicts.
    Enforces conversation_id as routing truth.
    """
    seed = new_conversation_state(conversation_id)
    if loaded is None:
        return seed

    root = _as_dict(loaded)
    if not root:
        return seed

    loaded_control = _as_dict(root.get("control"))
    loaded_dataset = _as_dict(root.get("dataset"))
    loaded_metadata = _as_dict(root.get("metadata"))
    loaded_messages = _coerce_messages(root.get("messages"))

    # -----------------------------
    # control
    # -----------------------------
    control: ControlState = new_control_state(conversation_id)

    v = loaded_control.get("stage")
    if isinstance(v, str):
        control["stage"] = v  # type: ignore[assignment]

    v = loaded_control.get("status")
    if isinstance(v, str):
        control["status"] = v  # type: ignore[assignment]

    v = loaded_control.get("post_action")
    if isinstance(v, str):
        control["post_action"] = v  # type: ignore[assignment]

    v = loaded_control.get("post_failure_suggested_stage")
    if v is None or isinstance(v, str):
        control["post_failure_suggested_stage"] = v  # type: ignore[assignment]

    control["last_error"] = _coerce_jsondict(loaded_control.get("last_error"))

    v = loaded_control.get("node_message")
    if isinstance(v, str):
        control["node_message"] = v

    v = loaded_control.get("pending_stage")
    if v is None or isinstance(v, str):
        control["pending_stage"] = v  # type: ignore[assignment]

    v = loaded_control.get("awaiting_user")
    if isinstance(v, bool):
        control["awaiting_user"] = v

    control["conversation_id"] = conversation_id

    # -----------------------------
    # dataset
    # -----------------------------
    dataset: DatasetState = empty_dataset_state()

    if "path" in loaded_dataset:
        dataset["path"] = _coerce_optional_str(loaded_dataset.get("path"))

    if "id" in loaded_dataset:
        dataset["id"] = _coerce_optional_uuid(loaded_dataset.get("id"))

    if "raw_schema" in loaded_dataset:
        dataset["raw_schema"] = _coerce_jsondict(loaded_dataset.get("raw_schema"))

    if "summary" in loaded_dataset:
        dataset["summary"] = _coerce_jsondict(loaded_dataset.get("summary"))

    if "load_error" in loaded_dataset:
        dataset["load_error"] = _coerce_optional_str(loaded_dataset.get("load_error"))

    if "get_file_last_user_msg_idx" in loaded_dataset:
        v = loaded_dataset.get("get_file_last_user_msg_idx")
        if isinstance(v, int):
            dataset["get_file_last_user_msg_idx"] = v

    # -----------------------------
    # metadata
    # -----------------------------
    metadata: MetadataState = empty_metadata_state()

    pd = loaded_metadata.get("proposed_design")
    if pd is None:
        metadata["proposed_design"] = None
    else:
        d = _as_dict(pd)
        metadata["proposed_design"] = cast(object, d) if d else None  # type: ignore[assignment]

    fd = loaded_metadata.get("final_design")
    if fd is None:
        metadata["final_design"] = None
    else:
        d = _as_dict(fd)
        metadata["final_design"] = cast(object, d) if d else None  # type: ignore[assignment]

    ld = loaded_metadata.get("draft")
    draft_obj = _as_dict(ld)
    if draft_obj:
        draft: DraftDesign = metadata["draft"]

        t = draft_obj.get("treatment")
        if t is None or isinstance(t, str):
            draft["treatment"] = t

        o = draft_obj.get("outcome")
        if o is None or isinstance(o, str):
            draft["outcome"] = o

        cs = draft_obj.get("covariate_strategy")
        if cs is None or isinstance(cs, str):
            draft["covariate_strategy"] = cs  # type: ignore[assignment]

        draft["covariates"] = _coerce_str_list(draft_obj.get("covariates"))
        draft["effect_modifiers"] = _coerce_str_list(draft_obj.get("effect_modifiers"))

        cq = draft_obj.get("causal_question")
        if cq is None or isinstance(cq, str):
            draft["causal_question"] = cq

        acc = draft_obj.get("accept")
        if isinstance(acc, bool):
            draft["accept"] = acc

        metadata["draft"] = draft

    v = loaded_metadata.get("last_user_msg_idx")
    if isinstance(v, int):
        metadata["last_user_msg_idx"] = v

    cm = loaded_metadata.get("canonical_metadata")
    if cm is None:
        metadata["canonical_metadata"] = None
    else:
        d = _as_dict(cm)
        metadata["canonical_metadata"] = d if d else None  # type: ignore[assignment]

    warnings = loaded_metadata.get("warnings")
    w_items = _as_list(warnings)
    w_out: list[JSONDict] = []
    for w in w_items:
        jw = _coerce_jsondict(w)
        if jw is not None:
            w_out.append(jw)
    metadata["warnings"] = w_out

    vr = loaded_metadata.get("validation_report")
    if vr is None:
        metadata["validation_report"] = None
    else:
        d = _as_dict(vr)
        metadata["validation_report"] = d if d else None  # type: ignore[assignment]

    vp = loaded_metadata.get("validation_passed")
    if isinstance(vp, bool) or vp is None:
        metadata["validation_passed"] = vp  # type: ignore[assignment]

    return {
        "control": control,
        "dataset": dataset,
        "metadata": metadata,
        "messages": loaded_messages,
    }


# =============================================================================
# Control/message helpers
# =============================================================================
def _control(state: ConversationState) -> ControlState:
    return state["control"]


def _msgs(state: ConversationState) -> list[BaseMessage]:
    return state["messages"]


def _last_is_human(state: ConversationState) -> bool:
    ms = _msgs(state)
    return bool(ms) and isinstance(ms[-1], HumanMessage)


def _node_message(c: ControlState) -> str:
    return str(c.get("node_message") or "").strip()


def _post_action(c: ControlState) -> ACTION:
    a = c.get("post_action")
    return a if isinstance(a, str) else "NONE"  # type: ignore[return-value]


def _status(c: ControlState) -> Status:
    s = c.get("status")
    return s if isinstance(s, str) else "PENDING"  # type: ignore[return-value]


def _has_output(c: ControlState) -> bool:
    return _post_action(c) in {"PRESENT", "PRESENT_AND_USER_INPUT"}

def _last_ai_text(state: ConversationState) -> str:
    msgs = _msgs(state)
    if not msgs:
        return ""
    last = msgs[-1]
    if not isinstance(last, AIMessage):
        return ""
    content = last.content # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    return content if isinstance(content, str) else str(content) # pyright: ignore[reportUnknownArgumentType]

def _log_state(tag: str, state: ConversationState) -> None:
    log.info("%s %s", tag, control_log_line(_control(state)))


def _log_error_if_any(c: ControlState) -> None:
    err = c.get("last_error")
    if err is None:
        return
    log.error(
        "workflow_error stage=%s conversation_id=%s error=%r",
        c.get("stage"),
        str(c.get("conversation_id")),
        err,
    )


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class WorkflowConfig:
    next_stage: Mapping[Stage, Stage]
    prev_stage: Mapping[Stage, Stage]
    valid_stages: set[Stage]
    max_internal_steps: int = 32


# =============================================================================
# Workflow
# =============================================================================
class SimpleWorkflow:
    """
    Semantics you requested:

    - ONE invoke runs internal steps until the FIRST time some node requests PRESENT
      (via node_message or post_action PRESENT/PRESENT_AND_USER_INPUT).
    - Then we emit exactly ONE AIMessage and RETURN (adapter can call invoke again).

    - NEEDS_INPUT is a latch: we return without emitting.
    """

    def __init__(
        self,
        *,
        repo: ConversationRepo,
        nodes: dict[Stage, NodeFn],
        cfg: WorkflowConfig,
        enhance: EnhancerFn
    ) -> None:
        self._repo = repo
        self._nodes = nodes
        self._cfg = cfg
        self._enhance = enhance

    def invoke(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_text: str | None = None,
    ) -> WorkflowResponse:
        try:
            loaded = self._repo.load(user_id=user_id, conversation_id=conversation_id)
        except KeyError:
            loaded = None

        # TODO: check it later
        state = _normalize_loaded_state(loaded=loaded, conversation_id=conversation_id)

        if user_text is not None:
            _msgs(state).append(HumanMessage(content=user_text))

        state2 = self._run_until_stop(state)

        self._repo.save(user_id=user_id, conversation_id=conversation_id, state=state2)

        output_txt = _last_ai_text(state2)
        log.info("%s %s", "OUTPUT_TXT_LOG",output_txt)

        c = _control(state2)
        needs_input = _post_action(c) == "NEEDS_INPUT"

        _log_state("invoke:saved and returned", state2)
        return WorkflowResponse(text=output_txt, needs_input=needs_input, conversation_id=conversation_id)
    # -------------------------------------------------------------------------
    # Internal runner
    # -------------------------------------------------------------------------
    def _run_until_stop(self, state: ConversationState) -> ConversationState:
        for step in range(self._cfg.max_internal_steps): # pyright: ignore[reportUnusedVariable]
            c = _control(state)
            stage = c["stage"]


            if stage not in self._cfg.valid_stages:
                raise ValueError(f"Invalid stage {stage!r}")

            if stage == "DONE":
                return state

            if _has_output(c):
                _log_error_if_any(c)
                state_out = self._emit_output_and_advance(state)
                return state_out

            if _post_action(c) == "NEEDS_INPUT":
                if _last_is_human(state):
                    c2: ControlState = {**c, "post_action": "NONE", "status": "PENDING"}
                    state = {**state, "control": c2}
                    continue

                return state

            node = self._nodes.get(stage)
            if node is None:
                raise KeyError(f"No node registered for stage {stage!r}")

            state2 = node(state)

            c2 = _control(state2)
            
            if _has_output(c2):
                _log_error_if_any(c2)
                state_out = self._emit_output_and_advance(state2)
                return state_out

            st2 = _status(c2)

            if st2 == "ABORTED":
                suggested = c2.get("post_failure_suggested_stage")
                fallback = self._cfg.prev_stage.get(stage, stage)
                nxt = suggested if suggested is not None else fallback

                c3: ControlState = {**c2, "stage": nxt, "status": "PENDING", "pending_stage": None}
                state = {**state2, "control": c3}
                continue

            if st2 == "DONE":
                pending = c2.get("pending_stage")
                nxt = pending if pending is not None else self._cfg.next_stage.get(stage, stage)

                c3 = {**c2, "stage": nxt, "status": "PENDING", "pending_stage": None}
                state = {**state2, "control": cast(ControlState, c3)}
                continue
            
            return state2

        log.warning("SimpleWorkflow._run_until_stop hit max_internal_steps=%s", self._cfg.max_internal_steps)
        return state

    def _emit_output_and_advance(self, state: ConversationState) -> ConversationState:
        """
        Emits exactly ONE AIMessage, clears node_message, advances control, then returns.

        Enhancement rule (matches your requirement):
          - If base payload is a REAL message (non-placeholder), DO NOT replace it.
          - If base payload is a placeholder ("", "(no message)"), then try enhancer and replace.
          - If enhancer empty/fails, fall back to base, and if base empty -> "(no message)".
        """
        c = _control(state)
        stage = c["stage"]
        st = _status(c)
        action = _post_action(c)

        node_message = _node_message(c)
        if not node_message and action in {"PRESENT", "PRESENT_AND_USER_INPUT"}:
            raise ValueError("ControlState requests PRESENT but node_message is empty.")

        try:
            node_message = self._enhance(state) if self._enhance is not None else None
        except Exception:
                log.exception("workflow_output_enhancer_failed")

        msg = AIMessage(content=node_message)

        out_messages = [*_msgs(state), msg]  # pyright: ignore[reportUnknownVariableType]

        # Advance control plane AFTER emission
        next_stage: Stage = stage
        next_status: Status = st

        pending = c.get("pending_stage")
        suggested = c.get("post_failure_suggested_stage")

        if st == "ABORTED":
            next_stage = suggested if suggested is not None else self._cfg.prev_stage.get(stage, stage)
            next_status = "PENDING"
        elif pending is not None:
            next_stage = pending
            next_status = "PENDING"
        elif st == "DONE" and stage != "DONE":
            next_stage = self._cfg.next_stage.get(stage, stage)
            next_status = "PENDING"

        wants_user = action == "PRESENT_AND_USER_INPUT"
        next_action: ACTION = "NEEDS_INPUT" if wants_user else "NONE"

        c2: ControlState = {
            **c,
            "stage": next_stage,
            "status": next_status,
            "post_action": next_action,
            "node_message": "",  # critical: clear to prevent re-emission
            "pending_stage": None,
        }

        return {**state, "control": c2, "messages": out_messages}
