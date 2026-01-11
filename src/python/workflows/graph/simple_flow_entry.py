from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import  Callable, Mapping, Optional, Sequence
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
    new_control_state,
)
from python.workflows.state.dataset_state import DatasetState, empty_dataset_state
from python.workflows.state.metadata_state import (
    DraftDesign,
    MetadataState,
    empty_metadata_state,
)
from python.workflows.utils.types import JSONDict

log = logging.getLogger(__name__)

NodeFn = Callable[[ConversationState], ConversationState]


# =============================================================================
# Canonical state init + normalization
# =============================================================================
def new_conversation_state(conversation_id: UUID) -> ConversationState:
    """Canonical initializer for a brand-new conversation."""
    return {
        "control": new_control_state(conversation_id),
        "dataset": empty_dataset_state(),
        "metadata": empty_metadata_state(),
        "messages": [],
    }


def _as_mapping(obj: object) -> Mapping[str, object]:
    return obj if isinstance(obj, Mapping) else {} # pyright: ignore[reportUnknownVariableType]


def _as_message_list(obj: object) -> list[BaseMessage]:
    return list(obj) if isinstance(obj, list) else [] # pyright: ignore[reportUnknownArgumentType]


def _coerce_jsondict(obj: object) -> JSONDict | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)  # type: ignore[return-value]  # JSON-ish
    return None


def _coerce_str_list(obj: object) -> list[str]:
    if not isinstance(obj, list):
        return []
    out: list[str] = []
    for x in obj: # pyright: ignore[reportUnknownVariableType]
        if isinstance(x, str):
            out.append(x)
    return out


def _coerce_optional_str(obj: object) -> str | None:
    return obj if isinstance(obj, str) else None


def _normalize_loaded_state(*, loaded: Optional[ConversationState], conversation_id: UUID) -> ConversationState:
    """
    Ensure a shape-complete ConversationState:
      - repo may return None (new convo)
      - persisted states may be missing keys
      - enforce conversation_id as routing truth
    """
    seed = new_conversation_state(conversation_id)
    if loaded is None:
        return seed

    loaded_control = _as_mapping(loaded.get("control"))
    loaded_dataset = _as_mapping(loaded.get("dataset"))
    loaded_metadata = _as_mapping(loaded.get("metadata"))
    loaded_messages = _as_message_list(loaded.get("messages"))

    # -----------------------------
    # control
    # -----------------------------
    control: ControlState = new_control_state(conversation_id)
    stage = loaded_control.get("stage")
    if isinstance(stage, str):
        control["stage"] = stage  # type: ignore[assignment]  # Stage literal validated in runner
    status = loaded_control.get("status")
    if isinstance(status, str):
        control["status"] = status  # type: ignore[assignment]  # Status literal validated in runner
    post_action = loaded_control.get("post_action")
    if isinstance(post_action, str):
        control["post_action"] = post_action  # type: ignore[assignment]  # ACTION literal validated in runner

    if "post_failure_suggested_stage" in loaded_control:
        v = loaded_control.get("post_failure_suggested_stage")
        if v is None or isinstance(v, str):
            control["post_failure_suggested_stage"] = v  # type: ignore[assignment]

    if "last_error" in loaded_control:
        control["last_error"] = _coerce_jsondict(loaded_control.get("last_error"))

    if "node_message" in loaded_control:
        nm = loaded_control.get("node_message")
        if isinstance(nm, str):
            control["node_message"] = nm

    if "pending_stage" in loaded_control:
        ps = loaded_control.get("pending_stage")
        if ps is None or isinstance(ps, str):
            control["pending_stage"] = ps  # type: ignore[assignment]

    if "awaiting_user" in loaded_control:
        au = loaded_control.get("awaiting_user")
        if isinstance(au, bool):
            control["awaiting_user"] = au

    # routing truth
    control["conversation_id"] = conversation_id

    # -----------------------------
    # dataset (total=False)
    # -----------------------------
    dataset: DatasetState = empty_dataset_state()
    if "path" in loaded_dataset:
        dataset["path"] = _coerce_optional_str(loaded_dataset.get("path"))
    if "id" in loaded_dataset:
        v = loaded_dataset.get("id")
        dataset["id"] = v if isinstance(v, UUID) else None
    if "raw_schema" in loaded_dataset:
        dataset["raw_schema"] = _coerce_jsondict(loaded_dataset.get("raw_schema"))
    if "summary" in loaded_dataset:
        dataset["summary"] = _coerce_jsondict(loaded_dataset.get("summary"))
    if "load_error" in loaded_dataset:
        dataset["load_error"] = _coerce_optional_str(loaded_dataset.get("load_error"))
    if "get_file_last_user_msg_idx" in loaded_dataset:
        v = loaded_dataset.get("get_file_last_user_msg_idx")
        dataset["get_file_last_user_msg_idx"] = v if isinstance(v, int) else dataset.get("get_file_last_user_msg_idx")

    # -----------------------------
    # metadata (shape-complete)
    # -----------------------------
    metadata: MetadataState = empty_metadata_state()

    # proposed_design / final_design can be None; accept dict-ish payloads
    pd = loaded_metadata.get("proposed_design")
    if pd is None or isinstance(pd, Mapping):
        metadata["proposed_design"] = None if pd is None else (dict(pd) if isinstance(pd, Mapping) else None)  # type: ignore[assignment]

    fd = loaded_metadata.get("final_design")
    if fd is None or isinstance(fd, Mapping):
        metadata["final_design"] = None if fd is None else (dict(fd) if isinstance(fd, Mapping) else None)  # type: ignore[assignment]

    # draft exists; overlay only known keys
    loaded_draft = loaded_metadata.get("draft")
    if isinstance(loaded_draft, Mapping):
        draft: DraftDesign = metadata["draft"]
        t = loaded_draft.get("treatment") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if t is None or isinstance(t, str):
            draft["treatment"] = t
        o = loaded_draft.get("outcome") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if o is None or isinstance(o, str):
            draft["outcome"] = o

        cs = loaded_draft.get("covariate_strategy") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if cs is None or isinstance(cs, str):
            draft["covariate_strategy"] = cs  # type: ignore[assignment]

        covs = loaded_draft.get("covariates") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(covs, list):
            draft["covariates"] = _coerce_str_list(covs) # pyright: ignore[reportUnknownArgumentType]

        ems = loaded_draft.get("effect_modifiers") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(ems, list):
            draft["effect_modifiers"] = _coerce_str_list(ems) # pyright: ignore[reportUnknownArgumentType]

        cq = loaded_draft.get("causal_question") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if cq is None or isinstance(cq, str):
            draft["causal_question"] = cq

        acc = loaded_draft.get("accept") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(acc, bool):
            draft["accept"] = acc

        metadata["draft"] = draft

    # remaining metadata fields
    lui = loaded_metadata.get("last_user_msg_idx")
    if isinstance(lui, int):
        metadata["last_user_msg_idx"] = lui

    cm = loaded_metadata.get("canonical_metadata")
    if cm is None or isinstance(cm, Mapping):
        metadata["canonical_metadata"] = None if cm is None else dict(cm) # pyright: ignore[reportUnknownArgumentType]

    warnings = loaded_metadata.get("warnings")
    if isinstance(warnings, list):
        # Keep only mapping-like items as JSONDict
        w_out: list[JSONDict] = []
        for w in warnings: # pyright: ignore[reportUnknownVariableType]
            jw = _coerce_jsondict(w) # pyright: ignore[reportUnknownArgumentType]
            if jw is not None:
                w_out.append(jw)
        metadata["warnings"] = w_out

    if "validation_report" in loaded_metadata:
        vr = loaded_metadata.get("validation_report")
        metadata["validation_report"] = None if vr is None else (dict(vr) if isinstance(vr, Mapping) else None) # pyright: ignore[reportUnknownArgumentType]

    if "validation_passed" in loaded_metadata:
        vp = loaded_metadata.get("validation_passed")
        metadata["validation_passed"] = vp if isinstance(vp, bool) or vp is None else None

    # -----------------------------
    # messages
    # -----------------------------
    # We trust repo to store BaseMessage objects. If it stores raw dicts, you need a serializer.
    messages: list[BaseMessage] = loaded_messages

    return {
        "control": control,
        "dataset": dataset,
        "metadata": metadata,
        "messages": messages,
    }


# =============================================================================
# Control/message helpers
# =============================================================================
def _control(state: ConversationState) -> ControlState:
    return state["control"]


def _msgs(state: ConversationState) -> list[BaseMessage]:
    return state["messages"]


def _last_is_human(state: ConversationState) -> bool:
    m = _msgs(state)
    return bool(m) and isinstance(m[-1], HumanMessage)


def _node_message(c: ControlState) -> str:
    return str(c.get("node_message") or "").strip()


def _post_action(c: ControlState) -> ACTION:
    a = c.get("post_action")
    return a if isinstance(a, str) else "NONE"  # type: ignore[return-value]


def _status(c: ControlState) -> Status:
    s = c.get("status")
    return s if isinstance(s, str) else "PENDING"  # type: ignore[return-value]


def _has_output(c: ControlState) -> bool:
    """
    Output is ready when:
      - node_message has content, OR
      - node explicitly requests PRESENT / PRESENT_AND_USER_INPUT
    """
    a = _post_action(c)
    return bool(_node_message(c)) or a in {"PRESENT", "PRESENT_AND_USER_INPUT"}


def _log_error_if_any(c: ControlState) -> None:
    err = c.get("last_error")
    if err is None:
        return
    try:
        log.error(
            "workflow_error stage=%s conversation_id=%s error=%r",
            c.get("stage"),
            str(c.get("conversation_id")),
            err,
        )
    except Exception:
        log.error("workflow_error error_logging_failed")


def _ai_texts_since(messages: Sequence[BaseMessage], start_idx: int) -> list[str]:
    """
    Extract AI text emitted during this invoke only.
    Prevents re-sending older AI messages when the workflow returns without emitting.
    """
    out: list[str] = []
    for m in messages[start_idx:]:
        if isinstance(m, AIMessage):
            content: object = m.content # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            txt = content if isinstance(content, str) else str(content) # pyright: ignore[reportUnknownArgumentType]
            txt = txt.strip()
            if txt:
                out.append(txt)
            continue

        # fallback for message types exposing `.type == "ai"`
        if getattr(m, "type", None) == "ai":
            content2: object = getattr(m, "content", "")
            txt2 = content2 if isinstance(content2, str) else str(content2)
            txt2 = txt2.strip()
            if txt2:
                out.append(txt2)

    return out


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
# Single public class (repo + runner + response)
# =============================================================================
class SimpleWorkflow:
    """
    Single public boundary.

    Public API:
        invoke(user_id, conversation_id, user_text?) -> WorkflowResponse

    Behavior:
      - loads ConversationState from repo (or initializes)
      - appends HumanMessage if user_text provided
      - runs state machine until it must stop:
          * NEEDS_INPUT latch (no new human msg) OR
          * output becomes ready (node_message / PRESENT / PRESENT_AND_USER_INPUT) OR
          * no progress possible
      - when output is ready, it appends AIMessage to messages, clears node_message,
        advances stage/status/post_action, then returns
      - saves updated state to repo
      - returns ONLY outputs produced during this invoke (no re-sends)

    Reliability note (later):
      - You can add output sequence numbers / last_seen_seq for replay.
      - Not implemented here by request.
    """

    def __init__(
        self,
        *,
        repo: ConversationRepo,
        nodes: dict[Stage, NodeFn],
        cfg: WorkflowConfig,
    ) -> None:
        self._repo = repo
        self._nodes = nodes
        self._cfg = cfg

    def invoke(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_text: str | None = None,
    ) -> WorkflowResponse:
        loaded: ConversationState | None
        try:
            loaded = self._repo.load(user_id=user_id, conversation_id=conversation_id)
        except KeyError:
            loaded = None

        state = _normalize_loaded_state(loaded=loaded, conversation_id=conversation_id)

        if user_text is not None:
            _msgs(state).append(HumanMessage(content=user_text))

        before_len = len(_msgs(state))

        state2 = self._run_until_stop(state)

        self._repo.save(user_id=user_id, conversation_id=conversation_id, state=state2)

        outs = _ai_texts_since(_msgs(state2), before_len)
        text = "\n\n".join(outs).strip()

        c = _control(state2)
        needs_input = _post_action(c) == "NEEDS_INPUT"

        return WorkflowResponse(
            text=text,
            needs_input=needs_input,
            conversation_id=conversation_id,
        )

    # -------------------------------------------------------------------------
    # Internal runner
    # -------------------------------------------------------------------------
    def _run_until_stop(self, state: ConversationState) -> ConversationState:
        for _ in range(self._cfg.max_internal_steps):
            c = _control(state)
            stage = c["stage"]

            if stage not in self._cfg.valid_stages:
                raise ValueError(f"Invalid stage {stage!r}")

            # If the persisted state already contains "output ready", do not run nodes.
            if _has_output(c):
                _log_error_if_any(c)
                return self._emit_output_and_advance(state)

            # Waiting latch (NEEDS_INPUT)
            if _post_action(c) == "NEEDS_INPUT":
                if _last_is_human(state):
                    # Unlock and continue
                    c2: ControlState = {
                        **c,
                        "post_action": "NONE",
                        "status": "PENDING",
                    }
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
                return self._emit_output_and_advance(state2)

            st = _status(c2)

            if st == "ABORTED":
                suggested = c2.get("post_failure_suggested_stage")
                fallback = self._cfg.prev_stage.get(stage, stage)
                nxt = suggested if suggested is not None else fallback

                c3: ControlState = {
                    **c2,
                    "stage": nxt,
                    "status": "PENDING",
                    "pending_stage": None,
                }
                state = {**state2, "control": c3}
                continue

            if stage != "DONE" and st == "DONE":
                pending = c2.get("pending_stage")
                nxt = pending if pending is not None else self._cfg.next_stage.get(stage, stage)

                c3: ControlState = {
                    **c2,
                    "stage": nxt,
                    "status": "PENDING",
                    "pending_stage": None,
                }
                state = {**state2, "control": c3}
                continue

            return state2

        log.warning("SimpleWorkflow._run_until_stop hit max_internal_steps=%s", self._cfg.max_internal_steps)
        return state

    def _emit_output_and_advance(self, state: ConversationState) -> ConversationState:
        """
        Emits output (AIMessage) and advances control plane.

        - Payload comes from control.node_message; falls back to "(no message)".
        - Appends AIMessage(payload) to state["messages"].
        - Clears node_message to avoid re-emitting in subsequent calls.
        - Converts PRESENT_AND_USER_INPUT -> NEEDS_INPUT latch.

        NOTE: replay/reliability (seq numbers, outbox, last_seen) intentionally not implemented.
        """
        c = _control(state)
        stage = c["stage"]
        st = _status(c)
        action = _post_action(c)

        payload = _node_message(c) or "(no message)"
        out_messages = [*_msgs(state), AIMessage(content=payload)] # pyright: ignore[reportUnknownVariableType]

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
            "node_message": "",
            "pending_stage": None,
        }

        return {
            **state,
            "control": c2,
            "messages": out_messages,
        }
