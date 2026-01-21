from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import  Final, Mapping, Sequence, Tuple, cast
from uuid import UUID
from langchain_core.messages import BaseMessage
from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.control_state import ControlState, Stage, Status


_JSON_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_NEXT_STAGE: Final[Mapping[Stage, Stage]] = {
    "LOAD_DATASET": "PROPOSE_AND_CONFIRM_METADATA",
    "PROPOSE_AND_CONFIRM_METADATA": "COMPILE_PROTOCOL",
    "COMPILE_PROTOCOL": "DONE",
}

_STAGE_DOC: Final[Mapping[Stage, str]] = {
    "LOAD_DATASET": "Load CSV from dataset.path. Writes dataset.summary/raw_schema (and maybe dataset.id).",
    "PROPOSE_AND_CONFIRM_METADATA": "Propose+confirm metadata: treatment/outcome/controls/covariates/etc.",
    "COMPILE_PROTOCOL": "Compile protocol state.",
    "DONE": "Workflow complete.",
}


def _noop_node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
    return state



def _parse_json_object_strict(text: str) -> dict: # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty LLM response")

    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root must be an object")
    return obj # pyright: ignore[reportUnknownVariableType]


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "human":
            return str(getattr(m, "content", "") or "").strip()
        name = m.__class__.__name__.lower()
        if "human" in name or "user" in name:
            return str(getattr(m, "content", "") or "").strip()
    return ""


def _last_ai_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "ai":
            return str(getattr(m, "content", "") or "").strip()
        name = m.__class__.__name__.lower()
        if "ai" in name or "assistant" in name:
            return str(getattr(m, "content", "") or "").strip()
    return ""


@dataclass(frozen=True)
class WorkflowRouter:
    llm: LLMService
    model_name: str
    nodes: Mapping[Stage, CallableNodeFunc]

    def route(
        self,
        state: ConversationState,
    ) -> Tuple[CallableNodeFunc, ConversationState]:
        control = state["control"]
        stage: Stage = control["current_stage"]
        status: Status = control["current_stage_status"]
        
        logging.warning(f"WorkflowRouter.route: stage={stage!r} status={status!r}")
        
        if status == "PENDING":
            return self._node_for(stage, self.nodes), state # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        if status == "DONE":
            next_stage = _NEXT_STAGE.get(stage, "DONE")
            next_state = self._advance(state, next_stage)
            return self._node_for(next_stage, self.nodes), next_state # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        if status == "ABORTED":
            recovered_stage = self._llm_choose_recovery_stage(state)
            next_state = self._advance(state, recovered_stage)
            return self._node_for(recovered_stage, self.nodes), next_state # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        raise ValueError(f"Unknown control.current_stage_status: {status!r}")

    def _node_for(self, stage: Stage, nodes: Mapping[Stage, CallableNodeFunc]) -> CallableNodeFunc:
        if stage == "DONE":
            return _noop_node
        fn = nodes.get(stage)
        if fn is None:
            raise ValueError(f"No node registered for stage={stage!r}")
        return fn

    def _advance(self, state: ConversationState, next_stage: Stage) -> ConversationState:
        control = state["control"]
        next_status: Status = "DONE" if next_stage == "DONE" else "PENDING"
        new_control: ControlState = {
            **control,
            "current_stage": next_stage,
            "current_stage_status": next_status,
            "action_required": "NONE",
            "node_message": None,
        }
        return {**state, "control": new_control}

    def _llm_choose_recovery_stage(self, state: ConversationState) -> Stage:
        control = state["control"]
        dataset = state.get("dataset", {})
        metadata = state.get("metadata", {})
        protocol = state.get("protocol", {})
        messages = cast(Sequence[BaseMessage], state.get("messages", []))

        snapshot = { # pyright: ignore[reportUnknownVariableType]
            "control": control,
            "dataset": dataset,
            "metadata": metadata,
            "protocol": protocol,
            "last_user_message": _last_human_text(messages),
            "last_assistant_message": _last_ai_text(messages),
            "stages": dict(_STAGE_DOC),
            "instructions": (
                "Pick the earliest stage that can safely recover.\n"
                "- Missing/invalid dataset path => GET_FILE\n"
                "- dataset.path present but schema/summary missing => LOAD_DATASET\n"
                "- dataset loaded but metadata incomplete/not accepted => PROPOSE_AND_CONFIRM_METADATA\n"
                "- everything done => DONE"
            ),
        }

        system = (
            "You are a workflow recovery router.\n"
            "Return ONLY one JSON object with EXACTLY keys:\n"
            '{ "next_stage": string, "why": string }\n'
            f"- next_stage MUST be one of: {list(_STAGE_DOC.keys())}\n"
            "- No markdown. No extra keys."
        )
        
        config = LLMConfig(
            model=self.model_name,
            temperature=0.0,
        )
        
        resp = self.llm.generate(
            config=config, 
            system_prompt=system, 
            user_prompt=json.dumps(snapshot, ensure_ascii=False, default=str), # Added default=str
            history=None
          )
        obj = _parse_json_object_strict(cast(object, resp).content)  # type: ignore[attr-defined]
        if set(obj.keys()) != {"next_stage", "why"}: # pyright: ignore[reportUnknownArgumentType]
            raise ValueError("Router LLM must return exactly: {next_stage, why}")

        ns = obj.get("next_stage") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(ns, str):
            raise ValueError("Router LLM returned non-string next_stage")

        if ns not in _STAGE_DOC:
            raise ValueError(f"Router LLM returned invalid stage: {ns!r}")

        return  ns
