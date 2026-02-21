from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import  Final, Literal, Mapping, Tuple, cast
from uuid import UUID
from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers

NODE = Literal[
    "LOAD_DATASET",
    "PROTOCOL_DISCUSSION",
    "COMPILE_PROTOCOL",
    "INFERENCE_READY",
    "VALIDATE_INFERENCE_READY",
    "VALIDATE_INFERENCE_READY_DISCUSSION",
    "MODEL_SELECTION",
    "MODEL_SELECTION_DISCUSSION",
    "MODEL_PARAMS_FIT_DISCUSSION",
    "MODEL_FIT",
    "DONE",
]

_NEXT_NODE: Final[Mapping[NODE, NODE]] = {
    "LOAD_DATASET": "PROTOCOL_DISCUSSION",
    "PROTOCOL_DISCUSSION": "COMPILE_PROTOCOL",
    "COMPILE_PROTOCOL": "INFERENCE_READY",
    "INFERENCE_READY": "VALIDATE_INFERENCE_READY",
    "VALIDATE_INFERENCE_READY": "VALIDATE_INFERENCE_READY_DISCUSSION",
    "VALIDATE_INFERENCE_READY_DISCUSSION": "MODEL_SELECTION",
    "MODEL_SELECTION": "MODEL_SELECTION_DISCUSSION",
    "MODEL_SELECTION_DISCUSSION": "MODEL_PARAMS_FIT_DISCUSSION",
    "MODEL_PARAMS_FIT_DISCUSSION": "MODEL_FIT",
    "MODEL_FIT": "DONE",
}



_JSON_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)




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
            next_stage = CONTROL_STATE_NEXT_STAGE.get(stage, "DONE")
            next_state = self._advance(state, next_stage)
            return self._node_for(next_stage, self.nodes), next_state # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        if status == "ABORTED":
            for _ in range(3):
                try:
                    recovered_stage, router_message = self._llm_choose_recovery_stage(state)
                    if recovered_stage in CONTROL_STATE_STAGE_DOC:
                         break
                except Exception:
                    continue
            else:
                raise ValueError("Failed to recover from ABORTED status")
            
            new_control: ControlState = {
            **control,
            "current_stage": recovered_stage,
            "current_stage_status": "PENDING",
            "action_required": "NONE",
            "node_message": None,
            }
            
            next_state = cast(ConversationState, {**state, "control": new_control})
            next_state["router_message"] = router_message
                        
            return self._node_for(recovered_stage, self.nodes), next_state 

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

    def _llm_choose_recovery_stage(self, state: ConversationState) -> tuple[Stage, str | None]:
        state_str= json.dumps(state, default=str, ensure_ascii=False, indent=2) 
        history = ConversationStateHelpers.to_chat_history_last_k(state= state, k=10, drop_last_user=True);

        snapshot = { # pyright: ignore[reportUnknownVariableType]
            "state": state_str,
            "stages": dict(CONTROL_STATE_STAGE_DOC),
            "instructions": (
                "Pick the earliest stage that can safely recover.\n"
                "Add a router message for that stage if it would help the node recover better. Router messages are only seen by the node, not the user.\n"
                "OUTPUT MUST be a JSON object with EXACTLY keys: { next_stage: string, router_message: string|null }\n"
            ),
        }

        system = (
            "You are a workflow recovery router.\n"
            "Return ONLY one JSON object with EXACTLY keys:\n"
            '{ "next_stage": string, "router_message": string|null }\n'
            f"- next_stage MUST be one of: {list(CONTROL_STATE_STAGE_DOC.keys())}\n"
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
            history=history
          )
        obj = _parse_json_object_strict(cast(object, resp).content)  # type: ignore[attr-defined]
        if set(obj.keys()) != {"next_stage", "router_message"}: # pyright: ignore[reportUnknownArgumentType]
            raise ValueError("Router LLM must return exactly: {next_stage, router_message}")

        ns = obj.get("next_stage") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(ns, str):
            raise ValueError("Router LLM returned non-string next_stage")
        
        router_message = obj.get("router_message") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if router_message is not None and not isinstance(router_message, str):
            raise ValueError("Router LLM returned non-string router_message")

        if ns not in CONTROL_STATE_STAGE_DOC:
            raise ValueError(f"Router LLM returned invalid stage: {ns!r}")
        
        return cast(Stage, ns), router_message

