from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_deps import ConfirmTransformedProtocolDeps
from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_state import (
    ConfirmTransformedProtocolPayloadModel,
    ConfirmTransformedProtocolState,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import TransformProtocolState
from python.implementation.workflows.utils.utils import json_sanitize
from python.implementation.workflows.utils.validation import ValidationIssueModel

from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_prompts import (
    confirm_transformed_protocol_node_fail_prompt,
    confirm_decision_system_prompt,
    confirm_decision_user_prompt_template,
    confirm_discussion_system_prompt,
    confirm_discussion_user_prompt_template,
    confirm_transformed_protocol_node_info,
)


class _DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    user_accepted: bool
    user_message: str = Field(..., min_length=1)
    error_message: Optional[str] = None
    improvement_instructions: Optional[str] = None
    

@dataclass(frozen=True)
class ConfirmTransformedProtocolNode(Node):
    NAME: ClassVar[str] = ConfirmTransformedProtocolState.NAME

    llm: LLMService
    model_name: str

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return confirm_transformed_protocol_node_info()


    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, Any],
        user_message: Optional[str],
        router_message: Optional[str],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        deps = ConfirmTransformedProtocolDeps.from_loaded(previous_state_dependencies)
        tp = deps.transform_protocol.payload
        
        clean_dataset_validation_issues = tp.cleaned_dataset_validation_issues
        transform_validation_issues = tp.transformation_issues
        
        has_fail = any(i.severity == "FAIL" for i in clean_dataset_validation_issues) or any(i.severity == "FAIL" for i in transform_validation_issues)
        if has_fail:
            issues_json = json_sanitize([i.model_dump(mode="json") for i in clean_dataset_validation_issues + transform_validation_issues])
            user_prompt= "\n\nValidation issues:\n" + issues_json
            system_prompt = confirm_transformed_protocol_node_fail_prompt()
            
            msg = _call_llm_for_discussion(self.llm, self.model_name, None, system_prompt, user_prompt)
            return ConfirmTransformedProtocolState(
                ConfirmTransformedProtocolPayloadModel(
                    user_accepted=None,
                    user_message=msg,
                    error_message=msg,
                )
            )

        # ---- If no WARN issues: accept immediately (policy) ----
        # If you want explicit user confirmation always, remove this block.
        if not has_warn:
            return ConfirmTransformedProtocolState(
                ConfirmTransformedProtocolPayloadModel(
                    user_accepted=True,
                    user_message="No warnings found. Proceeding.",
                    error_message=None,
                )
            )

        # ---- If user has not replied yet: generate discussion message (LLM #1) ----
        if user_message is None or not user_message.strip():
            cfg = LLMConfig(model=self.model_name, temperature=0.2, top_p=0.95, max_tokens=1600)

            prompt = confirm_discussion_user_prompt_template().format(
                PROTOCOL_JSON=protocol_json,
                ROLES_JSON=roles_json,
                ISSUES_JSON=issues_json,
            )

            resp = self.llm.generate(
                system_prompt=confirm_discussion_system_prompt(),
                user_prompt=prompt,
                config=cfg,
                history=messages_history,
            )

            return ConfirmTransformedProtocolState(
                ConfirmTransformedProtocolPayloadModel(
                    user_accepted=None,
                    user_message=resp.content.strip(),
                    error_message=None,
                )
            )

        # ---- User replied: classify accept/reject (LLM #2 strict JSON) ----
        cfg2 = LLMConfig(model=self.model_name, temperature=0.0, top_p=1.0, max_tokens=600)

        decision_prompt = confirm_decision_user_prompt_template().format(
            ISSUES_JSON=issues_json,
            ASSISTANT_DISCUSSION="",  # you can store last assistant discussion in state later if you want
            USER_REPLY=user_message.strip(),
        )

        decision = self.llm.generate_json(
            schema=_DecisionModel,
            system_prompt=confirm_decision_system_prompt(),
            user_prompt=decision_prompt,
            config=cfg2,
            history=messages_history,
            max_attempts=3,
        )

        if decision.user_accepted:
            return ConfirmTransformedProtocolState(
                ConfirmTransformedProtocolPayloadModel(
                    user_accepted=True,
                    user_message=decision.user_message,
                    error_message=None,
                )
            )

        # Reject: either user disagreed or was ambiguous (we treat ambiguous as reject + clarify in message)
        msg = decision.user_message
        if decision.improvement_instructions:
            msg = f"{msg}\n\nRequested changes:\n{decision.improvement_instructions}"

        return ConfirmTransformedProtocolState(
            ConfirmTransformedProtocolPayloadModel(
                user_accepted=False,
                user_message=msg,
                error_message=decision.error_message or "User rejected due to validation warnings.",
            )
        )


def _call_llm_for_discussion(llm: LLMService, model_name: str, messages_history: Optional[Sequence[ChatMessage]], system_prompt: str, user_prompt: str) -> str:
    cfg = LLMConfig(model=model_name, temperature=0.7)
    resp = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=cfg,
        history=messages_history,
    )
    return resp.content.strip()        