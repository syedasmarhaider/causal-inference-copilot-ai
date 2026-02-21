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

from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_prompts import (
    confirm_transformed_protocol_node_fail_prompt,
    confirm_decision_system_prompt,
    confirm_decision_user_prompt_template,
    confirm_discussion_system_prompt,
    confirm_discussion_user_prompt_template,
    confirm_transformed_protocol_node_info,
    discuss_decision_user_prompt_template,
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
         
        has_warn = any(i.severity == "WARN" for i in clean_dataset_validation_issues) or any(i.severity == "WARN" for i in transform_validation_issues)
        if has_warn:
            issues_json = json_sanitize([i.model_dump(mode="json") for i in clean_dataset_validation_issues + transform_validation_issues])
            system_prompt = confirm_decision_system_prompt()
            user_prompt = discuss_decision_user_prompt_template().format(issues=issues_json)
            message_for_user = _call_llm_for_discussion(self.llm, self.model_name, messages_history, system_prompt, user_prompt)
            
            self.llm.generate_json(
                system_prompt=confirm_decision_system_prompt(),
                user_prompt=confirm_decision_user_prompt_template().format(issues=issues_json),
                config=LLMConfig(model=self.model_name, temperature=0.7),
                history=messages_history[-10:] if messages_history else None,
                response_model=_DecisionModel,
            
            
        
        
        
        else:
            return ConfirmTransformedProtocolState(
                ConfirmTransformedProtocolPayloadModel(
                    user_accepted=True,
                    user_message="No validation issues found. Proceeding.",
                    error_message=None,
                )
            )      


def _call_llm_for_discussion(llm: LLMService, model_name: str, messages_history: Optional[Sequence[ChatMessage]], system_prompt: str, user_prompt: str) -> str:
    last_10_messages = messages_history[-10:] if messages_history else None
    cfg = LLMConfig(model=model_name, temperature=0.7)
    resp = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=cfg,
        history=last_10_messages,
    )
    return resp.content.strip()        