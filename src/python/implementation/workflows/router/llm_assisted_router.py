from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from python.domain.repo.analytics_repo import AnalyticsRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import NextDecision, Router
from python.domain.workflows.state import State
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferenceState,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_node import (
    CompileAndValidateNode,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidateState,
)
from python.implementation.workflows.nodes.dataset.dataset_node import DatasetNode
from python.implementation.workflows.nodes.dataset.dataset_prompts import (
    prev_state_revert_message,
)
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_selection.model_selection_node import (
    ModelSelectionNode,
)
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.noop_done.noop_done_node import NoopDoneNode
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    ProtocolDiscussionNode,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.router.llm_assisted_router_prompts import (
    ABORTED_ROUTER_SYSTEM_PROMPT,
    PENDING_ROUTER_SYSTEM_PROMPT,
)
from python.implementation.workflows.tools.tools_factory import DefaultToolFactory

log = get_logger(__name__, component="LLMAssistedRouter", log_type="workflow_router")
_DATASET_PENDING_CLARIFICATION_MESSAGE = (
    "I’m still helping with the current dataset. What exactly do you want me to inspect, "
    "filter, summarize, or change in the data? If you want to start causal setup instead, "
    "tell me the treatment, outcome, study type, and time zero."
)


class _RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    state_name: str | None = None
    router_confirmation_message_for_user: str | None = None


@dataclass(frozen=True)
class OtherStateInfo:
    name: str
    purpose: str
    should_persists_by_workflow: bool


class LLMAssistedRouterRouter(Router):
    def __init__(
        self,
        *,
        llm: LLMService,
    ) -> None:
        self._llm = llm
        self._next_state_names_map: Mapping[str, str | None] = init_next_state_names()
        self._node_name_to_description_map: Mapping[str, str] = get_node_name_with_description()
        self._callable_map: Mapping[str, Sequence[OtherStateInfo]] = (
            states_can_call_other_states_during_execution_map()
        )
        self._recoverable_map: Mapping[str, Sequence[str]] = recoverable_states_map()

    def get_initial_state_name(self) -> str:
        return DatasetState.NAME

    def get_done_state_name(self) -> str:
        return NoopDoneState.NAME

    def get_next_state_names(
        self,
        current_state_name: str,
    ) -> Sequence[str]:
        next_states: list[str] = []
        visited: set[str] = set()
        cursor = current_state_name
        while True:
            if cursor in visited:
                break
            visited.add(cursor)
            nxt = self._next_state_names_map.get(cursor)
            if nxt is None:
                break
            next_states.append(nxt)
            cursor = nxt
        return next_states

    def decide_next(
        self,
        *,
        current_state: State | None,
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        if current_state is None:
            return NextDecision(state_name=DatasetState.NAME,should_persists_by_workflow=True)

        current_name = _state_name(current_state)
        status = _state_status(current_state)
        recent_messages = _last_two_messages(messages_history)
        latest_user_message = _latest_user_message(messages_history)

        # TODO: replace this temporary router-level shortcut once dataset/dashboard revert flows
        # become fully explicit and no longer rely on the magic revert_data_changes message.
        if latest_user_message == prev_state_revert_message and current_name in {
            DatasetState.NAME,
            ProtocolDiscussionState.NAME,
        }:
            return NextDecision(
                state_name=DatasetState.NAME,
                should_persists_by_workflow=False)

        if status in ("DONE", "FREEZED"):
            next_name = self._next_state_names_map.get(current_name)
            if next_name is None:
                return NextDecision(state_name=current_name, should_persists_by_workflow=True)
            return NextDecision(state_name=next_name, should_persists_by_workflow=True)

        if status == "PENDING":
            return self._decide_pending(
                current_state_name=current_name,
                recent_messages=recent_messages,
            )

        if status == "ABORTED":
            return self._decide_aborted(
                current_state=current_state,
                recent_messages=recent_messages,
            )

        raise ValueError(f"Unexpected state status {status!r} for {current_name!r}")

    def _decide_pending(
        self,
        *,
        current_state_name: str,
        recent_messages: Sequence[ChatMessage],
    ) -> NextDecision:
        candidates = _pending_candidates(
            current_state_name=current_state_name, callable_map=self._callable_map
        )

        current_state_context = _build_current_state_context(
            current_state_name=current_state_name,
            node_descriptions=self._node_name_to_description_map,
        )
        fellow_state_context = _build_pending_fellow_state_context(
            current_state_name=current_state_name,
            callable_map=self._callable_map,
            node_descriptions=self._node_name_to_description_map,
        )

        try:
            decision = self._llm.generate_json(
                schema=_RouteDecision,
                system_prompt=PENDING_ROUTER_SYSTEM_PROMPT,
                user_prompt=build_pending_router_user_prompt(
                    current_state_context=current_state_context,
                    fellow_state_context=fellow_state_context,
                    recent_messages=recent_messages,
                ),
                config=LLMConfig(model="mini", temperature=0.4),
                history=None,
                max_attempts=2,
            )
        except Exception as exc:
            log.exception(
                "pending router llm decision failed",
                current_state_name=current_state_name,
                error=str(exc),
            )
            return NextDecision(
                state_name=None,
                should_persists_by_workflow=False,
                router_confirmation_message_for_user=_pending_clarification_message(
                    current_state_name=current_state_name,
                    fallback_message=(
                        "Sorry I couldn't understand the intended route. Please clarify what you want to do next."
                    ),
                ),
            )

        if decision.state_name is None:
            return NextDecision(
                state_name=None,
                should_persists_by_workflow=False,
                router_confirmation_message_for_user=_pending_clarification_message(
                    current_state_name=current_state_name,
                    fallback_message=decision.router_confirmation_message_for_user
                    or "Sorry I couldn't understand the intended question. Please clarify what you want to do.",
                ),
            )

        if decision.state_name not in candidates:
            return NextDecision(
                state_name=None,
                should_persists_by_workflow=False,
                router_confirmation_message_for_user=_pending_clarification_message(
                    current_state_name=current_state_name,
                    fallback_message=(
                        "I could not map that request to an allowed stage from the current state. "
                        "Please clarify the intended stage."
                    ),
                ),
            )

        return NextDecision(
            state_name=decision.state_name,
            should_persists_by_workflow=_pending_should_persist_by_workflow(
                current_state_name=current_state_name,
                selected_state_name=decision.state_name,
                callable_map=self._callable_map,
            ),
        )

    def _decide_aborted(
        self,
        *,
        current_state: State,
        recent_messages: Sequence[ChatMessage],
    ) -> NextDecision:
        current_state_name = _state_name(current_state)
        candidates = _recoverable_candidates(
            current_state_name=current_state_name,
            recoverable_map=self._recoverable_map,
        )
        if len(candidates) == 1:
            return NextDecision(state_name=candidates[0], should_persists_by_workflow=True )

        state_error = _state_error(current_state)
        current_error = state_error.error if state_error is not None else None
        current_system_message = _latest_system_message(_state_messages(current_state))

        candidate_context = _build_aborted_candidate_context(
            candidates=candidates,
            node_descriptions=self._node_name_to_description_map,
        )

        try:
            decision = self._llm.generate_json(
                schema=_RouteDecision,
                system_prompt=ABORTED_ROUTER_SYSTEM_PROMPT,
                user_prompt=build_aborted_router_user_prompt(
                    current_state_name=current_state_name,
                    current_error=current_error,
                    current_system_message=current_system_message,
                    candidate_context=candidate_context,
                    recent_messages=recent_messages,
                ),
                config=LLMConfig(model="basic", temperature=0.1),
                history=None,
                max_attempts=2,
            )
        except Exception as exc:
            log.exception(
                "aborted router llm decision failed",
                current_state_name=current_state_name,
                error=str(exc),
            )
            return NextDecision(
                state_name=None,
                should_persists_by_workflow=False,
                router_confirmation_message_for_user=(
                    "I could not determine the best recovery stage. Please clarify where you want to recover."
                ),
            )

        if decision.state_name is None:
            return NextDecision(
                state_name=None,
                should_persists_by_workflow=False,
                router_confirmation_message_for_user=decision.router_confirmation_message_for_user
                or "Please confirm which stage you want to recover from.",
            )

        if decision.state_name not in candidates:
            return NextDecision(
                state_name=None,
                should_persists_by_workflow=False,
                router_confirmation_message_for_user=(
                    "I could not select a valid recoverable stage. Please clarify where to recover."
                ),
            )

        return NextDecision(state_name=decision.state_name, should_persists_by_workflow=True)


def _pending_candidates(
    *,
    current_state_name: str,
    callable_map: Mapping[str, Sequence[OtherStateInfo]],
) -> list[str]:
    candidates: list[str] = [current_state_name]
    for other in callable_map.get(current_state_name, ()):
        if other.name not in candidates:
            candidates.append(other.name)
    return candidates


def _pending_should_persist_by_workflow(
    *,
    current_state_name: str,
    selected_state_name: str,
    callable_map: Mapping[str, Sequence[OtherStateInfo]],
) -> bool:
    if selected_state_name == current_state_name:
        return True

    for other in callable_map.get(current_state_name, ()):
        if other.name == selected_state_name:
            return other.should_persists_by_workflow

    return False


def _recoverable_candidates(
    *,
    current_state_name: str,
    recoverable_map: Mapping[str, Sequence[str] | str],
) -> list[str]:
    raw_candidates = recoverable_map.get(current_state_name, (current_state_name,))
    if isinstance(raw_candidates, str):
        candidates = [raw_candidates]
    else:
        candidates = list(raw_candidates)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    if not deduped:
        return [current_state_name]
    return deduped


def _pending_clarification_message(
    *,
    current_state_name: str,
    fallback_message: str,
) -> str:
    if current_state_name == DatasetState.NAME:
        return _DATASET_PENDING_CLARIFICATION_MESSAGE
    return fallback_message


def _build_current_state_context(
    *,
    current_state_name: str,
    node_descriptions: Mapping[str, str],
) -> dict[str, str]:
    return {
        "state_name": current_state_name,
        "node_info": node_descriptions.get(current_state_name, ""),
    }


def _build_pending_fellow_state_context(
    *,
    current_state_name: str,
    callable_map: Mapping[str, Sequence[OtherStateInfo]],
    node_descriptions: Mapping[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "state_name": other.name,
            "node_info": node_descriptions.get(other.name, ""),
            "callable_purpose": other.purpose,
        }
        for other in callable_map.get(current_state_name, ())
    ]


def _build_aborted_candidate_context(
    *,
    candidates: Sequence[str],
    node_descriptions: Mapping[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "state_name": state_name,
            "node_info": node_descriptions.get(state_name, ""),
        }
        for state_name in candidates
    ]


def _last_two_messages(messages_history: Sequence[ChatMessage] | None) -> list[ChatMessage]:
    if not messages_history:
        return []
    non_empty_messages = [message for message in messages_history if message.content.strip()]
    if not non_empty_messages:
        return []
    return list(non_empty_messages[-2:])


def _latest_user_message(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history:
        return None
    for message in reversed(messages_history):
        if message.role != "user":
            continue
        content = message.content.strip()
        if content:
            return content
    return None


def _latest_system_message(messages: Sequence[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role == "system" and message.content.strip():
            return message.content.strip()
    return None


def build_pending_router_user_prompt(
    *,
    current_state_context: Mapping[str, str],
    fellow_state_context: Sequence[Mapping[str, str]],
    recent_messages: Sequence[ChatMessage],
) -> str:
    payload: dict[str, Any] = {
        "current_state": dict(current_state_context),
        "routing_mode": "pending",
        "rules": {
            "forward_progression_allowed": False,
            "must_choose_current_or_fellow_state": True,
            "prefer_current_state_by_default": True,
            "prefer_latest_system_message": True,
        },
        "recent_messages": [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in recent_messages
        ],
        "fellow_states": list(fellow_state_context),
        "output_schema": {
            "state_name": "<current state_name, fellow state_name, or null>",
            "router_confirmation_message_for_user": "<short clarification question if state_name is null>",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def build_aborted_router_user_prompt(
    *,
    current_state_name: str,
    current_error: str | None,
    current_system_message: str | None,
    candidate_context: Sequence[Mapping[str, str]],
    recent_messages: Sequence[ChatMessage],
) -> str:
    payload: dict[str, Any] = {
        "current_state": current_state_name,
        "routing_mode": "aborted_recovery",
        "current_error": current_error,
        "current_system_message": current_system_message,
        "rules": {
            "must_choose_from_candidates": True,
            "prefer_latest_system_message": True,
        },
        "recent_messages": [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in recent_messages
        ],
        "recoverable_candidates": list(candidate_context),
        "output_schema": {
            "state_name": "<exact recoverable state_name or null>",
            "router_confirmation_message_for_user": "<short clarification question if state_name is null>",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _read_state_attr(state: State, attr_name: str) -> Any:
    value = getattr(state, attr_name)
    if callable(value):
        return value()
    return value


def _state_name(state: State) -> str:
    value = _read_state_attr(state, "name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("state.name must resolve to a non-empty string")
    return value.strip()


def _state_status(state: State) -> str:
    value = _read_state_attr(state, "status")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("state.status must resolve to a non-empty string")
    return value.strip()


def _state_messages(state: State) -> Sequence[ChatMessage]:
    try:
        value = _read_state_attr(state, "messages")
    except AttributeError:
        return ()
    if value is None:
        return ()
    return value


def _state_error(state: State) -> Any:
    try:
        return _read_state_attr(state, "error")
    except AttributeError:
        return None


def init_next_state_names() -> Mapping[str, str | None]:
    return {
        ProtocolDiscussionState.NAME: DatasetState.NAME,
        DatasetState.NAME: CompileAndValidateNode.NAME,
        CompileAndValidateState.NAME: ModelSelectionState.NAME,
        ModelSelectionState.NAME: ModelTrainState.NAME,
        ModelTrainState.NAME: CausalInferenceState.NAME,
        CausalInferenceState.NAME: NoopDoneState.NAME,
    }


def states_can_call_other_states_during_execution_map() -> Mapping[str, Sequence[OtherStateInfo]]:
    return {
        DatasetState.NAME: (
            OtherStateInfo(
                name=ProtocolDiscussionState.NAME,
                purpose="Protocol discussion collect treatment, outcome, study type, and time zero information to run causal models",
                should_persists_by_workflow= True
            ),
        ),
        ProtocolDiscussionState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Switch to dataset cleaning/inspection during protocol discussion. it cleans data run insights generate graphs",
                should_persists_by_workflow= False
            ),
        ),
        CompileAndValidateState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Answer questions related to data, data characteristics, and insights.",
                should_persists_by_workflow= False
            ),
        ),
        ModelSelectionState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Answer questions related to data, data characteristics, and insights.",
                should_persists_by_workflow= False
            ),
            OtherStateInfo(
                name=CompileAndValidateState.NAME,
                purpose="Answer questions related to the compiled causal, questions about validation.",
                should_persists_by_workflow= False
            ),
        ),
        CausalInferenceState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Route raw data-graph or raw data-analysis requests back to dataset node.",
                should_persists_by_workflow= False
            ),
            OtherStateInfo(
                name=CompileAndValidateState.NAME,
                purpose="Answer questions related to the compiled causal, questions about validation.",
                should_persists_by_workflow= False
            ),
            OtherStateInfo(
                name=ModelSelectionNode.NAME,
                purpose="Answer questions related to model selection and selection rationale.",
                should_persists_by_workflow= False
            ),
        ),
        NoopDoneState.NAME: (),
    }


def recoverable_states_map() -> Mapping[str, Sequence[str]]:
    return {
        DatasetState.NAME: (DatasetState.NAME,),
        ProtocolDiscussionState.NAME: (ProtocolDiscussionState.NAME,),
        CompileAndValidateState.NAME: (
            DatasetState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        ModelSelectionState.NAME: (ProtocolDiscussionState.NAME,),
        ModelTrainState.NAME: (
            ProtocolDiscussionState.NAME,
            ModelSelectionState.NAME,
        ),
        CausalInferenceState.NAME: (
            ModelSelectionState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        NoopDoneState.NAME: (NoopDoneState.NAME,),
    }


def get_node_name_with_description() -> Mapping[str, str]:
    return {
        DatasetNode.NAME: DatasetNode.get_info(),
        ProtocolDiscussionNode.NAME: ProtocolDiscussionNode.get_info(),
        CompileAndValidateNode.NAME: CompileAndValidateNode.get_info(),
        ModelSelectionNode.NAME: ModelSelectionNode.get_info(),
        ModelTrainNode.NAME: ModelTrainNode.get_info(),
        CausalInferenceNode.NAME: CausalInferenceNode.get_info(),
        NoopDoneNode.NAME: NoopDoneNode.get_info(),
    }


def build_state_classes_by_name() -> Mapping[str, type[State]]:
    return {
        DatasetState.NAME: DatasetState,
        ProtocolDiscussionState.NAME: ProtocolDiscussionState,
        CompileAndValidateState.NAME: CompileAndValidateState,
        ModelSelectionState.NAME: ModelSelectionState,
        ModelTrainState.NAME: ModelTrainState,
        CausalInferenceState.NAME: CausalInferenceState,
        NoopDoneState.NAME: NoopDoneState,
    }


def init_all_nodoes_with_name_as_key(
    llm: LLMService,
    data_repo: DataRepo,
    models_repo: ModelsRepo,
    analytics_repo: AnalyticsRepo,
) -> dict[str, Node]:
    tool_factory = DefaultToolFactory(
        data_repo=data_repo,
        models_repo=models_repo,
        analytics_repo=analytics_repo,
        llm_service=llm,
    )

    dataset_node = DatasetNode(data_repo=data_repo, llm=llm, tools_factory=tool_factory)
    protocol_discussion_node = ProtocolDiscussionNode(llm=llm)
    compile_and_validate_node = CompileAndValidateNode(
        llm=llm,
        data_repo=data_repo,
        tool_factory=tool_factory,
    )
    model_selection_node = ModelSelectionNode(llm=llm, tool_factory=tool_factory)
    model_train_node = ModelTrainNode(
        llm=llm,
        data_repo=data_repo,
        tool_factory=tool_factory,
    )
    causal_inference_node = CausalInferenceNode(
        llm=llm,
        data_repo=data_repo,
        tool_factory=tool_factory,
    )
    done_node = NoopDoneNode()

    return {
        dataset_node.name: dataset_node,
        protocol_discussion_node.name: protocol_discussion_node,
        compile_and_validate_node.name: compile_and_validate_node,
        model_selection_node.name: model_selection_node,
        model_train_node.name: model_train_node,
        causal_inference_node.name: causal_inference_node,
        done_node.name: done_node,
    }
