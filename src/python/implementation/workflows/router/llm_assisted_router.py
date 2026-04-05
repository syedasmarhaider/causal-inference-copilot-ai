from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

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
from python.implementation.workflows.tools.tools_factory import DefaultToolFactory

log = get_logger(__name__, component="LLMAssistedRouter", log_type="workflow_router")


class _RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    state_name: str | None = None
    router_confirmation_message_for_user: str | None = None


@dataclass(frozen=True)
class OtherStateInfo:
    name: str
    purpose: str


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
            return NextDecision(state_name=DatasetState.NAME, persist_as_active=True)

        current_name = current_state.name()
        status = current_state.status()
        recent_messages = _last_two_messages(messages_history)

        if status in ("DONE", "FREEZED"):
            next_name = self._deterministic_done_next_state(
                current_state_name=current_name,
                recent_messages=recent_messages,
            )
            if next_name is None:
                return NextDecision(state_name=current_name, persist_as_active=True)
            return NextDecision(state_name=next_name, persist_as_active=True)

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

    def _deterministic_done_next_state(
        self,
        *,
        current_state_name: str,
        recent_messages: Sequence[ChatMessage],
    ) -> str | None:
        _ = recent_messages
        if current_state_name == CausalInferenceNode.NAME:
            return CausalInferenceNode.NAME

        return self._next_state_names_map.get(current_state_name)

    def _decide_pending(
        self,
        *,
        current_state_name: str,
        recent_messages: Sequence[ChatMessage],
    ) -> NextDecision:
        candidates = _pending_candidates(current_state_name=current_state_name, callable_map=self._callable_map)
        decision = self._llm.generate_json(
            schema=_RouteDecision,
            system_prompt=_pending_router_system_prompt(),
            user_prompt=_pending_router_user_prompt(
                current_state_name=current_state_name,
                candidates=candidates,
                recent_messages=recent_messages,
                node_descriptions=self._node_name_to_description_map,
                callable_map=self._callable_map,
            ),
            config=LLMConfig(model="mini", temperature=0.1),
            history=list(recent_messages) or None,
            max_attempts=2,
        )

        if decision.state_name is None:
            return NextDecision(
                state_name=None,
                persist_as_active=None,
                router_confirmation_message_for_user=decision.router_confirmation_message_for_user
                or "Please clarify which stage you want: stay here or switch to a related stage.",
            )

        if decision.state_name not in candidates:
            return NextDecision(
                state_name=None,
                persist_as_active=None,
                router_confirmation_message_for_user=(
                    "I could not map that request to an allowed stage from the current state. "
                    "Please clarify the intended stage."
                ),
            )

        return NextDecision(
            state_name=decision.state_name,
            persist_as_active=_persist_for_pending_transition(
                current_state_name=current_state_name,
                selected_state_name=decision.state_name,
            ),
        )

    def _decide_aborted(
        self,
        *,
        current_state: State,
        recent_messages: Sequence[ChatMessage],
    ) -> NextDecision:
        current_state_name = current_state.name()
        candidates = list(self._recoverable_map.get(current_state_name, (current_state_name,)))

        state_error = current_state.error()
        current_error = state_error.error if state_error is not None else None
        current_system_message = _latest_system_message(current_state.messages())

        decision = self._llm.generate_json(
            schema=_RouteDecision,
            system_prompt=_aborted_router_system_prompt(),
            user_prompt=_aborted_router_user_prompt(
                current_state_name=current_state_name,
                current_error=current_error,
                current_system_message=current_system_message,
                candidates=candidates,
                recent_messages=recent_messages,
                node_descriptions=self._node_name_to_description_map,
            ),
            config=LLMConfig(model="basic", temperature=0.1),
            history=list(recent_messages) or None,
            max_attempts=2,
        )

        if decision.state_name is None:
            return NextDecision(
                state_name=None,
                persist_as_active=None,
                router_confirmation_message_for_user=decision.router_confirmation_message_for_user
                or "Please confirm which stage you want to recover from.",
            )

        if decision.state_name not in candidates:
            return NextDecision(
                state_name=None,
                persist_as_active=None,
                router_confirmation_message_for_user=(
                    "I could not select a valid recoverable stage. Please clarify where to recover."
                ),
            )

        return NextDecision(state_name=decision.state_name, persist_as_active=True)


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


def _persist_for_pending_transition(
    *,
    current_state_name: str,
    selected_state_name: str,
) -> bool:
    if selected_state_name == current_state_name:
        return True

    if {current_state_name, selected_state_name} == {DatasetState.NAME, ProtocolDiscussionState.NAME}:
        return True

    if (
        selected_state_name == DatasetState.NAME
        and current_state_name
        in {
            CompileAndValidateState.NAME,
            ModelSelectionState.NAME,
            ModelTrainState.NAME,
            CausalInferenceState.NAME,
        }
    ):
        return False

    return True


def _last_two_messages(messages_history: Sequence[ChatMessage] | None) -> list[ChatMessage]:
    if not messages_history:
        return []
    return list(messages_history[-2:])


def _latest_system_message(messages: Sequence[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role == "system" and message.content.strip():
            return message.content.strip()
    return None


def _pending_router_system_prompt() -> str:
    return (
        "You are a strict workflow router. Choose exactly one allowed state from candidates. "
        "The current state is PENDING, so forward progression is not allowed. "
        "If one of the recent messages is a system message, prioritize it strongly over older context. "
        "If unclear, return state_name=null with a short confirmation question."
    )


def _aborted_router_system_prompt() -> str:
    return (
        "You are a strict workflow recovery router. The current state is ABORTED. "
        "Choose one recoverable state from the provided candidates. "
        "If one of the recent messages is a system message, prioritize it strongly over older context. "
        "If unclear, return state_name=null with a short confirmation question."
    )


def _pending_router_user_prompt(
    *,
    current_state_name: str,
    candidates: Sequence[str],
    recent_messages: Sequence[ChatMessage],
    node_descriptions: Mapping[str, str],
    callable_map: Mapping[str, Sequence[OtherStateInfo]],
) -> str:
    candidate_context: list[dict[str, str]] = []
    purpose_by_name = {
        item.name: item.purpose
        for item in callable_map.get(current_state_name, ())
    }
    for name in candidates:
        candidate_context.append(
            {
                "state_name": name,
                "node_info": node_descriptions.get(name, ""),
                "callable_purpose": purpose_by_name.get(name, ""),
            }
        )

    payload = {
        "current_state": current_state_name,
        "recent_messages": [
            {"role": m.role, "content": m.content}
            for m in recent_messages
        ],
        "prioritize_system_instruction": any(m.role == "system" for m in recent_messages),
        "allowed_candidates": candidate_context,
        "output_schema": {
            "state_name": "<candidate state name or null>",
            "router_confirmation_message_for_user": "<short question if state_name is null>",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _aborted_router_user_prompt(
    *,
    current_state_name: str,
    current_error: str | None,
    current_system_message: str | None,
    candidates: Sequence[str],
    recent_messages: Sequence[ChatMessage],
    node_descriptions: Mapping[str, str],
) -> str:
    payload = {
        "current_state": current_state_name,
        "current_error": current_error,
        "current_system_message": current_system_message,
        "recoverable_candidates": [
            {
                "state_name": name,
                "node_info": node_descriptions.get(name, ""),
            }
            for name in candidates
        ],
        "recent_messages": [
            {"role": m.role, "content": m.content}
            for m in recent_messages
        ],
        "prioritize_system_instruction": any(m.role == "system" for m in recent_messages),
        "output_schema": {
            "state_name": "<recoverable state name or null>",
            "router_confirmation_message_for_user": "<short question if state_name is null>",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


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
                purpose="Move to protocol discussion when user is working on causal protocol definition.",
            ),
        ),
        ProtocolDiscussionState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Switch to dataset cleaning/inspection during protocol discussion.",
            ),
        ),
        CompileAndValidateState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Answer questions related to data, data characteristics, and insights.",
            ),
        ),
        ModelSelectionState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Answer questions related to data, data characteristics, and insights.",
            ),
            OtherStateInfo(
                name=CompileAndValidateState.NAME,
                purpose="Answer questions related to the compiled causal, questions about validation.",
            ),
        ),
        CausalInferenceState.NAME: (
            OtherStateInfo(
                name=DatasetState.NAME,
                purpose="Route raw data-graph or raw data-analysis requests back to dataset node.",
            ),
            OtherStateInfo(
                name=CompileAndValidateState.NAME,
                purpose="Answer questions related to the compiled causal, questions about validation.",
            ),
            OtherStateInfo(
                name=ModelSelectionNode.NAME,
                purpose="Answer questions related to model selection and selection rationale.",
            ),
        ),
        NoopDoneState.NAME: (),
    }


def recoverable_states_map() -> Mapping[str, Sequence[str]]:
    return {
        DatasetState.NAME: (DatasetState.NAME,),
        ProtocolDiscussionState.NAME: (ProtocolDiscussionState.NAME, DatasetState.NAME),
        CompileAndValidateState.NAME: (
            DatasetState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        ModelSelectionState.NAME: (
            ProtocolDiscussionState.NAME,
        ),
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
