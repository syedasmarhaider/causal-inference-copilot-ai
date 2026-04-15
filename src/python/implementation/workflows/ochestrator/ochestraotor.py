from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from python.domain.models.models import ChatMessage
from python.domain.repo.analytics_repo import AnalyticsRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.node_state import State
from python.implementation.service.logging.default_logging import get_app_logger
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
from python.implementation.workflows.nodes.model_train.model_train_node import (
    ModelTrainNode,
)
from python.implementation.workflows.nodes.model_train.model_train_state import (
    ModelTrainState,
)
from python.implementation.workflows.nodes.noop_done.noop_done_node import (
    NoopDoneNode,
)
from python.implementation.workflows.nodes.noop_done.noop_done_state import (
    NoopDoneState,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    ProtocolDiscussionNode,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.ochestrator.ochestrator_global_state import (
    OchestratorReadOnlyGlobalState,
    OchestratorWritableGlobalState,
)
from python.implementation.workflows.ochestrator.ochestrator_prompts import (
    OCHESTRATOR_ABORTED_SYSTEM_PROMPT,
    OCHESTRATOR_PENDING_ORCHESTRATOR_ANSWER_SYSTEM_PROMPT,
    OCHESTRATOR_PENDING_ROUTE_SYSTEM_PROMPT,
)
from python.implementation.workflows.tools.tools_factory import DefaultToolFactory


@dataclass(frozen=True)
class OchestrationResponse:
    messages: Sequence[ChatMessage]
    state: State
    ochestrator_state: OchestratorReadOnlyGlobalState


class Ochestrator:
    _llm: LLMService
    _workflow_repo: WorkflowStateRepo
    _node_name_to_description_map: Mapping[str, str]
    _state_name_to_description_map: Mapping[str, str]
    _state_classes_by_name: Mapping[str, type[State]]
    _node_name_by_state_name: Mapping[str, str]
    _state_name_by_node_name: Mapping[str, str]
    _next_node_names_by_current_state_name: Mapping[str, str]
    _recoverable_candidates_map: Mapping[str, Sequence[str]]

    def __init__(
        self,
        workflow_repo: WorkflowStateRepo,
        llm: LLMService,
        data_repo: DataRepo,
        models_repo: ModelsRepo,
        analytics_repo: AnalyticsRepo,
    ) -> None:
        self._workflow_repo = workflow_repo
        self._llm = llm

        self.nodes_by_name = init_all_nodes_with_name_as_key(
            llm=llm,
            data_repo=data_repo,
            models_repo=models_repo,
            analytics_repo=analytics_repo,
        )
        self._state_classes_by_name = build_state_classes_by_name()
        self._node_name_by_state_name = build_node_name_by_state_name()
        self._state_name_by_node_name = build_state_name_by_node_name()
        self._next_node_names_by_current_state_name = build_next_node_name_by_current_state_name()
        self._recoverable_candidates_map = recoverable_states_map()

        self._node_name_to_description_map = get_node_name_with_description()
        self._state_name_to_description_map = {
            state_name: self._node_name_to_description_map.get(node_name, "")
            for state_name, node_name in self._node_name_by_state_name.items()
        }
        self._log = get_app_logger(
            __name__,
            component=self.__class__.__name__,
            log_type="workflow_service",
        )

    def answer(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        user_message: ChatMessage,
    ) -> OchestrationResponse:
        self._workflow_repo.append_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message=user_message,
        )

        messages_history = self._workflow_repo.load_message_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=15,
        )

        ochestrator_state = self._workflow_repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        
        if ochestrator_state is None:
            ochestrator_state = OchestratorWritableGlobalState.init_empty()

        if not isinstance(ochestrator_state, OchestratorWritableGlobalState):
            raise ValueError(
                "Ochestrator state should be of type OchestratorWritableGlobalState"
            )

        needed_node_name = ochestrator_state.needs_node_name() 
        current_state = self._load_state_for_node_or_init(
            user_id=user_id,
            conversation_id=conversation_id,
            node_name=needed_node_name,
        )

        if current_state.status() == "DONE":
            self._log.warning(
                "Resetting stale DONE state before rerun",
                state_name=current_state.name(),
                needed_node_name=needed_node_name,
            )
            current_state.set_status_pending()
            if current_state.status() == "DONE":
                raise ValueError(
                    f"State {current_state.name()!r} remained DONE after stale-state reset"
                )

        match current_state.status():
            case "PENDING":
                response = self.handle_pending(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    current_state=current_state,
                    ochestrator_state=ochestrator_state,
                    messages_history=messages_history,
                )
                
            case "ABORTED":
                response = self.handle_abort(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    current_state=current_state,
                    ochestrator_state=ochestrator_state,
                    messages_history=messages_history,
                )
            case unsupported_status:
                raise ValueError(f"Unsupported state status: {unsupported_status!r}")


        return OchestrationResponse(
            messages=response.messages,
            state=response.state,
            ochestrator_state=response.ochestrator_state,
        )

    def handle_abort(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        current_state: State,
        ochestrator_state: OchestratorWritableGlobalState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> OchestrationResponse:
        if current_state.status() != "ABORTED":
            raise ValueError(
                f"handle_abort expected ABORTED state, got {current_state.status()!r}"
            )

        recovery_state_name = self._decide_recovery_state_name_on_abort(
            current_state=current_state,
        )
        recovery_node_name = self._node_name_by_state_name[recovery_state_name]

        ochestrator_state.rollback_orchestrator_global_state(recovery_node_name)

        self._delete_forward_states_after_recovery_point(
            user_id=user_id,
            conversation_id=conversation_id,
            recovery_state_name=recovery_state_name,
        )

        if recovery_state_name == current_state.name():
            recovery_state = current_state
        else:
            recovery_state = self._load_state_by_name_or_init(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=recovery_state_name,
            )

        recovery_state.set_status_pending()
        
        self._workflow_repo.store_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=ochestrator_state,
        )
        self._workflow_repo.store_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=recovery_state,
        )

        return self._run_node_and_persist(
            user_id=user_id,
            conversation_id=conversation_id,
            node_name=recovery_node_name,
            input_state=recovery_state,
            ochestrator_state=ochestrator_state,
            messages_history=messages_history,
        )
        
    def handle_pending(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        current_state: State,
        ochestrator_state: OchestratorWritableGlobalState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> OchestrationResponse:
        node_name_to_run = self._decide_node_name_or_ochestration_on_pending(
            current_state=current_state,
            ochestrator_state=ochestrator_state,
            messages_history=messages_history,
        )

        if node_name_to_run == "ORCHESTRATOR_ANSWER":
            orchestrator_answer_message = self._answer_orchestrator_question_on_pending(
                current_state=current_state,
                ochestrator_state=ochestrator_state,
                messages_history=messages_history,
            )
            return OchestrationResponse(
                messages=[orchestrator_answer_message],
                state=current_state,
                ochestrator_state=ochestrator_state,
            )

        current_node_name = self._node_name_by_state_name.get(current_state.name())
        if current_node_name is None:
            raise ValueError(
                f"No node name registered for state {current_state.name()!r}"
            )
   
        state_to_run = current_state
        if node_name_to_run != current_node_name:
                state_to_run = self._load_state_for_node_or_init(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    node_name=node_name_to_run,
                )   
                
        return self._run_node_and_persist(
            user_id=user_id,
            conversation_id=conversation_id,
            node_name=node_name_to_run,
            input_state=state_to_run,
            ochestrator_state=ochestrator_state,
            messages_history=messages_history,
        )

    def _run_node_and_persist(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        node_name: str,
        input_state: State,
        ochestrator_state: OchestratorWritableGlobalState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> OchestrationResponse:
        node = self.nodes_by_name.get(node_name)
        if node is None:
            raise ValueError(f"Node with name {node_name!r} was not found")
        
        resulted_state = node.run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=input_state,
            readonly_orchestrator_state=ochestrator_state,
            messages_history=messages_history,
        )

        ochestrator_state.update_ochestration_working_state_if_node_done(
            state=resulted_state,
        )
        
        self._workflow_repo.store_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=ochestrator_state,
        )
        self._workflow_repo.store_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=resulted_state,
        )

        delta_messages =  resulted_state.messages()
        return OchestrationResponse(
            messages= delta_messages,
            state=resulted_state,
            ochestrator_state=ochestrator_state,
        )

    def _load_state_for_node_or_init(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        node_name: str,
    ) -> State:
        state_name = self._state_name_by_node_name.get(node_name)
        if state_name is None:
            raise ValueError(f"No state name registered for node {node_name!r}")

        return self._load_state_by_name_or_init(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )

    def _load_state_by_name_or_init(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> State:
        state = self._workflow_repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )
        if state is not None:
            return state

        return self._init_state(state_name)

    def _init_state(self, state_name: str) -> State:
        state_cls = self._state_classes_by_name.get(state_name)
        if state_cls is None:
            raise ValueError(
                f"No state class registered for state name {state_name!r}"
            )

        init_empty = getattr(state_cls, "init_empty", None)
        if init_empty is None or not callable(init_empty):
            raise ValueError(
                f"State class {state_cls.__name__!r} does not expose callable init_empty()"
            )

        state = init_empty()
        if not isinstance(state, State):
            raise ValueError(
                f"init_empty() for {state_cls.__name__!r} did not return a State instance"
            )

        return state

    def _decide_node_name_or_ochestration_on_pending(
        self,
        *,
        current_state: State,
        ochestrator_state: OchestratorReadOnlyGlobalState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> str:
        last_2_messages = _last_two_messages(messages_history)

        current_state_name = current_state.name()
        current_node_name = self._node_name_by_state_name.get(current_state_name)
        if current_node_name is None:
            raise ValueError(
                f"No node name registered for state {current_state_name!r}"
            )

        latest_user_message = _latest_user_message(messages_history)
        if latest_user_message is None:
            return current_node_name

        latest_state_system_message = _latest_system_message(current_state.messages())
        current_state_description = self._state_name_to_description_map.get(
            current_state_name,
            "",
        )
        
        other_state_description = ""
        if current_state.name() == DatasetState.NAME:
            other_state_description = self._state_name_to_description_map.get(
                ProtocolDiscussionState.NAME,
                "",
            )
        else:
             other_state_description = self._state_name_to_description_map.get(
                DatasetState.NAME,
                "",
            )    
             
        decision = self._llm.generate_json(
            schema=_PendingRouteDecision,
            system_prompt=OCHESTRATOR_PENDING_ROUTE_SYSTEM_PROMPT,
            user_prompt=(
                f"Current pending state: {current_state_name!r}\n"
                f"Current state description: {current_state_description!r}\n"
                f"Other state description: {other_state_description!r}\n"
                f"Latest user message: {latest_user_message!r}\n"
                f"Latest system message inside current state: {latest_state_system_message!r}\n"
                "Return the best route intent."
            ),
            config=LLMConfig(model="mini", temperature=0.1),
            history=last_2_messages,
            max_attempts=2,
        )

        match decision.route_intent:
            case "CURRENT_STATE":
                return current_node_name
            case "DATASET":
                return DatasetNode.NAME
            case "ORCHESTRATOR_ANSWER":
                return "ORCHESTRATOR_ANSWER"

        raise ValueError(
            f"Unsupported pending route intent: {decision.route_intent!r}"
        )

    def _answer_orchestrator_question_on_pending(
        self,
        *,
        current_state: State,
        ochestrator_state: OchestratorReadOnlyGlobalState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> ChatMessage:
        last_3_messages: list[ChatMessage] = (
            list(messages_history[-3:]) if messages_history else []
        )
        latest_user_message = _latest_user_message(messages_history)
        if latest_user_message is None:
            raise ValueError(
                "Latest user message is required for pending orchestrator answer"
            )

        current_state_name = current_state.name()
        current_state_description = self._state_name_to_description_map.get(
            current_state_name,
            "",
        )
        latest_state_system_message = _latest_system_message(current_state.messages())
        orchestrator_payload = _build_pending_orchestrator_payload(ochestrator_state)

        next_node_name = self._next_node_names_by_current_state_name.get(
            current_state_name
        )
        next_node_description = (
            self._node_name_to_description_map.get(next_node_name, "")
            if next_node_name is not None
            else ""
        )

        answer_text = self._llm.generate(
            system_prompt=OCHESTRATOR_PENDING_ORCHESTRATOR_ANSWER_SYSTEM_PROMPT,
            user_prompt=(
                f"Current pending state: {current_state_name!r}\n"
                f"Current state description: {current_state_description!r}\n"
                f"Latest system message inside current state: {latest_state_system_message!r}\n"
                f"Orchestrator payload: {orchestrator_payload!r}\n"
                f"Next node after completion of current state: {next_node_name!r}\n"
                f"Next node description: {next_node_description!r}\n"
                f"User question: {latest_user_message!r}\n"
            ),
            history=last_3_messages,
            config=LLMConfig(model="basic", temperature=0.2),
        ).content

        return ChatMessage(role="assistant", content=answer_text)

    def _decide_recovery_state_name_on_abort(
        self,
        *,
        current_state: State,
    ) -> str:
        current_state_name = current_state.name()
        candidates = self._recoverable_candidates_map.get(
            current_state_name,
            (current_state_name,),
        )

        if len(candidates) == 1:
            return candidates[0]

        state_error = current_state.error()
        current_error = state_error.error if state_error is not None else None
        current_system_message = _latest_system_message(current_state.messages())

        candidate_context = _build_aborted_candidate_context(
            candidates=candidates,
            node_descriptions=self._state_name_to_description_map,
        )

        decision = self._llm.generate_json(
            schema=_AbortedRouteDecision,
            system_prompt=OCHESTRATOR_ABORTED_SYSTEM_PROMPT,
            user_prompt=(
                f"The current state is {current_state_name!r} and it is in ABORTED status. "
                f"The error is: {current_error!r}. "
                f"The latest system message is: {current_system_message!r}. "
                f"Choose the most appropriate recovery state from: {candidate_context}"
            ),
            config=LLMConfig(model="basic", temperature=0.1),
            history=None,
            max_attempts=2,
        )

        if decision.state_name is None:
            raise ValueError(
                "LLM failed to choose a recovery state for ABORTED state"
            )
        if decision.state_name not in candidates:
            raise ValueError(
                f"LLM chose invalid recovery state {decision.state_name!r}; "
                f"candidates were {candidates!r}"
            )

        return decision.state_name


    def _delete_forward_states_after_recovery_point(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        recovery_state_name: str,
    ) -> None:
        states_to_delete = _forward_state_names_to_delete_after_recovery(
            recovery_state_name=recovery_state_name
        )

        for state_name in states_to_delete:
            self._workflow_repo.delete_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name,
            )


def _last_two_messages(
    messages_history: Sequence[ChatMessage] | None,
) -> list[ChatMessage]:
    if not messages_history:
        return []

    non_empty_messages = [
        message for message in messages_history if message.content.strip()
    ]
    if not non_empty_messages:
        return []

    return list(non_empty_messages[-2:])


def _latest_user_message(
    messages_history: Sequence[ChatMessage] | None,
) -> str | None:
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

def _forward_state_names_to_delete_after_recovery(
    *,
    recovery_state_name: str,
) -> tuple[str, ...]:
    if recovery_state_name == ProtocolDiscussionState.NAME:
        return (
            DatasetState.NAME,
            CompileAndValidateState.NAME,
            ModelSelectionState.NAME,
            ModelTrainState.NAME,
            CausalInferenceState.NAME,
            NoopDoneState.NAME,
        )

    if recovery_state_name == DatasetState.NAME:
        return (
            CompileAndValidateState.NAME,
            ModelSelectionState.NAME,
            ModelTrainState.NAME,
            CausalInferenceState.NAME,
            NoopDoneState.NAME,
        )

    if recovery_state_name == CompileAndValidateState.NAME:
        return (
            ModelSelectionState.NAME,
            ModelTrainState.NAME,
            CausalInferenceState.NAME,
            NoopDoneState.NAME,
        )

    if recovery_state_name == ModelSelectionState.NAME:
        return (
            ModelTrainState.NAME,
            CausalInferenceState.NAME,
            NoopDoneState.NAME,
        )

    if recovery_state_name == ModelTrainState.NAME:
        return (
            CausalInferenceState.NAME,
            NoopDoneState.NAME,
        )

    if recovery_state_name == CausalInferenceState.NAME:
        return (NoopDoneState.NAME,)

    if recovery_state_name == NoopDoneState.NAME:
        return ()

    raise ValueError(
        f"Unsupported recovery state for forward-state deletion: {recovery_state_name!r}"
    )


class _PendingRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    route_intent: Literal[
        "CURRENT_STATE",
        "DATASET",
        "ORCHESTRATOR_ANSWER",
    ]


def _build_pending_orchestrator_payload(
    ochestrator_state: OchestratorReadOnlyGlobalState,
) -> dict[str, Any]:
    keys = (
        "causal_spec",
        "data_transformation_plan",
        "validation_issues",
        "selected_model",
    )
    return {key: ochestrator_state.get(key) for key in keys}

class _AbortedRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    state_name: str | None = None


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


def recoverable_states_map() -> Mapping[str, Sequence[str]]:
    return {
        DatasetState.NAME: (DatasetState.NAME,),
        ProtocolDiscussionState.NAME: (ProtocolDiscussionState.NAME,),
        CompileAndValidateState.NAME: (
            DatasetState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        ModelSelectionState.NAME: (
            DatasetState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        ModelTrainState.NAME: (
            ModelSelectionState.NAME,
            DatasetState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        CausalInferenceState.NAME: (
            ModelSelectionState.NAME,
            DatasetState.NAME,
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


def build_node_name_by_state_name() -> Mapping[str, str]:
    return {
        DatasetState.NAME: DatasetNode.NAME,
        ProtocolDiscussionState.NAME: ProtocolDiscussionNode.NAME,
        CompileAndValidateState.NAME: CompileAndValidateNode.NAME,
        ModelSelectionState.NAME: ModelSelectionNode.NAME,
        ModelTrainState.NAME: ModelTrainNode.NAME,
        CausalInferenceState.NAME: CausalInferenceNode.NAME,
        NoopDoneState.NAME: NoopDoneNode.NAME,
    }


def build_state_name_by_node_name() -> Mapping[str, str]:
    return {
        DatasetNode.NAME: DatasetState.NAME,
        ProtocolDiscussionNode.NAME: ProtocolDiscussionState.NAME,
        CompileAndValidateNode.NAME: CompileAndValidateState.NAME,
        ModelSelectionNode.NAME: ModelSelectionState.NAME,
        ModelTrainNode.NAME: ModelTrainState.NAME,
        CausalInferenceNode.NAME: CausalInferenceState.NAME,
        NoopDoneNode.NAME: NoopDoneState.NAME,
    }


def build_next_node_name_by_current_state_name() -> Mapping[str, str]:
    return {
        DatasetState.NAME: ProtocolDiscussionNode.NAME,
        ProtocolDiscussionState.NAME: CompileAndValidateNode.NAME,
        CompileAndValidateState.NAME: ModelSelectionNode.NAME,
        ModelSelectionState.NAME: ModelTrainNode.NAME,
        ModelTrainState.NAME: CausalInferenceNode.NAME,
        CausalInferenceState.NAME: NoopDoneNode.NAME,
        NoopDoneState.NAME: NoopDoneNode.NAME,
    }


def init_all_nodes_with_name_as_key(
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

    dataset_node = DatasetNode(
        data_repo=data_repo,
        llm=llm,
        tools_factory=tool_factory,
    )
    protocol_discussion_node = ProtocolDiscussionNode(llm=llm)
    compile_and_validate_node = CompileAndValidateNode(
        llm=llm,
        data_repo=data_repo,
        tool_factory=tool_factory,
    )
    model_selection_node = ModelSelectionNode(
        llm=llm,
        tool_factory=tool_factory,
    )
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
