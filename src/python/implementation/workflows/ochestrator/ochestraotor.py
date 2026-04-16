from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from python.domain.models.models import ChatMessage
from python.domain.repo.analytics_repo import AnalyticsRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.domain.workflows.node import Action, Node, NodeRequest, Status
from python.domain.workflows.node_state import NodeState
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import CausalInferenceNode
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import CausalInferenceState
from python.implementation.workflows.nodes.data_compilation.data_compilation_node import DataCompilationNode
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import DataCompilationState
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_node import DataManupulationNode
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import DataManupulationState
from python.implementation.workflows.nodes.data_statistics.data_statistics_node import DataStatisticsNode
from python.implementation.workflows.nodes.data_statistics.data_statistics_state import DataStatisticsState
from python.implementation.workflows.nodes.data_validation.data_validation_node import DataValidationNode
from python.implementation.workflows.nodes.data_validation.data_validation_state import DataValidationState
from python.implementation.workflows.nodes.general_queries.general_queries_node import GeneralQueriesNode
from python.implementation.workflows.nodes.general_queries.general_queries_state import GeneralQueriesState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import ModelSelectionState
from python.implementation.workflows.nodes.model_selection.model_selection_node import ModelSelectionNode
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.noop_done.noop_done_node import NoopDoneNode
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import ProtocolDiscussionNode
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.ochestrator.ochestrator_prompts import ROUTE_SYSTEM_PROMPT
from python.implementation.workflows.ochestrator.writable_ochestrator_state import WritableOchestratorState
from python.implementation.workflows.tools.tools_factory import DefaultToolFactory

# Linear workflow order used for forward-state deletion on abort recovery
_WORKFLOW_STATE_ORDER: tuple[str, ...] = (
    DataManupulationState.NAME,
    ProtocolDiscussionState.NAME,
    DataCompilationState.NAME,
    DataValidationState.NAME,
    ModelSelectionState.NAME,
    ModelTrainState.NAME,
    CausalInferenceState.NAME,
    NoopDoneState.NAME,
)

@dataclass(frozen=True)
class OchestrationResponse:
    messages: Sequence[ChatMessage]
    action: Action
    current_state: str
    current_status: Status
    current_data_id: UUID | None = None
    is_dataset_frozen: bool | None = None


class _RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    node_name: str


class Ochestrator:
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
        self._state_name_by_node_name = build_state_name_by_node_name()
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
        # 1. Save + reload history
        if user_message.content.strip():
            self._workflow_repo.append_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message=user_message,
            )
        history = self._workflow_repo.load_message_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=15,
        )

        # 2. Load orchestrator state
        orch_state = self._workflow_repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if orch_state is None:
            orch_state = WritableOchestratorState.init_empty()
        if not isinstance(orch_state, WritableOchestratorState):
            raise ValueError("Orchestrator state must be WritableOchestratorState")
        
        

        # 3. Pick node to run
        last_msg = history[-1] if history else None
        last_is_user = last_msg is not None and last_msg.role == "user"

        needed_node = orch_state.get_current_node_name()

        if not last_is_user:
            node_name = needed_node
        else:
            companions = orch_state.get_current_node_companion_names(needed_node)
            candidates = list(dict.fromkeys([needed_node, *companions, GeneralQueriesNode.NAME]))
            node_name = self._llm_pick_node(candidates=candidates, history=history)

        # 4. Load node state + run
        node_state = self._load_node_state_or_init(
            user_id=user_id,
            conversation_id=conversation_id,
            node_name=node_name,
        )
        node = self.nodes_by_name.get(node_name)
        if node is None:
            raise ValueError(f"Node {node_name!r} not found")

        result = node.run(
            request=NodeRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                node_state=node_state,
                orchestrator_state=orch_state,
                read_only_messages_history=history,
            )
        )

        # 5. Persist
        new_orch_state = result.new_orchestrator_state
        if not isinstance(new_orch_state, WritableOchestratorState):
            raise ValueError("New orchestrator state must be WritableOchestratorState")
        self._workflow_repo.store_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=new_orch_state,
        )
        self._workflow_repo.store_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=result.new_node_state,
        )
        if result.response_messages:
            self._workflow_repo.append_messages(
                user_id=user_id,
                conversation_id=conversation_id,
                messages=result.response_messages,
            )

        # 6. Filter system messages for caller
        user_facing: list[ChatMessage] = [
            m for m in (result.response_messages or []) if m.role != "system"
        ]

        # 7. Abort handling
        if result.status == "ABORTED":
            new_orch_state.rocover_failure(node_name)
            for state_name in new_orch_state.get_forward_states_after_node(node_name):
                self._workflow_repo.delete_state(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    state_name=state_name,
                )
            self._workflow_repo.store_ochestrator_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state=new_orch_state,
            )
            self._workflow_repo.append_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message=  ChatMessage(
                    role="system",
                    content=(
                        "The workflow encountered an error and has been recovered. "
                    ),       )
        )
            
        return OchestrationResponse(
            messages=user_facing,
            action=result.action,
            current_state=new_orch_state.get_current_node_name(),
            current_status=result.status,
            current_data_id=new_orch_state.get("working_dataset_ids")[-1] if new_orch_state.get("working_dataset_ids") else None,
            is_dataset_frozen=new_orch_state.get("working_dataset_frozen"),
        )
         
    def _llm_pick_node(
        self,
        *,
        candidates: list[str],
        history: Sequence[ChatMessage],
    ) -> str:
        candidates_text = ", ".join(candidates)
        try:
            decision = self._llm.generate_json(
                schema=_RouteDecision,
                system_prompt=ROUTE_SYSTEM_PROMPT,
                user_prompt=(
                    f"Available nodes: [{candidates_text}]\n"
                    "Pick the best node for the user's message."
                ),
                config=LLMConfig(model="mini", temperature=0.1),
                history=list(history[-3:]) if history else None,
                max_attempts=2,
            )
            if decision.node_name in candidates:
                return decision.node_name
        except Exception:
            self._log.warning("LLM route pick failed, falling back to GENERAL_QUERIES")
        return GeneralQueriesNode.NAME

    def _load_node_state_or_init(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        node_name: str,
    ) -> NodeState:
        state_name = self._state_name_by_node_name.get(node_name)
        if state_name is None:
            raise ValueError(f"No state name registered for node {node_name!r}")

        loaded = self._workflow_repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )
        if loaded is not None:
            return loaded

        state_cls = self._state_classes_by_name.get(state_name)
        if state_cls is None:
            raise ValueError(f"No state class registered for {state_name!r}")

        init_empty = getattr(state_cls, "init_empty", None)
        if not callable(init_empty):
            raise ValueError(f"{state_cls.__name__} has no init_empty()")

        state = init_empty()
        if not isinstance(state, NodeState):
            raise ValueError(f"init_empty() for {state_cls.__name__} did not return a NodeState")
        return state


def build_state_classes_by_name() -> Mapping[str, type[NodeState]]:
    return {
        DataManupulationState.NAME: DataManupulationState,
        ProtocolDiscussionState.NAME: ProtocolDiscussionState,
        DataCompilationState.NAME: DataCompilationState,
        DataStatisticsState.NAME: DataStatisticsState,
        DataValidationState.NAME: DataValidationState,
        ModelSelectionState.NAME: ModelSelectionState,
        ModelTrainState.NAME: ModelTrainState,
        CausalInferenceState.NAME: CausalInferenceState,
        NoopDoneState.NAME: NoopDoneState,
        GeneralQueriesState.NAME: GeneralQueriesState,
    }


def build_state_name_by_node_name() -> Mapping[str, str]:
    return {
        DataManupulationNode.NAME: DataManupulationState.NAME,
        ProtocolDiscussionNode.NAME: ProtocolDiscussionState.NAME,
        DataCompilationNode.NAME: DataCompilationState.NAME,
        DataStatisticsNode.NAME: DataStatisticsState.NAME,
        DataValidationNode.NAME: DataValidationState.NAME,
        ModelSelectionNode.NAME: ModelSelectionState.NAME,
        ModelTrainNode.NAME: ModelTrainState.NAME,
        CausalInferenceNode.NAME: CausalInferenceState.NAME,
        NoopDoneNode.NAME: NoopDoneState.NAME,
        GeneralQueriesNode.NAME: GeneralQueriesState.NAME,
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

    data_manupulation_node = DataManupulationNode(
        data_repo=data_repo,
        llm=llm,
        tools_factory=tool_factory,
    )
    protocol_discussion_node = ProtocolDiscussionNode(llm=llm)
    data_compilation_node = DataCompilationNode(
        data_repo=data_repo,
        llm=llm,
        tools_factory=tool_factory,
    )
    data_statistics_node = DataStatisticsNode(
        data_repo=data_repo,
        llm=llm,
        tools_factory=tool_factory,
    )
    data_validation_node = DataValidationNode(
        data_repo=data_repo,
        llm=llm,
        tools_factory=tool_factory,
    )
    model_selection_node = ModelSelectionNode(
        llm=llm,
        tools_factory=tool_factory,
    )
    model_train_node = ModelTrainNode(
        llm=llm,
        data_repo=data_repo,
        tools_factory=tool_factory,
    )
    causal_inference_node = CausalInferenceNode(
        llm=llm,
        data_repo=data_repo,
        tools_factory=tool_factory,
    )
    done_node = NoopDoneNode()
    general_queries_node = GeneralQueriesNode(llm=llm)

    return {
        data_manupulation_node.name: data_manupulation_node,
        protocol_discussion_node.name: protocol_discussion_node,
        data_compilation_node.name: data_compilation_node,
        data_statistics_node.name: data_statistics_node,
        data_validation_node.name: data_validation_node,
        model_selection_node.name: model_selection_node,
        model_train_node.name: model_train_node,
        causal_inference_node.name: causal_inference_node,
        done_node.name: done_node,
        general_queries_node.name: general_queries_node,
    }
