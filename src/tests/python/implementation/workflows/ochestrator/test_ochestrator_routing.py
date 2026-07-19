from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from python.domain.models.models import ChatMessage
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
)
from python.implementation.workflows.nodes.causal_validate.causal_validate_node import (
    CausalValidateNode,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_node import (
    DataManupulationNode,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
from python.implementation.workflows.nodes.general_queries.general_queries_node import (
    GeneralQueriesNode,
)
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.shap_explanation.shap_explanation_node import (
    ShapExplanationNode,
)
from python.implementation.workflows.ochestrator.causal_ochestrator_state import (
    CausalOchestratorState,
)
from python.implementation.workflows.ochestrator.data_ochestrator_state import (
    DataOchestratorState,
)
from python.implementation.workflows.ochestrator.ochestraotor import (
    Ochestrator,
    build_node_name_by_node_name,
    build_state_classes_by_name,
    build_state_name_by_node_name,
    causal_validate_enabled_from_env,
    init_all_nodes_with_name_as_key,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass
class _StubLogger:
    warning_calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def warning(self, message: str, /, **kwargs: object) -> None:
        self.warning_calls.append((message, kwargs))


class _RaisingLLM:
    def generate(self, **_: object) -> object:
        raise NotImplementedError

    def generate_json(self, **_: object) -> object:
        raise RuntimeError("routing failed")


class _ReturningLLM:
    def __init__(self, node_name: str) -> None:
        self._node_name = node_name

    def generate(self, **_: object) -> object:
        raise NotImplementedError

    def generate_json(self, **_: object) -> object:
        return SimpleNamespace(node_name=self._node_name)


def _build_orchestrator_for_routing(llm: object) -> Ochestrator:
    orchestrator = object.__new__(Ochestrator)
    orchestrator._llm = llm
    orchestrator._log = _StubLogger()
    orchestrator._node_classes_by_name = build_node_name_by_node_name()
    orchestrator._shap_enabled = True
    orchestrator._causal_validate_enabled = True
    return orchestrator


def _user_history() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="What should I do next?")]


def _summary() -> DatasetSummaryModel:
    return DatasetSummaryModel(n_rows=10, profiles=[])


def test_llm_pick_node_falls_back_to_current_node_for_data_conversations() -> None:
    orchestrator = _build_orchestrator_for_routing(_RaisingLLM())
    orch_state = DataOchestratorState.init_empty()

    selected = orchestrator._llm_pick_node(
        ochestrator_state=orch_state,
        current_node=DataManupulationNode.NAME,
        companions=[],
        history=_user_history(),
    )

    assert selected == DataManupulationNode.NAME


def test_llm_pick_node_rejects_general_queries_for_data_conversations() -> None:
    orchestrator = _build_orchestrator_for_routing(_ReturningLLM(GeneralQueriesNode.NAME))
    orch_state = DataOchestratorState.init_empty()
    orch_state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": DataOchestratorState.INIT_DATA_ID,
            "latest_dataset_summary": _summary(),
        },
    )

    selected = orchestrator._llm_pick_node(
        ochestrator_state=orch_state,
        current_node=DataManupulationNode.NAME,
        companions=orch_state.get_current_node_companion_names(DataManupulationNode.NAME),
        history=_user_history(),
    )

    assert selected == DataManupulationNode.NAME


def test_llm_pick_node_can_still_fall_back_to_general_queries_when_allowed() -> None:
    orchestrator = _build_orchestrator_for_routing(_RaisingLLM())
    orch_state = CausalOchestratorState.init_empty()
    orch_state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": CausalOchestratorState.INIT_DATA_ID,
            "latest_dataset_summary": _summary(),
        },
    )

    selected = orchestrator._llm_pick_node(
        ochestrator_state=orch_state,
        current_node=DataManupulationNode.NAME,
        companions=orch_state.get_current_node_companion_names(DataManupulationNode.NAME),
        history=_user_history(),
    )

    assert selected == GeneralQueriesNode.NAME


def test_orchestrator_removes_shap_companion_when_disabled() -> None:
    orchestrator = _build_orchestrator_for_routing(_RaisingLLM())
    orchestrator._shap_enabled = False

    filtered = orchestrator._get_enabled_companion_names(
        [ShapExplanationNode.NAME, GeneralQueriesNode.NAME]
    )

    assert filtered == [GeneralQueriesNode.NAME]


def test_orchestrator_removes_causal_validate_companion_when_disabled() -> None:
    orchestrator = _build_orchestrator_for_routing(_RaisingLLM())
    orchestrator._causal_validate_enabled = False

    filtered = orchestrator._get_enabled_companion_names(
        [CausalValidateNode.NAME, ShapExplanationNode.NAME, GeneralQueriesNode.NAME]
    )

    assert filtered == [ShapExplanationNode.NAME, GeneralQueriesNode.NAME]


def test_causal_validate_flag_defaults_disabled_and_accepts_true(monkeypatch) -> None:
    monkeypatch.delenv("CAUSAL_VALIDATE_ENABLED", raising=False)
    assert causal_validate_enabled_from_env() is False

    monkeypatch.setenv("CAUSAL_VALIDATE_ENABLED", "true")
    assert causal_validate_enabled_from_env() is True


def test_causal_validate_node_and_state_are_registered() -> None:
    assert build_node_name_by_node_name()[CausalValidateNode.NAME] is CausalValidateNode
    assert build_state_name_by_node_name()[CausalValidateNode.NAME] == CausalValidateNode.NAME
    assert CausalValidateNode.NAME in build_state_classes_by_name()


def test_causal_validate_node_is_initialized_with_the_other_nodes() -> None:
    nodes = init_all_nodes_with_name_as_key(
        llm=_RaisingLLM(),
        data_repo=object(),
        models_repo=object(),
        analytics_repo=object(),
    )

    assert isinstance(nodes[CausalValidateNode.NAME], CausalValidateNode)


def test_causal_validate_is_a_causal_inference_companion() -> None:
    orch_state = CausalOchestratorState.init_empty()

    companions = orch_state.get_current_node_companion_names(CausalInferenceNode.NAME)

    assert companions == [
        CausalValidateNode.NAME,
        ShapExplanationNode.NAME,
        GeneralQueriesNode.NAME,
    ]


def test_llm_can_route_from_causal_inference_to_enabled_causal_validate() -> None:
    orchestrator = _build_orchestrator_for_routing(_ReturningLLM(CausalValidateNode.NAME))
    orch_state = CausalOchestratorState.init_empty()
    companions = orchestrator._get_current_node_companion_names(
        ochestrator_state=orch_state,
        node_name=CausalInferenceNode.NAME,
    )

    selected = orchestrator._llm_pick_node(
        ochestrator_state=orch_state,
        current_node=CausalInferenceNode.NAME,
        companions=companions,
        history=_user_history(),
    )

    assert selected == CausalValidateNode.NAME


def test_training_rollback_invalidates_cached_causal_validation() -> None:
    orch_state = CausalOchestratorState.init_empty()

    forward_states = orch_state.get_forward_states_after_node(ModelTrainNode.NAME)

    assert CausalValidateNode.NAME in forward_states


def test_shap_node_keeps_follow_up_queries_in_the_causal_workflow() -> None:
    orch_state = CausalOchestratorState.init_empty()

    companions = orch_state.get_current_node_companion_names(ShapExplanationNode.NAME)

    assert companions == [
        CausalInferenceNode.NAME,
        GeneralQueriesNode.NAME,
    ]
