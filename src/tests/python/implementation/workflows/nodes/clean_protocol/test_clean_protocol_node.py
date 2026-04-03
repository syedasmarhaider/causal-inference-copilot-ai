from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pandas as pd

from python.implementation.workflows.nodes.clean_protocol.clean_protocol_node import (
    CleanProtocolNode,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage


@dataclass
class _FakeDataRepo:
    save_calls: list[dict[str, object]] = field(default_factory=list)

    def get_csv_data(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID, limit: int | None = None) -> pd.DataFrame:
        del user_id, conversation_id, dataset_id, limit
        raise NotImplementedError

    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> None:
        del user_id, conversation_id, dataset_id, df, overwrite, include_index
        raise NotImplementedError

    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: str,
        overwrite: bool = True,
    ) -> None:
        self.save_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "artifact_id": artifact_id,
                "content": content,
                "mime": mime,
                "overwrite": overwrite,
            }
        )

    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: str | None = None,
    ) -> bytes:
        del user_id, conversation_id, artifact_id, expected_mime
        raise NotImplementedError

    def get_artifact_mime(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> str:
        del user_id, conversation_id, artifact_id
        raise NotImplementedError


class _FakeCausalDataProfilingTool:
    def generate_causal_missingness_by_group_graph(self, df: pd.DataFrame, protocol: object) -> GraphImage:
        del df, protocol
        return GraphImage(
            key="missingness",
            title="Missingness",
            mime="image/png",
            content=b"missingness",
        )

    def generate_comparability_overlap_histogram(self, df: pd.DataFrame, protocol: object) -> GraphImage:
        del df, protocol
        raise ValueError("One group is empty after treatment parsing. treated=0, control=0")

    def generate_propensity_vs_top_confounders_graphs(self, df: pd.DataFrame, protocol: object) -> list[GraphImage]:
        del df, protocol
        return [
            GraphImage(
                key="propensity",
                title="Propensity",
                mime="image/png",
                content=b"propensity",
            )
        ]


def _build_binary_causal_spec() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "1",
                "control": "0",
            },
            "outcome_spec": {
                "kind": "continuous",
                "column": "outcome",
            },
            "covariates": ["age"],
            "effect_modifiers": [],
            "experiment_type": "OBSERVATIONAL",
        }
    )


def test_generate_final_graphs_keeps_other_graphs_when_overlap_generation_fails() -> None:
    data_repo = _FakeDataRepo()
    node = CleanProtocolNode(data_repo=data_repo, llm=object())
    tool = _FakeCausalDataProfilingTool()
    df = pd.DataFrame(
        [
            {"treatment": "1", "outcome": 10.0, "age": 60},
            {"treatment": "0", "outcome": 8.0, "age": 55},
        ]
    )

    artifact_ids = node._generate_final_graphs(
        user_id=uuid4(),
        conversation_id=uuid4(),
        df=df,
        causal_spec=_build_binary_causal_spec(),
        tool=tool,  # type: ignore[arg-type]
    )

    assert artifact_ids is not None
    assert len(artifact_ids) == 2
    assert len(data_repo.save_calls) == 2
    assert {call["content"] for call in data_repo.save_calls} == {b"missingness", b"propensity"}
