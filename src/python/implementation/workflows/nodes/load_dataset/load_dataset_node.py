from __future__ import annotations

from collections.abc import Mapping
import json
from typing import ClassVar, Optional, Sequence, cast
from uuid import UUID
import uuid

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.load_dataset.load_dataset_prompts import load_dataset_node_info, load_dataset_system_prompt
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetPayloadModel, LoadDatasetState
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetProfilingError, DatasetProfilingTool
from python.implementation.workflows.utils.utils import JSONDict


def _llm_message_strict(llm: LLMService, *, model_name: str, snapshot: JSONDict) -> str:
    cfg = LLMConfig(model="basic", temperature=0.5)
    msg = llm.generate(
        config=cfg,
        system_prompt=load_dataset_system_prompt(),
        user_prompt=json.dumps(snapshot, ensure_ascii=False),
        history=None,
    ).content
    if not msg:
        raise ValueError("LOAD_DATASET: LLM returned empty message")
    return msg


def _format_columns_block(cols: list[str]) -> str:
    lines = [f"Columns ({len(cols)}):"]
    for i, c in enumerate(cols, start=1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


class LoadDatasetNode(Node):
    NAME: ClassVar[str] = LoadDatasetState.NAME
    def __init__(
        self,
        *,
        data_repo: DataRepo,
        llm: LLMService,
        model_name: str = "basic",
    ) -> None:
        self._data_repo = data_repo
        self._llm = llm
        self._model_name = model_name

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return load_dataset_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Optional[Sequence[ChatMessage]],
        state: State,
    ) -> State:
        
        data_profiling_tool = cast(DatasetProfilingTool, tool_factory.get_tool(DatasetProfilingTool.NAME))        
        if not isinstance(state, LoadDatasetState):
            raise TypeError(f"{self.name}: expected LoadDatasetState, got {type(state).__name__}")
        
        if state.payload.id is None:
            return LoadDatasetState(
                payload=LoadDatasetPayloadModel(
                    id=None,
                    summary=None,
                    load_error="dataset_id missing",
                    user_message="Dataset ID is missing. Please re-upload or select a dataset.",
                )
            )

        dataset_id = state.payload.id

        # ---- Load dataframe ----
        try:
            df = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=1_000_000,
            )
        except Exception as e:
            snapshot: JSONDict = {
                "intent": "LOAD_FAILED",
                "error": str(e),
                "hint": "Verify the configured CSV exists and is readable, then try again.",
                "context": {},
            }
            msg = _llm_message_strict(self._llm, model_name=self._model_name, snapshot=snapshot)
            return LoadDatasetState(
                payload=LoadDatasetPayloadModel(
                    id=dataset_id,
                    summary=None,
                    load_error=str(e),
                    user_message=msg,
                )
            )

        # ---- Profile summary (user-actionable failures handled) ----
        try:
            summary = data_profiling_tool.extract_dataset_summary(
                df,
                max_categories=1000,
                sample_distinct=1000,
                compute_quantiles=True,
                strict=True,
            )
            graphs_list = data_profiling_tool.generate_basic_stats_graphs(df=df)
            artifact_ids : Sequence[UUID] = []
            for graph in graphs_list:
                artifact_id = uuid.uuid4()
                graph_bytes = graph.content
                graph_mime = graph.mime
                _ = self._data_repo.save_artifact(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    artifact_id=artifact_id,
                    content=graph_bytes,
                    mime=graph_mime,
                    overwrite=True,
                )
                artifact_ids.append(artifact_id)
                
        except DatasetProfilingError as pe:
            details = getattr(pe, "details", None)
            snapshot: JSONDict = {
                "intent": "SUMMARY_FAILED",
                "error": str(pe),
                "details": (
                    {
                        "column": getattr(details, "column", None),
                        "reason": getattr(details, "reason", None),
                        "hint": getattr(details, "hint", None),
                        "evidence": getattr(details, "evidence", None),
                    }
                    if details is not None
                    else None
                ),
                "hint": "Fix the dataset schema/format and reload.",
                "context": {},
            }
            msg = _llm_message_strict(self._llm, model_name=self._model_name, snapshot=snapshot)
            return LoadDatasetState(
                payload=LoadDatasetPayloadModel(
                    id=state.payload.id,
                    summary=None,
                    load_error=str(pe),
                    user_message=msg,
                )
            )

        # ---- Success message ----
        n_rows, n_cols = df.shape
        cols = [str(c) for c in df.columns.tolist()]

        snapshot_ok: JSONDict = {
            "intent": "LOADED_OK",
            "dataset_preview": {"rows": int(n_rows), "cols": int(n_cols), "columns": cols},
            "summary_stats": {"rows": int(n_rows), "cols": int(n_cols)},
        }
        msg_ok = _llm_message_strict(self._llm, model_name=self._model_name, snapshot=snapshot_ok)

        final_msg = f"{_format_columns_block(cols)}\n\n{msg_ok}".strip()
        
        return LoadDatasetState(
            payload=LoadDatasetPayloadModel(
                id=state.payload.id,
                summary=summary,
                graph_picture_ids=artifact_ids,
                load_error=None,
                user_message=final_msg,
            )
        )