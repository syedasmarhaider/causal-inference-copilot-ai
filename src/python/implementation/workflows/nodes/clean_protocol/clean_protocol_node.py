from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from typing import Any, ClassVar, List, Literal, Optional, Sequence, cast
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_deps import (
    CleanProtocolDeps,
)
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_prompts import (
    CLEAN_PROTOCOL_COMPATIBILITY_FAILURE_PROMPT,
    CLEAN_PROTOCOL_DATA_QUESTION_MESSAGE_PROMPT,
    CLEAN_PROTOCOL_FINAL_ACCEPTANCE_PROMPT,
    CLEAN_PROTOCOL_INITIAL_COMPILE_SPEC_PROMPT,
    CLEAN_PROTOCOL_INTENT_GATE_PROMPT,
    CLEAN_PROTOCOL_ITERATION_MESSAGE_PROMPT,
    CLEAN_PROTOCOL_QUESTION_SQL_PROMPT,
    CLEAN_PROTOCOL_REFRESH_SPEC_PROMPT,
    CLEAN_PROTOCOL_REVERT_MESSAGE_PROMPT,
    CLEAN_PROTOCOL_REVERT_UNAVAILABLE_PROMPT,
    CLEAN_PROTOCOL_SQL_PLAN_PROMPT,
    get_clean_protocol_node_info,
)
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CausalSpecHistoryItemModel,
    CleanDataDiffModel,
    CleanIterationRecordModel,
    CleanProtocolPayloadModel,
    CleanProtocolState,
    SQLHistoryItemModel,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_processing.data_processing_tool import (
    DuckDBInMemorySQLTool,
    SQLStatements,
)
from python.implementation.workflows.tools.data_profiling.causal_data_profiling_tool import (
    CausalDataProfilingTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage


log = logging.getLogger(__name__)


class _IntentGateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal[
        "ANSWER_QUESTION",
        "MODIFY",
        "REVERT",
        "ACCEPT",
        "CHANGE_PROTOCOL_DISCUSSION",
        "ABORT",
    ]
    reason: str
    reply_to_user: str
    revert_target: Optional[Literal["PREVIOUS_STEP", "ORIGINAL_DATASET"]] = None

    @model_validator(mode="after")
    def _validate_revert_target(self) -> "_IntentGateModel":
        if self.action == "REVERT" and self.revert_target is None:
            raise ValueError("revert_target is required when action is REVERT")
        if self.action != "REVERT" and self.revert_target is not None:
            raise ValueError("revert_target must be null when action is not REVERT")
        return self


class _UserMessageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message_for_user: str


@dataclass(frozen=True)
class CleanProtocolNode(Node):
    NAME: ClassVar[str] = CleanProtocolState.NAME

    data_repo: DataRepo
    llm: LLMService

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_clean_protocol_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        if not isinstance(state, CleanProtocolState):
            raise TypeError(
                f"{self.name}: expected CleanProtocolState, got {type(state).__name__}"
            )

        try:
            data_profiling_tool = cast(
                DatasetProfilingTool,
                tool_factory.get_tool(DatasetProfilingTool.NAME),
            )
            causal_data_profiling_tool = cast(
                CausalDataProfilingTool,
                tool_factory.get_tool(CausalDataProfilingTool.NAME),
            )
            data_processing_tool = cast(
                DuckDBInMemorySQLTool,
                tool_factory.get_tool(DuckDBInMemorySQLTool.NAME),
            )

            deps = CleanProtocolDeps.from_loaded(previous_state_dependencies)
            dataset_id = deps.load_dataset.payload.id
            if dataset_id is None:
                return self._abort_state(
                    payload=state.payload,
                    message="Dataset id is missing. Re-run LOAD_DATASET.",
                    reason="LOAD_DATASET.id is missing; cannot load data.",
                )

            protocol_discussion = deps.protocol_discussion.payload.discussion.strip()
            if not protocol_discussion:
                return self._abort_state(
                    payload=state.payload,
                    message=(
                        "Protocol discussion is missing, so I cannot build the causal "
                        "specs for cleaning. Please return to protocol discussion."
                    ),
                    reason="PROTOCOL_DISCUSSION discussion is empty.",
                )

            first_run = _is_first_run(state.payload)
            source_dataset_id = dataset_id if first_run else state.payload.clean_dataset_id
            if source_dataset_id is None:
                return self._abort_state(
                    payload=state.payload,
                    message="No source dataset found for cleaning iteration.",
                    reason="CLEAN_PROTOCOL source dataset id is missing.",
                )

            source_df = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=source_dataset_id,
                limit=None,
            )
            if source_df.empty and int(source_df.shape[1]) == 0:
                return self._abort_state(
                    payload=state.payload,
                    message=(
                        "Current dataset has zero rows and zero columns; cleaning "
                        "cannot continue."
                    ),
                    reason=(
                        "Dataset is empty and has no columns "
                        f"(dataset_id={source_dataset_id})."
                    ),
                )

            recent_history = messages_history[-12:] if messages_history else None

            if first_run:
                source_summary = self._extract_summary(
                    tool=data_profiling_tool,
                    df=source_df,
                )
                try:
                    initial_causal_spec = self._compile_initial_causal_spec(
                        history=recent_history,
                        protocol_discussion=protocol_discussion,
                        dataset_summary=source_summary,
                    )
                except Exception as e:
                    return self._abort_state(
                        payload=state.payload,
                        message=(
                            "I could not translate the current protocol discussion into "
                            "causal specs for cleaning. Please revise the protocol "
                            "discussion and try again."
                        ),
                        reason=(
                            "Failed to compile causal specs from PROTOCOL_DISCUSSION: "
                            f"{e!r}"
                        ),
                    )

                return self._run_modify_iteration(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    source_dataset_id=source_dataset_id,
                    source_df=source_df,
                    protocol_discussion=protocol_discussion,
                    causal_spec=initial_causal_spec,
                    state=state,
                    data_processing_tool=data_processing_tool,
                    data_profiling_tool=data_profiling_tool,
                    history=recent_history,
                    user_request=(
                        "Initial cleaning pass from protocol discussion. Create the "
                        "first cleaned dataset revision."
                    ),
                    mode="INITIAL",
                )

            active_causal_spec = state.payload.compiled_causal_spec
            if active_causal_spec is None:
                source_summary = self._extract_summary(
                    tool=data_profiling_tool,
                    df=source_df,
                )
                try:
                    active_causal_spec = self._compile_initial_causal_spec(
                        history=recent_history,
                        protocol_discussion=protocol_discussion,
                        dataset_summary=source_summary,
                    )
                except Exception as e:
                    return self._abort_state(
                        payload=state.payload,
                        message=(
                            "The current cleaning state is missing causal specs. Please "
                            "return to protocol discussion and try again."
                        ),
                        reason=(
                            "Missing compiled_causal_spec and recompilation failed: "
                            f"{e!r}"
                        ),
                    )

            intent = self._decide_intent(
                state=state,
                history=recent_history,
            )

            if intent.action == "CHANGE_PROTOCOL_DISCUSSION":
                normalized_reason = intent.reason.strip()
                protocol_change_reason = (
                    f"User wants to change protocol discussion: {normalized_reason}"
                    if normalized_reason
                    else "User wants to change protocol discussion."
                )
                return self._abort_state(
                    payload=state.payload,
                    message=intent.reply_to_user,
                    reason=protocol_change_reason,
                )

            if intent.action == "ANSWER_QUESTION":
                user_question = _last_user_text(history=messages_history)
                return self._answer_data_question(
                    state=state,
                    source_df=source_df,
                    protocol_discussion=protocol_discussion,
                    causal_spec=active_causal_spec,
                    data_profiling_tool=data_profiling_tool,
                    data_processing_tool=data_processing_tool,
                    history=recent_history,
                    user_question=user_question or intent.reply_to_user,
                )

            if intent.action == "REVERT":
                return self._revert_cleaning_iteration(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    state=state,
                    load_dataset_id=dataset_id,
                    protocol_discussion=protocol_discussion,
                    data_profiling_tool=data_profiling_tool,
                    history=recent_history,
                    revert_target=cast(
                        Literal["PREVIOUS_STEP", "ORIGINAL_DATASET"],
                        intent.revert_target,
                    ),
                )

            if intent.action == "ABORT":
                return self._abort_state(
                    payload=state.payload,
                    message=intent.reply_to_user,
                    reason=intent.reason,
                )

            if intent.action == "ACCEPT":
                compat_err = _minimal_compatibility_error(source_df, active_causal_spec)
                if compat_err is not None:
                    msg = self._render_compatibility_failure_message(
                        history=recent_history,
                        compatibility_error=compat_err,
                        payload=state.payload,
                    )
                    return self._pending_state(
                        payload=state.payload,
                        message=msg.message_for_user,
                    )

                artifacts = self._generate_final_graphs(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    df=source_df,
                    causal_spec=active_causal_spec,
                    tool=causal_data_profiling_tool,
                )

                final_message = self._render_final_acceptance_message(
                    history=recent_history,
                    payload=state.payload,
                    graph_count=len(artifacts or []),
                )
                return CleanProtocolState(
                    payload=state.payload.model_copy(
                        update={
                            "cleaning_error": None,
                            "user_acceptance": True,
                            "graph_picture_ids": artifacts,
                            "user_message": final_message.message_for_user,
                        }
                    )
                )

            user_request = _last_user_text(history=messages_history)
            if not user_request:
                user_request = intent.reply_to_user
            if not user_request:
                user_request = "Apply another cleaning revision."

            return self._run_modify_iteration(
                user_id=user_id,
                conversation_id=conversation_id,
                source_dataset_id=source_dataset_id,
                source_df=source_df,
                protocol_discussion=protocol_discussion,
                causal_spec=active_causal_spec,
                state=state,
                data_processing_tool=data_processing_tool,
                data_profiling_tool=data_profiling_tool,
                history=recent_history,
                user_request=user_request,
                mode="MODIFY",
            )

        except Exception as e:
            log.exception("CleanProtocolNode failed unexpectedly")
            return self._abort_state(
                payload=state.payload,
                message=f"Clean protocol failed unexpectedly: {e!r}",
                reason=f"Clean protocol failed unexpectedly: {e!r}",
            )

    def _run_modify_iteration(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        source_dataset_id: UUID,
        source_df: pd.DataFrame,
        protocol_discussion: str,
        causal_spec: CausalSpec,
        state: CleanProtocolState,
        data_processing_tool: DuckDBInMemorySQLTool,
        data_profiling_tool: DatasetProfilingTool,
        history: Optional[Sequence[ChatMessage]],
        user_request: str,
        mode: Literal["INITIAL", "MODIFY"],
    ) -> CleanProtocolState:
        source_summary = self._extract_summary(
            tool=data_profiling_tool,
            df=source_df,
        )

        sql_request = self._generate_sql_request(
            mode=mode,
            user_request=user_request,
            protocol_discussion=protocol_discussion,
            causal_spec=causal_spec,
            source_summary=source_summary,
            state=state.payload,
            history=history,
        )

        try:
            sql_result = data_processing_tool.execute(
                dataframe=source_df,
                sql_request=sql_request,
            )
        except Exception as e:
            msg = self._render_compatibility_failure_message(
                history=history,
                compatibility_error=f"SQL execution failed: {e!r}",
                payload=state.payload,
            )
            return self._pending_state(
                payload=state.payload,
                message=msg.message_for_user,
            )

        if not sql_result.has_result_set:
            msg = self._render_compatibility_failure_message(
                history=history,
                compatibility_error=(
                    "The generated SQL did not return a final result set. Please "
                    "request another cleaning modification."
                ),
                payload=state.payload,
            )
            return self._pending_state(
                payload=state.payload,
                message=msg.message_for_user,
            )

        cleaned_df_candidate = sql_result.dataframe
        cleaned_summary_candidate = self._extract_summary(
            tool=data_profiling_tool,
            df=cleaned_df_candidate,
        )

        try:
            refreshed_causal_spec = self._refresh_causal_spec(
                history=history,
                protocol_discussion=protocol_discussion,
                cleaned_summary=cleaned_summary_candidate,
                previous_causal_spec=causal_spec,
                user_request=user_request,
                sql_request=sql_request,
            )
        except Exception as e:
            msg = self._render_compatibility_failure_message(
                history=history,
                compatibility_error=(
                    "Failed to refresh the causal spec from the cleaned dataset: "
                    f"{e!r}. Please request another cleaning revision."
                ),
                payload=state.payload,
            )
            return self._pending_state(
                payload=state.payload,
                message=msg.message_for_user,
            )

        required_modeling_columns = _required_modeling_columns(refreshed_causal_spec)
        if not required_modeling_columns:
            msg = self._render_compatibility_failure_message(
                history=history,
                compatibility_error=(
                    "Refreshed causal spec did not contain modeling columns."
                ),
                payload=state.payload,
            )
            return self._pending_state(
                payload=state.payload,
                message=msg.message_for_user,
            )

        missing_required_columns = [
            col
            for col in required_modeling_columns
            if col not in cleaned_df_candidate.columns
        ]
        if missing_required_columns:
            msg = self._render_compatibility_failure_message(
                history=history,
                compatibility_error=(
                    "The SQL result is missing columns required by the refreshed "
                    f"causal spec: {missing_required_columns}. Required columns are: "
                    f"{required_modeling_columns}."
                ),
                payload=state.payload,
            )
            return self._pending_state(
                payload=state.payload,
                message=msg.message_for_user,
            )

        cleaned_df = cleaned_df_candidate.loc[:, required_modeling_columns].copy()

        clean_dataset_id = uuid4()
        self.data_repo.save_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=clean_dataset_id,
            df=cleaned_df,
            overwrite=True,
            include_index=False,
        )

        cleaned_summary = self._extract_summary(
            tool=data_profiling_tool,
            df=cleaned_df,
        )

        next_iteration = int(state.payload.iteration_index) + 1
        diff = _build_diff(before_df=source_df, after_df=cleaned_df)

        sql_history_item = SQLHistoryItemModel(
            iteration_index=next_iteration,
            source_dataset_id=source_dataset_id,
            output_dataset_id=clean_dataset_id,
            sql_request=sql_request,
        )
        iteration_record = CleanIterationRecordModel(
            iteration_index=next_iteration,
            source_dataset_id=source_dataset_id,
            output_dataset_id=clean_dataset_id,
            diff=diff,
            summary=cleaned_summary,
        )
        spec_history_item = CausalSpecHistoryItemModel(
            iteration_index=next_iteration,
            causal_spec=refreshed_causal_spec,
        )

        user_message = self._render_iteration_user_message(
            history=history,
            source_dataset_id=source_dataset_id,
            output_dataset_id=clean_dataset_id,
            diff=diff,
            sql_request=sql_request,
            source_summary=source_summary,
            cleaned_summary=cleaned_summary,
            updated_causal_spec=refreshed_causal_spec,
            iteration_index=next_iteration,
        )

        return CleanProtocolState(
            payload=state.payload.model_copy(
                update={
                    "clean_dataset_id": clean_dataset_id,
                    "cleaning_error": None,
                    "user_message": user_message.message_for_user,
                    "summary": cleaned_summary,
                    "user_acceptance": None,
                    "graph_picture_ids": None,
                    "iteration_index": next_iteration,
                    "latest_diff": diff,
                    "compiled_causal_spec": refreshed_causal_spec,
                    "sql_history": [*state.payload.sql_history, sql_history_item],
                    "iteration_history": [
                        *state.payload.iteration_history,
                        iteration_record,
                    ],
                    "causal_spec_history": [
                        *state.payload.causal_spec_history,
                        spec_history_item,
                    ],
                }
            )
        )

    def _decide_intent(
        self,
        *,
        state: CleanProtocolState,
        history: Optional[Sequence[ChatMessage]],
    ) -> _IntentGateModel:
        latest_sql = (
            state.payload.sql_history[-1].sql_request
            if state.payload.sql_history
            else None
        )
        latest_diff = state.payload.latest_diff
        payload = {
            "iteration_index": state.payload.iteration_index,
            "latest_user_message": _last_user_text(history=history),
            "current_dataset_summary": (
                state.payload.summary.model_dump(mode="json")
                if state.payload.summary is not None
                else None
            ),
            "latest_sql_request": (
                latest_sql.model_dump(mode="json") if latest_sql is not None else None
            ),
            "latest_diff": (
                latest_diff.model_dump(mode="json") if latest_diff is not None else None
            ),
            "current_causal_spec": (
                state.payload.compiled_causal_spec.model_dump(mode="json")
                if state.payload.compiled_causal_spec is not None
                else None
            ),
            "available_revert_targets": {
                "can_revert_previous_step": bool(state.payload.iteration_index >= 1),
                "can_revert_original_dataset": True,
            },
        }

        return self.llm.generate_json(
            schema=_IntentGateModel,
            system_prompt=CLEAN_PROTOCOL_INTENT_GATE_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.2),
            history=history,
            max_attempts=2,
        )

    def _compile_initial_causal_spec(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        protocol_discussion: str,
        dataset_summary: Any,
    ) -> CausalSpec:
        payload = {
            "protocol_discussion": protocol_discussion,
            "current_dataset_summary": dataset_summary.model_dump(mode="json"),
        }
        return self.llm.generate_json(
            schema=CausalSpec,
            system_prompt=CLEAN_PROTOCOL_INITIAL_COMPILE_SPEC_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.0),
            history=history,
            max_attempts=3,
        )

    def _answer_data_question(
        self,
        *,
        state: CleanProtocolState,
        source_df: pd.DataFrame,
        protocol_discussion: str,
        causal_spec: CausalSpec,
        data_profiling_tool: DatasetProfilingTool,
        data_processing_tool: DuckDBInMemorySQLTool,
        history: Optional[Sequence[ChatMessage]],
        user_question: str,
    ) -> CleanProtocolState:
        source_summary = self._extract_summary(
            tool=data_profiling_tool,
            df=source_df,
        )
        question_sql = self._generate_question_sql(
            history=history,
            user_question=user_question,
            protocol_discussion=protocol_discussion,
            causal_spec=causal_spec,
            source_summary=source_summary,
            state=state.payload,
        )

        question_error: Optional[str] = None
        result_preview: dict[str, Any]
        try:
            question_result = data_processing_tool.execute(
                dataframe=source_df,
                sql_request=question_sql,
            )
            if not question_result.has_result_set:
                question_error = "The analytic SQL did not return a result set."
                result_preview = {
                    "row_count": 0,
                    "columns": [],
                    "rows": [],
                }
            else:
                result_preview = _result_preview(question_result.dataframe)
        except Exception as e:
            question_error = f"Failed to run analytic SQL for the question: {e!r}"
            result_preview = {
                "row_count": 0,
                "columns": [],
                "rows": [],
            }

        message = self._render_data_question_message(
            history=history,
            payload=state.payload,
            user_question=user_question,
            source_summary=source_summary,
            causal_spec=causal_spec,
            sql_request=question_sql,
            result_preview=result_preview,
            question_error=question_error,
        )
        return self._pending_state(
            payload=state.payload,
            message=message.message_for_user,
        )

    def _revert_cleaning_iteration(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: CleanProtocolState,
        load_dataset_id: UUID,
        protocol_discussion: str,
        data_profiling_tool: DatasetProfilingTool,
        history: Optional[Sequence[ChatMessage]],
        revert_target: Literal["PREVIOUS_STEP", "ORIGINAL_DATASET"],
    ) -> CleanProtocolState:
        if (
            revert_target == "ORIGINAL_DATASET"
            or int(state.payload.iteration_index) <= 1
            or len(state.payload.iteration_history) <= 1
        ):
            raw_df = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=load_dataset_id,
                limit=None,
            )
            raw_summary = self._extract_summary(
                tool=data_profiling_tool,
                df=raw_df,
            )
            try:
                raw_spec = self._compile_initial_causal_spec(
                    history=history,
                    protocol_discussion=protocol_discussion,
                    dataset_summary=raw_summary,
                )
            except Exception as e:
                return self._abort_state(
                    payload=state.payload,
                    message=(
                        "I could not restore the original dataset because rebuilding "
                        "the causal spec failed. Please return to protocol discussion."
                    ),
                    reason=(
                        "Failed to recompile causal spec while reverting to original "
                        f"dataset: {e!r}"
                    ),
                )

            message = self._render_revert_message(
                history=history,
                payload=state.payload,
                revert_target="ORIGINAL_DATASET",
                restored_iteration_index=0,
                restored_summary=raw_summary,
                restored_causal_spec=raw_spec,
                restored_latest_diff=None,
            )
            return CleanProtocolState(
                payload=state.payload.model_copy(
                    update={
                        "clean_dataset_id": load_dataset_id,
                        "cleaning_error": None,
                        "user_message": message.message_for_user,
                        "summary": raw_summary,
                        "user_acceptance": None,
                        "graph_picture_ids": None,
                        "iteration_index": 0,
                        "latest_diff": None,
                        "compiled_causal_spec": raw_spec,
                        "sql_history": [],
                        "iteration_history": [],
                        "causal_spec_history": [],
                    }
                )
            )

        if not state.payload.iteration_history or not state.payload.causal_spec_history:
            message = self._render_revert_unavailable_message(
                history=history,
                payload=state.payload,
            )
            return self._pending_state(
                payload=state.payload,
                message=message.message_for_user,
            )

        restored_iteration_history = state.payload.iteration_history[:-1]
        restored_sql_history = state.payload.sql_history[:-1]
        restored_spec_history = state.payload.causal_spec_history[:-1]

        if not restored_iteration_history or not restored_spec_history:
            message = self._render_revert_unavailable_message(
                history=history,
                payload=state.payload,
            )
            return self._pending_state(
                payload=state.payload,
                message=message.message_for_user,
            )

        restored_iteration = restored_iteration_history[-1]
        restored_spec = restored_spec_history[-1].causal_spec
        message = self._render_revert_message(
            history=history,
            payload=state.payload,
            revert_target="PREVIOUS_STEP",
            restored_iteration_index=int(restored_iteration.iteration_index),
            restored_summary=restored_iteration.summary,
            restored_causal_spec=restored_spec,
            restored_latest_diff=restored_iteration.diff,
        )

        return CleanProtocolState(
            payload=state.payload.model_copy(
                update={
                    "clean_dataset_id": restored_iteration.output_dataset_id,
                    "cleaning_error": None,
                    "user_message": message.message_for_user,
                    "summary": restored_iteration.summary,
                    "user_acceptance": None,
                    "graph_picture_ids": None,
                    "iteration_index": int(restored_iteration.iteration_index),
                    "latest_diff": restored_iteration.diff,
                    "compiled_causal_spec": restored_spec,
                    "sql_history": restored_sql_history,
                    "iteration_history": restored_iteration_history,
                    "causal_spec_history": restored_spec_history,
                }
            )
        )

    def _generate_sql_request(
        self,
        *,
        mode: Literal["INITIAL", "MODIFY"],
        user_request: str,
        protocol_discussion: str,
        causal_spec: CausalSpec,
        source_summary: Any,
        state: CleanProtocolPayloadModel,
        history: Optional[Sequence[ChatMessage]],
    ) -> SQLStatements:
        payload = {
            "mode": mode,
            "table_name": "cohort_df",
            "user_request": user_request,
            "protocol_discussion": protocol_discussion,
            "current_causal_spec": causal_spec.model_dump(mode="json"),
            "time_zero_policy": (
                "Use time-zero columns only for filtering or derived transformations. "
                "Do not keep raw time-zero helper columns in the final modeling output "
                "unless they are themselves modeling columns."
            ),
            "final_output_contract": (
                "The final result set must contain only treatment, outcome, "
                "covariates, and effect modifiers."
            ),
            "current_dataset_summary": source_summary.model_dump(mode="json"),
            "iteration_index": state.iteration_index,
            "latest_diff": (
                state.latest_diff.model_dump(mode="json")
                if state.latest_diff is not None
                else None
            ),
            "previous_sql_history": [
                item.sql_request.model_dump(mode="json")
                for item in state.sql_history[-5:]
            ],
        }

        return self.llm.generate_json(
            schema=SQLStatements,
            system_prompt=CLEAN_PROTOCOL_SQL_PLAN_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.1),
            history=history,
            max_attempts=3,
        )

    def _generate_question_sql(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        user_question: str,
        protocol_discussion: str,
        causal_spec: CausalSpec,
        source_summary: Any,
        state: CleanProtocolPayloadModel,
    ) -> SQLStatements:
        payload = {
            "table_name": "cohort_df",
            "user_question": user_question,
            "protocol_discussion": protocol_discussion,
            "current_causal_spec": causal_spec.model_dump(mode="json"),
            "current_dataset_summary": source_summary.model_dump(mode="json"),
            "iteration_index": state.iteration_index,
            "latest_diff": (
                state.latest_diff.model_dump(mode="json")
                if state.latest_diff is not None
                else None
            ),
            "previous_sql_history": [
                item.sql_request.model_dump(mode="json")
                for item in state.sql_history[-5:]
            ],
        }
        return self.llm.generate_json(
            schema=SQLStatements,
            system_prompt=CLEAN_PROTOCOL_QUESTION_SQL_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.0),
            history=history,
            max_attempts=3,
        )

    def _refresh_causal_spec(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        protocol_discussion: str,
        cleaned_summary: Any,
        previous_causal_spec: CausalSpec,
        user_request: str,
        sql_request: SQLStatements,
    ) -> CausalSpec:
        payload = {
            "protocol_discussion": protocol_discussion,
            "latest_user_request": user_request,
            "latest_sql_request": sql_request.model_dump(mode="json"),
            "previous_causal_spec": previous_causal_spec.model_dump(mode="json"),
            "current_dataset_summary": cleaned_summary.model_dump(mode="json"),
        }
        return self.llm.generate_json(
            schema=CausalSpec,
            system_prompt=CLEAN_PROTOCOL_REFRESH_SPEC_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.0),
            history=history,
            max_attempts=2,
        )

    def _render_data_question_message(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        payload: CleanProtocolPayloadModel,
        user_question: str,
        source_summary: Any,
        causal_spec: CausalSpec,
        sql_request: SQLStatements,
        result_preview: dict[str, Any],
        question_error: Optional[str],
    ) -> _UserMessageModel:
        prompt_payload = {
            "iteration_index": payload.iteration_index,
            "user_question": user_question,
            "question_error": question_error,
            "current_dataset_summary": source_summary.model_dump(mode="json"),
            "current_causal_spec": causal_spec.model_dump(mode="json"),
            "analytic_sql_request": sql_request.model_dump(mode="json"),
            "analytic_result_preview": result_preview,
            "latest_diff": (
                payload.latest_diff.model_dump(mode="json")
                if payload.latest_diff is not None
                else None
            ),
        }
        return self.llm.generate_json(
            schema=_UserMessageModel,
            system_prompt=CLEAN_PROTOCOL_DATA_QUESTION_MESSAGE_PROMPT,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.3),
            history=history,
            max_attempts=2,
        )

    def _render_iteration_user_message(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        source_dataset_id: UUID,
        output_dataset_id: UUID,
        diff: CleanDataDiffModel,
        sql_request: SQLStatements,
        source_summary: Any,
        cleaned_summary: Any,
        updated_causal_spec: CausalSpec,
        iteration_index: int,
    ) -> _UserMessageModel:
        payload = {
            "iteration_index": iteration_index,
            "source_dataset_id": str(source_dataset_id),
            "output_dataset_id": str(output_dataset_id),
            "diff": diff.model_dump(mode="json"),
            "sql_request": sql_request.model_dump(mode="json"),
            "source_summary": source_summary.model_dump(mode="json"),
            "cleaned_summary": cleaned_summary.model_dump(mode="json"),
            "updated_causal_spec": updated_causal_spec.model_dump(mode="json"),
        }
        return self.llm.generate_json(
            schema=_UserMessageModel,
            system_prompt=CLEAN_PROTOCOL_ITERATION_MESSAGE_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.6),
            history=history,
            max_attempts=2,
        )

    def _render_revert_message(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        payload: CleanProtocolPayloadModel,
        revert_target: Literal["PREVIOUS_STEP", "ORIGINAL_DATASET"],
        restored_iteration_index: int,
        restored_summary: Any,
        restored_causal_spec: CausalSpec,
        restored_latest_diff: Optional[CleanDataDiffModel],
    ) -> _UserMessageModel:
        prompt_payload = {
            "current_iteration_index": payload.iteration_index,
            "revert_target": revert_target,
            "restored_iteration_index": restored_iteration_index,
            "restored_summary": restored_summary.model_dump(mode="json"),
            "restored_causal_spec": restored_causal_spec.model_dump(mode="json"),
            "restored_latest_diff": (
                restored_latest_diff.model_dump(mode="json")
                if restored_latest_diff is not None
                else None
            ),
        }
        return self.llm.generate_json(
            schema=_UserMessageModel,
            system_prompt=CLEAN_PROTOCOL_REVERT_MESSAGE_PROMPT,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.3),
            history=history,
            max_attempts=2,
        )

    def _render_revert_unavailable_message(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        payload: CleanProtocolPayloadModel,
    ) -> _UserMessageModel:
        prompt_payload = {
            "current_iteration_index": payload.iteration_index,
            "has_clean_dataset_id": payload.clean_dataset_id is not None,
            "has_history": bool(payload.iteration_history),
        }
        return self.llm.generate_json(
            schema=_UserMessageModel,
            system_prompt=CLEAN_PROTOCOL_REVERT_UNAVAILABLE_PROMPT,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.3),
            history=history,
            max_attempts=2,
        )

    def _render_compatibility_failure_message(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        compatibility_error: str,
        payload: CleanProtocolPayloadModel,
    ) -> _UserMessageModel:
        prompt_payload = {
            "compatibility_error": compatibility_error,
            "iteration_index": payload.iteration_index,
            "latest_diff": (
                payload.latest_diff.model_dump(mode="json")
                if payload.latest_diff is not None
                else None
            ),
            "latest_sql": (
                payload.sql_history[-1].sql_request.model_dump(mode="json")
                if payload.sql_history
                else None
            ),
        }
        return self.llm.generate_json(
            schema=_UserMessageModel,
            system_prompt=CLEAN_PROTOCOL_COMPATIBILITY_FAILURE_PROMPT,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.4),
            history=history,
            max_attempts=2,
        )

    def _render_final_acceptance_message(
        self,
        *,
        history: Optional[Sequence[ChatMessage]],
        payload: CleanProtocolPayloadModel,
        graph_count: int,
    ) -> _UserMessageModel:
        prompt_payload = {
            "iteration_index": payload.iteration_index,
            "clean_dataset_id": (
                str(payload.clean_dataset_id)
                if payload.clean_dataset_id is not None
                else None
            ),
            "latest_diff": (
                payload.latest_diff.model_dump(mode="json")
                if payload.latest_diff is not None
                else None
            ),
            "graph_count": int(graph_count),
        }
        return self.llm.generate_json(
            schema=_UserMessageModel,
            system_prompt=CLEAN_PROTOCOL_FINAL_ACCEPTANCE_PROMPT,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.3),
            history=history,
            max_attempts=2,
        )

    def _generate_final_graphs(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        df: pd.DataFrame,
        causal_spec: CausalSpec,
        tool: CausalDataProfilingTool,
    ) -> Optional[Sequence[UUID]]:
        if (
            causal_spec.treatment_spec.kind != "binary"
            or (len(causal_spec.covariates) + len(causal_spec.effect_modifiers)) == 0
        ):
            return None

        try:
            graphs: List[GraphImage] = [
                tool.generate_causal_missingness_by_group_graph(
                    df=df,
                    protocol=causal_spec,
                ),
                tool.generate_comparability_overlap_histogram(
                    df=df,
                    protocol=causal_spec,
                ),
            ]
            graphs.extend(
                tool.generate_propensity_vs_top_confounders_graphs(
                    df=df,
                    protocol=causal_spec,
                )
            )

            artifacts: List[UUID] = []
            for graph in graphs:
                artifact_id = uuid4()
                self.data_repo.save_artifact(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    artifact_id=artifact_id,
                    content=graph.content,
                    mime=graph.mime,
                    overwrite=True,
                )
                artifacts.append(artifact_id)
            return artifacts
        except Exception:
            log.exception("CleanProtocolNode: final graph generation failed")
            return None

    @staticmethod
    def _extract_summary(
        *,
        tool: DatasetProfilingTool,
        df: pd.DataFrame,
    ) -> Any:
        return tool.extract_dataset_summary(
            df,
            max_categories=1000,
            sample_distinct=1000,
            compute_quantiles=True,
            strict=False,
        )

    @staticmethod
    def _pending_state(
        *,
        payload: CleanProtocolPayloadModel,
        message: str,
    ) -> CleanProtocolState:
        return CleanProtocolState(
            payload=payload.model_copy(
                update={
                    "cleaning_error": None,
                    "user_acceptance": None,
                    "graph_picture_ids": None,
                    "user_message": message,
                }
            )
        )

    @staticmethod
    def _abort_state(
        *,
        payload: CleanProtocolPayloadModel,
        message: str,
        reason: str,
    ) -> CleanProtocolState:
        return CleanProtocolState(
            payload=payload.model_copy(
                update={
                    "cleaning_error": reason,
                    "user_acceptance": False,
                    "graph_picture_ids": None,
                    "user_message": message,
                }
            )
        )


def _build_diff(*, before_df: pd.DataFrame, after_df: pd.DataFrame) -> CleanDataDiffModel:
    before_cols = [str(c) for c in before_df.columns]
    after_cols = [str(c) for c in after_df.columns]
    before_set = set(before_cols)
    after_set = set(after_cols)

    rows_before = int(before_df.shape[0])
    rows_after = int(after_df.shape[0])
    cols_before = int(before_df.shape[1])
    cols_after = int(after_df.shape[1])

    return CleanDataDiffModel(
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        rows_delta=rows_after - rows_before,
        cols_delta=cols_after - cols_before,
        added_columns=sorted([c for c in after_cols if c not in before_set]),
        removed_columns=sorted([c for c in before_cols if c not in after_set]),
    )


def _is_first_run(payload: CleanProtocolPayloadModel) -> bool:
    if payload.clean_dataset_id is None:
        return True
    if payload.summary is None:
        return True
    if payload.compiled_causal_spec is None:
        return True
    return False


def _result_preview(df: pd.DataFrame, *, max_rows: int = 25, max_cols: int = 12) -> dict[str, Any]:
    preview_df = df.iloc[:max_rows, :max_cols].copy()
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [str(col) for col in preview_df.columns],
        "rows": json.loads(preview_df.to_json(orient="records", date_format="iso")),
        "truncated_rows": bool(df.shape[0] > max_rows),
        "truncated_columns": bool(df.shape[1] > max_cols),
    }


def _last_user_text(history: Optional[Sequence[ChatMessage]]) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _minimal_compatibility_error(df: pd.DataFrame, causal_spec: CausalSpec) -> Optional[str]:
    if int(df.shape[0]) <= 0:
        return "Cleaned dataset has zero rows."
    if int(df.shape[1]) <= 0:
        return "Cleaned dataset has zero columns."

    required_columns = _required_modeling_columns(causal_spec)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return f"Cleaned dataset is missing required modeling columns: {missing}"

    required_set = set(required_columns)
    extra_columns = [str(c) for c in df.columns if str(c) not in required_set]
    if extra_columns:
        return (
            "Cleaned dataset contains non-modeling columns: "
            f"{extra_columns}. Allowed columns are: {required_columns}"
        )
    return None


def _required_modeling_columns(causal_spec: CausalSpec) -> List[str]:
    columns: List[str] = [
        str(causal_spec.treatment_spec.column),
        str(causal_spec.outcome_spec.column),
    ]
    columns.extend(str(c) for c in causal_spec.covariates)
    columns.extend(str(c) for c in causal_spec.effect_modifiers)

    required: List[str] = []
    seen: set[str] = set()
    for column in columns:
        normalized = column.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        required.append(normalized)
    return required
