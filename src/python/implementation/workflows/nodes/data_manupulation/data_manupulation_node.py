from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from collections.abc import Sequence
from typing import Any, ClassVar, cast
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from python.domain.models.models import ChatMessage, get_chat_messages_role_and_message_json
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.domain.workflows.node import Action, Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_deps import (
    DataManupulationDeps,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_prompts import (
    data_manupulation_final_response_system_prompt,
    data_manupulation_intent_classification_system_prompt,
    data_manupulation_node_info,
    data_manupulation_out_of_scope_system_prompt,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.utils.utils import JSONDict, safe_err

log = get_app_logger(__name__, component="data_manupulation_node", log_type="node")

_DATA_MANIPULATION_RETRY_ATTEMPTS = 3
_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16
_INITIAL_SUMMARY_MAX_COLUMNS = 8
_REVERT_DATA_CHANGES_MESSAGE = "revert_data_changes"
_FROZEN_DATASET_REVERT_MESSAGE = (
    "The dataset is frozen, so a revert request is not possible here. "
    "Revert to a previous workflow state first to do that."
)
_OFF_TOPIC_FALLBACK_MESSAGE = (
    "This data manipulation stage is only for dataset-changing operations. I can clean, "
    "filter, reshape, rename, recode, derive, or otherwise update the working dataset. "
    "For read-only questions, statistics, or charts, use the appropriate stage."
)


class DataManupulationIntentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent_dataset_mutation: bool = False
    intent_dataset_mutation_brief: str = ""
    intent_out_of_scope: bool = False

    @field_validator("intent_dataset_mutation_brief", mode="before")
    @classmethod
    def _normalize_brief(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @model_validator(mode="after")
    def _validate(self) -> DataManupulationIntentModel:
        if self.intent_out_of_scope and self.intent_dataset_mutation:
            raise ValueError(
                "intent_out_of_scope cannot be combined with intent_dataset_mutation"
            )
        if self.intent_dataset_mutation and not self.intent_dataset_mutation_brief:
            raise ValueError("intent_dataset_mutation_brief is required")
        return self

    def has_any_intent(self) -> bool:
        return self.intent_dataset_mutation


class DataManupulationNode(Node):
    NAME: ClassVar[str] = DataManupulationState.NAME

    def __init__(
        self,
        *,
        data_repo: DataRepo,
        llm: LLMService,
        tools_factory: ToolFactory,
    ) -> None:
        self._data_repo = data_repo
        self._llm = llm
        self._data_manipulation_tool = cast(
            DataManipulationTool, tools_factory.get_tool(DataManipulationTool.NAME)
        )
        self._profiling_tool = cast(
            DatasetProfilingTool, tools_factory.get_tool(DatasetProfilingTool.NAME)
        )

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return data_manupulation_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, DataManupulationState):
            raise TypeError(
                f"{self.name}: expected DataManupulationState, got "
                f"{type(request.node_state).__name__}"
            )

        latest_user_message = _latest_user_message(request.read_only_messages_history)
        if self._is_revert_request(latest_user_message):
            return self._handle_revert_request(request=request)

        deps = DataManupulationDeps.from_request(request)

        try:
            current_df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=1_000_000,
            )
        except Exception as exc:
            log.exception(
                "failed to load data manupulation dataset",
                dataset_id=str(deps.dataset_id),
                error=safe_err(exc),
            )
            return self._needs_data_result(
                request=request,
                user_message=(
                    "Upload dataset csv"
                ),
            )

        current_summary = deps.dataset_summary
        if current_summary is None:
            current_summary = self._profiling_tool.extract_dataset_summary(
                current_df,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )
            
            ochestrator_state = request.orchestrator_state
            ochestrator_state.set(
                request.node_state.name(),
                {
                    "working_dataset_id": deps.dataset_id,
                    "latest_dataset_summary": current_summary,
                },
            )

        current_summary_json = self._profiling_tool.dataset_summary_to_json(current_summary)

        if not latest_user_message:
            return self._needs_input_result(
                request=request,
                user_message=self._build_ready_message(summary=current_summary),
            )

        history_text = _recent_history_text(request.read_only_messages_history)
        try:
            intent = self._classify_intent(
                latest_user_message=latest_user_message,
                chat_history=history_text,
                dataset_summary=current_summary_json,
            )
        except Exception as exc:
            log.exception("failed to classify data manupulation intent", error=safe_err(exc))
            return self._needs_input_result(
                request=request,
                user_message=(
                    "I could not classify that manipulation request. Please ask again more "
                    "directly."
                ),
            )

        if intent.intent_out_of_scope or not intent.has_any_intent():
            try:
                out_of_scope_message = self._build_out_of_scope_message(
                    user_message=latest_user_message,
                    dataset_summary_json=current_summary_json,
                )
            except Exception as exc:
                log.exception("failed to build out-of-scope message", error=safe_err(exc))
                out_of_scope_message = _OFF_TOPIC_FALLBACK_MESSAGE
            return self._needs_input_result(
                request=request,
                user_message=out_of_scope_message,
            )

        try:
            manipulation_result = self._run_manupulation(
                request=request,
                dataframe=current_df,
                current_summary_json=current_summary_json,
                instructions=intent.intent_dataset_mutation_brief or latest_user_message,
            )
        except Exception as exc:
            log.exception("failed to run data manupulation", error=safe_err(exc))
            return self._needs_input_result(
                request=request,
                user_message=(
                    "I could not apply that dataset change. Please rephrase the requested "
                    "transformation."
                ),
            )

        try:
            final_message = self._build_final_message(
                manipulation_result=manipulation_result,
                dataset_context={
                    "original_user_message": latest_user_message,
                    "active_dataset_rows": manipulation_result["row_count"],
                    "active_dataset_columns": manipulation_result["columns"],
                },
            )
        except Exception as exc:
            log.exception("failed to build final data manupulation response", error=safe_err(exc))
            final_message = self._build_final_message_fallback(
                manipulation_result=manipulation_result
            )

        return self._needs_input_result(
            request=request,
            user_message=final_message,
            action=(
                "NONE"
                if self._is_protocol_discussion_complete_and_data_cleaning_pending(request)
                else "NEEDS_INPUT"
            ),
        )

    def _is_revert_request(self, latest_user_message: str | None) -> bool:
        if latest_user_message is None:
            return False
        return latest_user_message.strip() == _REVERT_DATA_CHANGES_MESSAGE

    def _handle_revert_request(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if request.orchestrator_state.get("working_dataset_frozen") is True:
            return self._needs_input_result(
                request=request,
                user_message=_FROZEN_DATASET_REVERT_MESSAGE,
            )

        working_dataset_ids_raw: Any = (
            request.orchestrator_state.get("working_dataset_ids") or []
        )
        working_dataset_ids = [
            dataset_id if isinstance(dataset_id, UUID) else UUID(str(dataset_id))
            for dataset_id in cast(list[Any], working_dataset_ids_raw)
        ]
        if len(working_dataset_ids) < 2:
            return self._needs_input_result(
                request=request,
                user_message="There is no previous dataset version to revert to.",
            )

        revert_to_dataset_id = working_dataset_ids[-2]
        try:
            reverted_df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=revert_to_dataset_id,
                limit=1_000_000,
            )
        except Exception as exc:
            log.exception(
                "failed to load previous data manipulation dataset",
                dataset_id=str(revert_to_dataset_id),
                error=safe_err(exc),
            )
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the previous working dataset version. "
                    "Please try again or restore the dataset manually."
                ),
            )

        reverted_summary = self._profiling_tool.extract_dataset_summary(
            reverted_df,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        request.orchestrator_state.set(
            request.node_state.name(),
            {
                "working_dataset_id": revert_to_dataset_id,
                "latest_dataset_summary": reverted_summary,
                "revert_request": True,
            },
        )

        reverted_profiles = list(reverted_summary.profiles)
        reverted_message = (
            "Reverted the working dataset to the previous version. "
            f"The restored dataset has {reverted_summary.n_rows} rows and "
            f"{len(reverted_profiles)} columns."
        )
        return self._needs_input_result(
            request=request,
            user_message=reverted_message,
            action=(
                "NONE"
                if self._is_protocol_discussion_complete_and_data_cleaning_pending(request)
                else "NEEDS_INPUT"
            ),
        )

    def _classify_intent(
        self,
        *,
        latest_user_message: str,
        chat_history: str | None,
        dataset_summary: str,
    ) -> DataManupulationIntentModel:
        payload: JSONDict = {
            "latest_user_message": latest_user_message,
            "chat_history": chat_history,
            "dataset_summary": dataset_summary,
        }
        return self._llm.generate_json(
            schema=DataManupulationIntentModel,
            system_prompt=data_manupulation_intent_classification_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.0, top_p=1.0),
            history=None,
            max_attempts=2,
        )

    def _run_manupulation(
        self,
        *,
        request: NodeRequest,
        dataframe: pd.DataFrame,
        current_summary_json: str,
        instructions: str,
    ) -> JSONDict:
        result_df = self._run_data_manipulation_tool(
            dataframe=dataframe,
            conversation_id=request.conversation_id,
            summary_json=current_summary_json,
            instructions=instructions,
        )

        new_dataset_id = uuid.uuid4()
        self._data_repo.save_csv_data(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            dataset_id=new_dataset_id,
            df=result_df,
            overwrite=True,
            include_index=False,
        )

        new_dataset_summary = self._profiling_tool.extract_dataset_summary(
            result_df,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        
        needs_cleaned_flag = self._is_protocol_discussion_complete_and_data_cleaning_pending(request)
        request.orchestrator_state.set(
            request.node_state.name(),
            {
                "working_dataset_id": new_dataset_id,
                "latest_dataset_summary": new_dataset_summary,
                "data_cleaned": True if not needs_cleaned_flag else None,
            },
        )

        return {
            "status": "dataset_updated",
            "instruction": instructions,
            "new_dataset_id": str(new_dataset_id),
            "row_count": int(len(result_df)),
            "columns": [str(column) for column in result_df.columns],
            "preview": _dataframe_preview(result_df),
        }

    def _run_data_manipulation_tool(
        self,
        *,
        dataframe: pd.DataFrame,
        conversation_id: UUID,
        summary_json: str,
        instructions: str,
    ) -> pd.DataFrame:
        manipulate = self._data_manipulation_tool.manipulate
        params = inspect.signature(manipulate).parameters

        kwargs: dict[str, Any] = {
            "dataframe": dataframe,
            "data_summary": summary_json,
            "instructions": instructions,
        }
        if "table_name" in params:
            kwargs["table_name"] = _conversation_id_to_table_name(conversation_id)
        elif "conversation_id" in params:
            kwargs["conversation_id"] = str(conversation_id)
        else:
            raise TypeError(
                "data manipulation tool must accept either 'table_name' or "
                "'conversation_id'"
            )
        if "retry_attempts" in params:
            kwargs["retry_attempts"] = _DATA_MANIPULATION_RETRY_ATTEMPTS

        return manipulate(**kwargs)

    def _build_ready_message(self, *, summary: DatasetSummaryModel) -> str:
        profiles = list(summary.profiles)
        shown_profiles = profiles[:_INITIAL_SUMMARY_MAX_COLUMNS]
        preview = ", ".join(
            f"{profile.name} ({profile.inferred_kind.lower()})"
            for profile in shown_profiles
        )
        extra_columns = len(profiles) - len(shown_profiles)
        if extra_columns > 0:
            preview += f", +{extra_columns} more"
        return (
            f"Data manipulation is ready — {summary.n_rows} rows, {len(profiles)} columns: "
            f"{preview}. Ask for a dataset-changing transformation."
        )

    def _build_out_of_scope_message(
        self,
        *,
        user_message: str,
        dataset_summary_json: str,
    ) -> str:
        payload: JSONDict = {
            "user_message": user_message,
            "dataset_summary": dataset_summary_json,
        }
        response = self._llm.generate(
            system_prompt=data_manupulation_out_of_scope_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.3),
            history=None,
        )
        return response.content

    def _build_final_message(
        self,
        *,
        manipulation_result: JSONDict,
        dataset_context: JSONDict,
    ) -> str:
        payload: JSONDict = {
            "manipulation_result": manipulation_result,
            "dataset_context": dataset_context,
        }
        response = self._llm.generate(
            system_prompt=data_manupulation_final_response_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            config=LLMConfig(model="basic", temperature=0.3),
            history=None,
        )
        return response.content

    def _build_final_message_fallback(
        self,
        *,
        manipulation_result: JSONDict,
    ) -> str:
        if str(manipulation_result.get("status", "")).strip() != "dataset_updated":
            return "The requested dataset update could not be completed."

        row_count = int(manipulation_result.get("row_count", 0) or 0)
        column_count = len(cast(list[str], manipulation_result.get("columns", [])))
        return (
            "Updated the working dataset and saved a new dataset version. "
            f"The updated dataset has {row_count} rows and {column_count} columns."
        )

    def _is_protocol_discussion_complete_and_data_cleaning_pending(self, request: NodeRequest) -> bool:
        protocol_discussion = request.orchestrator_state.get("protocol_discussion")
        data_cleaned = request.orchestrator_state.get("data_cleaned")
        if protocol_discussion is None or data_cleaned is None or data_cleaned is not True:
            return True
        return False


    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
        action: Action = "NEEDS_INPUT",
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataManupulationState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action=action,
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _needs_data_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataManupulationState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )


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


def _recent_history_text(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history or len(messages_history) <= 1:
        return None

    recent_messages = list(messages_history[-5:-1])
    if not recent_messages:
        return None

    return get_chat_messages_role_and_message_json(recent_messages)


def _dataframe_preview(dataframe: pd.DataFrame, *, row_limit: int = 10) -> JSONDict:
    preview_df = dataframe.head(row_limit).copy()
    for column in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[column]):
            preview_df[column] = preview_df[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    return {
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "preview_rows": preview_df.to_dict(orient="records"),
    }


def _conversation_id_to_table_name(conversation_id: UUID) -> str:
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).hexdigest()
    return f"{_WORKING_TABLE_PREFIX}{digest[:_WORKING_TABLE_HASH_HEX_LEN]}"
