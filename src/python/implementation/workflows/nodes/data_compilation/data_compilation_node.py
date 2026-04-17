from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.data_compilation.data_compilation_deps import (
    DataCompilationDeps,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_causal_spec_prompt,
    data_compilation_node_info,
    data_compilation_review_decision_prompt,
    data_compilation_review_summary_prompt,
    data_compilation_single_column_transformation_plan_prompt,
    data_compilation_transformation_plan_prompt,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationPayloadModel,
    DataCompilationState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    DatasetSummaryModel,
    DatetimeColumnProfileModel,
    NumericColumnProfileModel,
    OtherColumnProfileModel,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_app_logger(__name__, component="data_compilation_node", log_type="node")


class _ReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class _ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["confirm", "revise", "clarify"]
    assistant_message: str = Field(..., min_length=1)


class _TransformPlanDraftColumn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    role: Literal["covariate", "effect_modifier"]
    preset: Literal[
        "drop",
        "passthrough",
        "cat_onehot",
        "num_standard",
        "num_minmax",
        "num_log1p",
        "datetime_epoch_seconds",
        "map_binary",
        "map_ordinal",
    ]
    mapping: dict[str, float] | None = None
    order: list[str] | None = None

    @model_validator(mode="after")
    def _validate_draft(self) -> _TransformPlanDraftColumn:
        if self.preset == "map_binary" and not self.mapping:
            raise ValueError("map_binary requires a grounded mapping")
        if self.preset == "map_ordinal" and not self.order:
            raise ValueError("map_ordinal requires a grounded order")
        if self.preset != "map_binary" and self.mapping is not None:
            raise ValueError("mapping is only allowed for map_binary")
        if self.preset != "map_ordinal" and self.order is not None:
            raise ValueError("order is only allowed for map_ordinal")
        return self


class DataCompilationNode(Node):
    NAME: ClassVar[str] = DataCompilationState.NAME

    def __init__(
        self,
        *,
        data_repo: DataRepo,
        llm: LLMService,
        tools_factory: ToolFactory,
    ) -> None:
        self._data_repo = data_repo
        self._llm = llm
        self._profiling_tool = cast(
            DatasetProfilingTool, tools_factory.get_tool(DatasetProfilingTool.NAME)
        )
        self._causal_specs_tool = cast(
            CausalSpecsTool, tools_factory.get_tool(CausalSpecsTool.NAME)
        )
        self._encoding_plan_tool = cast(
            EncodingPlanTool, tools_factory.get_tool(EncodingPlanTool.NAME)
        )

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return data_compilation_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, DataCompilationState):
            raise TypeError(
                f"{self.name}: expected DataCompilationState, got "
                f"{type(request.node_state).__name__}"
            )

        payload = request.node_state.payload.model_copy(deep=True)
        try:
            deps = DataCompilationDeps.from_request(request)
        except Exception as exc:
            log.exception("data compilation dependencies missing", error=safe_err(exc))
            return self._needs_data_result(
                request=request,
                user_message=(
                    "The compilation stage is missing the active dataset or the confirmed "
                    "protocol. Please complete dataset cleaning and protocol confirmation first."
                ),
            )
        try:
            source_df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=1_000_000,
            )
        except Exception as exc:
            log.exception(
                "failed to load data compilation source dataset",
                dataset_id=str(deps.dataset_id),
                error=safe_err(exc),
            )
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the active working dataset for compilation. Please "
                    "re-upload or reselect the dataset and try again."
                ),
            )

        source_summary = deps.dataset_summary
        payload, source_changed = self._bind_payload_to_source(
            payload=payload,
            dataset_id=deps.dataset_id,
            protocol_discussion=deps.protocol_discussion,
        )

        latest_user_message = _latest_user_message(request.read_only_messages_history)

        if payload.phase == "REVIEW_READY":
            if not self._review_payload_complete(payload):
                log.warning(
                    "data compilation review payload incomplete; recompiling",
                    conversation_id=str(request.conversation_id),
                    source_dataset_id=str(deps.dataset_id),
                )
                payload = payload.reset_for_recompile(
                    dataset_id=deps.dataset_id,
                    protocol_discussion=deps.protocol_discussion,
                )
                return self._compile(
                    request=request,
                    payload=payload,
                    source_df=source_df,
                    source_dataset_id=deps.dataset_id,
                    source_summary=source_summary,
                    protocol_discussion=deps.protocol_discussion,
                    source_changed=False,
                )
            if latest_user_message is None:
                return self._needs_input_result(
                    request=request,
                    payload=payload,
                    user_message=payload.assistant_message
                    or "Please confirm the compiled dataset and transformation plan.",
                )
            return self._handle_review_response(
                request=request,
                payload=payload,
                latest_user_message=latest_user_message,
            )

        if payload.phase == "CONFIRMED" and not source_changed:
            return self._done_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message or "The compiled setup is already confirmed.",
            )

        if payload.phase == "FAILED" and not source_changed:
            return self._aborted_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "The compilation step is blocked and needs upstream revision.",
            )

        return self._compile(
            request=request,
            payload=payload,
            source_df=source_df,
            source_dataset_id=deps.dataset_id,
            source_summary=source_summary,
            protocol_discussion=deps.protocol_discussion,
            source_changed=source_changed,
        )

    def _bind_payload_to_source(
        self,
        *,
        payload: DataCompilationPayloadModel,
        dataset_id: UUID,
        protocol_discussion: str,
    ) -> tuple[DataCompilationPayloadModel, bool]:
        if (
            payload.source_dataset_id == dataset_id
            and payload.source_protocol_discussion == protocol_discussion
        ):
            return payload, False

        if (
            payload.source_dataset_id is None
            and payload.source_protocol_discussion is None
            and payload.phase == "INIT"
        ):
            return payload.bind_source(
                dataset_id=dataset_id,
                protocol_discussion=protocol_discussion,
            ), False

        return payload.reset_for_recompile(
            dataset_id=dataset_id,
            protocol_discussion=protocol_discussion,
        ), True

    def _compile(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        source_df: pd.DataFrame,
        source_dataset_id: UUID,
        source_summary: DatasetSummaryModel,
        protocol_discussion: str,
        source_changed: bool,
    ) -> NodeExecutionResult:
        try:
            causal_spec = self._compile_causal_spec(
                protocol_discussion=protocol_discussion,
                source_summary=source_summary,
            )
        except Exception as exc:
            log.exception("data compilation causal spec failed", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not compile the confirmed protocol into a causal specification. "
                    "Please revise the confirmed protocol details and try again."
                ),
                error_message=f"causal specification compilation failed: {safe_err(exc)}",
            )

        try:
            compiled_df = self._build_protocol_scope_dataframe(
                dataframe=source_df,
                causal_spec=causal_spec,
            )
            compiled_dataset_id = uuid.uuid4()
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=compiled_dataset_id,
                df=compiled_df,
                overwrite=True,
                include_index=False,
            )
            compiled_dataset_summary = self._profile_dataset(compiled_df)
        except Exception as exc:
            log.exception("data compilation dataset build failed", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not compile the protocol-scope dataset from the confirmed "
                    "protocol. Please revise the protocol assumptions and try again."
                ),
                error_message=f"compiled dataset generation failed: {safe_err(exc)}",
            )

        try:
            transformation_plan = self._compile_transformation_plan(
                causal_spec=causal_spec,
                compiled_dataset_summary=compiled_dataset_summary,
            )
        except Exception as exc:
            log.exception("data compilation transformation plan failed", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I compiled the dataset, but I could not build the baseline "
                    "transformation plan yet. Please revise the protocol or encodings and try again."
                ),
                error_message=f"transformation plan compilation failed: {safe_err(exc)}",
            )

        try:
            review_message = self._build_review_summary_message(
                protocol_discussion=protocol_discussion,
                compiled_causal_spec=causal_spec,
                compiled_dataset_summary=compiled_dataset_summary,
                transformation_plan=transformation_plan,
                messages_history=request.read_only_messages_history,
            )
        except Exception as exc:
            log.exception("data compilation review summary failed", error=safe_err(exc))
            review_message = _build_review_summary_fallback(
                compiled_dataset_summary=compiled_dataset_summary,
                compiled_causal_spec=causal_spec,
                transformation_plan=transformation_plan,
            )

        if source_changed:
            review_message = (
                "The active dataset or confirmed protocol changed, so I recompiled the "
                f"dataset and transformation plan. {review_message}"
            )

        review_payload = payload.model_copy(
            update={
                "source_dataset_id": source_dataset_id,
                "source_protocol_discussion": protocol_discussion,
                "compiled_dataset_id": compiled_dataset_id,
                "compiled_dataset_summary": compiled_dataset_summary,
                "compiled_causal_spec": causal_spec,
                "transformation_plan": transformation_plan,
                "phase": "REVIEW_READY",
                "assistant_message": review_message,
                "system_message": None,
                "error_message": None,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=review_payload,
            user_message=review_message,
        )

    def _compile_causal_spec(
        self,
        *,
        protocol_discussion: str,
        source_summary: DatasetSummaryModel,
    ) -> CausalSpec:
        context_payload = {
            "protocol_discussion": protocol_discussion,
            "dataset_summary": _dataset_summary_prompt_payload(source_summary),
        }
        causal_schema = self._causal_specs_tool.build_backdoor_schema(
            data_summary=source_summary,
        )
        causal_spec = self._llm.generate_json(
            schema=causal_schema,
            system_prompt=data_compilation_causal_spec_prompt(),
            user_prompt=json.dumps(context_payload, ensure_ascii=False),
            config=LLMConfig(model="pro", temperature=0.1),
            history=None,
            max_attempts=3,
        )
        return self._causal_specs_tool.post_validate_backdoor_spec(
            causal_spec=causal_spec,
            data_summary=source_summary,
        )

    def _build_protocol_scope_dataframe(
        self,
        *,
        dataframe: pd.DataFrame,
        causal_spec: CausalSpec,
    ) -> pd.DataFrame:
        protocol_scope_columns = _protocol_scope_columns(causal_spec)
        missing_columns = [
            column for column in protocol_scope_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                "compiled causal spec references columns missing from the working dataset: "
                f"{missing_columns}"
            )
        return dataframe.loc[:, protocol_scope_columns].copy()

    def _compile_transformation_plan(
        self,
        *,
        causal_spec: CausalSpec,
        compiled_dataset_summary: DatasetSummaryModel,
    ) -> TransformPlan:
        expected_role_by_column = _protocol_scope_role_by_column(causal_spec)
        eligible_columns = list(expected_role_by_column.keys())
        context_payload = {
            "compiled_dataset_summary": _dataset_summary_prompt_payload(
                compiled_dataset_summary,
                include_columns=eligible_columns,
            ),
            "eligible_columns": eligible_columns,
            "expected_role_by_column": expected_role_by_column,
            "required_plan_column_count": len(expected_role_by_column),
        }
        draft_schema = _build_transform_plan_draft_schema(
            expected_role_by_column=expected_role_by_column,
        )
        validation_summary = _eligible_dataset_summary(
            compiled_dataset_summary,
            include_columns=eligible_columns,
        )
        try:
            transform_plan_draft = self._generate_batch_transform_plan_draft(
                draft_schema=draft_schema,
                context_payload=context_payload,
            )
        except Exception as exc:
            log.warning(
                "batch transformation plan draft failed, falling back to per-column generation",
                error=_exception_chain_text(exc),
                eligible_columns=eligible_columns,
                expected_role_by_column=expected_role_by_column,
            )
            transform_plan_draft = self._generate_columnwise_transform_plan_draft(
                expected_role_by_column=expected_role_by_column,
                validation_summary=validation_summary,
            )

        try:
            transform_plan_payload = _materialize_transform_plan_payload_from_draft(
                draft=transform_plan_draft,
                validation_summary=validation_summary,
            )
        except Exception as exc:
            log.exception(
                "data compilation transformation plan draft materialization failed",
                error=_exception_chain_text(exc),
                eligible_columns=eligible_columns,
                expected_role_by_column=expected_role_by_column,
                generated_draft=transform_plan_draft.model_dump(mode="json"),
            )
            raise

        try:
            return self._encoding_plan_tool.validate_encoding_payload(
                payload=transform_plan_payload,
                data_summary=validation_summary,
                covariate_columns=causal_spec.covariates,
                effect_modifier_columns=causal_spec.effect_modifiers,
            )
        except Exception as exc:
            log.exception(
                "data compilation transformation plan validation failed",
                error=_exception_chain_text(exc),
                eligible_columns=eligible_columns,
                expected_role_by_column=expected_role_by_column,
                generated_draft=transform_plan_draft.model_dump(mode="json"),
                generated_plan=transform_plan_payload,
            )
            raise

    def _generate_batch_transform_plan_draft(
        self,
        *,
        draft_schema: type[BaseModel],
        context_payload: dict[str, Any],
    ) -> BaseModel:
        return self._llm.generate_json(
            schema=draft_schema,
            system_prompt=data_compilation_transformation_plan_prompt(),
            user_prompt=json.dumps(context_payload, ensure_ascii=False),
            config=LLMConfig(
                model="basic",
                temperature=1.0,
                max_tokens=2000,
            ),
            history=None,
            max_attempts=1,
        )

    def _generate_columnwise_transform_plan_draft(
        self,
        *,
        expected_role_by_column: dict[str, str],
        validation_summary: DatasetSummaryModel,
    ) -> BaseModel:
        profiles_by_name = {
            str(profile.name).strip(): profile for profile in validation_summary.profiles
        }
        generated_columns: list[_TransformPlanDraftColumn] = []

        for column, role in expected_role_by_column.items():
            profile = profiles_by_name.get(column)
            if profile is None:
                raise ValueError(
                    f"eligible transformation-plan column is missing from validation summary: {column}"
                )

            schema = _build_role_specific_transform_plan_draft_column_type(
                columns=(column,),
                role=cast(Literal["covariate", "effect_modifier"], role),
            )
            payload = {
                "column_name": column,
                "expected_role": role,
                "column_profile": _column_prompt_payload(profile),
            }

            try:
                generated_columns.append(
                    self._llm.generate_json(
                        schema=schema,
                        system_prompt=data_compilation_single_column_transformation_plan_prompt(),
                        user_prompt=json.dumps(payload, ensure_ascii=False),
                        config=LLMConfig(
                            model="basic",
                            temperature=1.0,
                            max_tokens=500,
                        ),
                        history=None,
                        max_attempts=1,
                    )
                )
            except Exception as exc:
                log.exception(
                    "columnwise transformation plan generation failed",
                    column=column,
                    role=role,
                    column_profile=_column_prompt_payload(profile),
                    error=_exception_chain_text(exc),
                )
                raise

        draft_wrapper_schema = create_model(
            f"TransformPlanDraftGenerated_{len(generated_columns)}",
            __module__=__name__,
            columns=(list[_TransformPlanDraftColumn], ...),
        )
        return draft_wrapper_schema.model_validate({"columns": generated_columns})

    def _review_payload_complete(self, payload: DataCompilationPayloadModel) -> bool:
        return (
            payload.compiled_dataset_id is not None
            and payload.compiled_dataset_summary is not None
            and payload.compiled_causal_spec is not None
            and payload.transformation_plan is not None
        )

    def _handle_review_response(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        latest_user_message: str,
    ) -> NodeExecutionResult:
        if not self._review_payload_complete(payload):
            return self._failed_result(
                request=request,
                payload=DataCompilationPayloadModel(),
                user_message=(
                    "The stored compilation review state is incomplete, so this step needs "
                    "to be recompiled from the latest dataset and confirmed protocol."
                ),
                error_message="review payload incomplete",
            )

        decision = self._llm.generate_json(
            schema=_ReviewDecision,
            system_prompt=data_compilation_review_decision_prompt(),
            user_prompt=json.dumps(
                {
                    "compiled_dataset_summary": payload.compiled_dataset_summary.model_dump(
                        mode="json"
                    ),
                    "compiled_causal_spec": payload.compiled_causal_spec.model_dump(
                        mode="json"
                    ),
                    "transformation_plan": payload.transformation_plan.model_dump(
                        mode="json"
                    ),
                    "latest_user_message": latest_user_message,
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.0),
            history=None,
            max_attempts=3,
        )

        if decision.action == "confirm":
            request.orchestrator_state.set(
                request.node_state.name(),
                {
                    "working_dataset_id": payload.compiled_dataset_id,
                    "latest_dataset_summary": payload.compiled_dataset_summary,
                    "causal_spec": payload.compiled_causal_spec,
                    "data_transformation_plan": payload.transformation_plan,
                },
            )
            confirmed_payload = payload.model_copy(
                update={
                    "phase": "CONFIRMED",
                    "assistant_message": decision.assistant_message,
                    "system_message": None,
                    "error_message": None,
                }
            )
            return self._done_result(
                request=request,
                payload=confirmed_payload,
                user_message=decision.assistant_message,
            )

        if decision.action == "revise":
            failed_payload = payload.model_copy(
                update={
                    "phase": "FAILED",
                    "assistant_message": decision.assistant_message,
                    "system_message": "DATA_COMPILATION_REVISE_REQUESTED",
                    "error_message": "user rejected the compiled review",
                }
            )
            return self._aborted_result(
                request=request,
                payload=failed_payload,
                user_message=decision.assistant_message,
            )

        review_payload = payload.model_copy(
            update={
                "assistant_message": decision.assistant_message,
                "system_message": None,
                "error_message": None,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=review_payload,
            user_message=decision.assistant_message,
        )

    def _build_review_summary_message(
        self,
        *,
        protocol_discussion: str,
        compiled_causal_spec: CausalSpec,
        compiled_dataset_summary: DatasetSummaryModel,
        transformation_plan: TransformPlan,
        messages_history: Sequence[ChatMessage] | None,
    ) -> str:
        history = list(messages_history[-4:]) if messages_history else None
        review_summary = self._llm.generate_json(
            schema=_ReviewSummary,
            system_prompt=data_compilation_review_summary_prompt(),
            user_prompt=json.dumps(
                {
                    "protocol_discussion": protocol_discussion,
                    "compiled_causal_spec": compiled_causal_spec.model_dump(mode="json"),
                    "compiled_dataset_summary": _dataset_summary_prompt_payload(
                        compiled_dataset_summary
                    ),
                    "transformation_plan": transformation_plan.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="mini", temperature=0.2),
            history=history,
            max_attempts=2,
        )
        return review_summary.assistant_message

    def _profile_dataset(self, dataframe: pd.DataFrame) -> DatasetSummaryModel:
        return self._profiling_tool.extract_dataset_summary(
            dataframe,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataCompilationState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _needs_data_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataCompilationState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _done_result(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataCompilationState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="DONE",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _aborted_result(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataCompilationState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="ABORTED",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _failed_result(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        user_message: str,
        error_message: str,
    ) -> NodeExecutionResult:
        failed_payload = payload.model_copy(
            update={
                "phase": "FAILED",
                "assistant_message": user_message,
                "system_message": "DATA_COMPILATION_FAILED",
                "error_message": error_message,
            }
        )
        return self._aborted_result(
            request=request,
            payload=failed_payload,
            user_message=user_message,
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


def _protocol_scope_columns(causal_spec: CausalSpec) -> list[str]:
    ordered_columns = [
        str(causal_spec.treatment_spec.column),
        str(causal_spec.outcome_spec.column),
        *(str(column) for column in causal_spec.covariates),
        *(str(column) for column in causal_spec.effect_modifiers),
    ]
    deduped: list[str] = []
    for column in ordered_columns:
        normalized = column.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _protocol_scope_role_by_column(causal_spec: CausalSpec) -> dict[str, str]:
    role_by_column: dict[str, str] = {}
    for column in causal_spec.covariates:
        normalized = str(column).strip()
        if normalized:
            role_by_column[normalized] = "covariate"
    for column in causal_spec.effect_modifiers:
        normalized = str(column).strip()
        if normalized:
            role_by_column[normalized] = "effect_modifier"
    return role_by_column


def _dataset_summary_prompt_payload(
    summary: DatasetSummaryModel,
    *,
    include_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    scoped_summary = _eligible_dataset_summary(summary, include_columns=include_columns)
    return {
        "n_rows": scoped_summary.n_rows,
        "columns": [_column_prompt_payload(profile) for profile in scoped_summary.profiles],
    }


def _eligible_dataset_summary(
    summary: DatasetSummaryModel,
    *,
    include_columns: Sequence[str] | None = None,
) -> DatasetSummaryModel:
    if include_columns is None:
        return summary
    include_set = {str(column).strip() for column in include_columns if str(column).strip()}
    return DatasetSummaryModel(
        n_rows=summary.n_rows,
        profiles=[
            profile
            for profile in summary.profiles
            if str(profile.name).strip() in include_set
        ],
    )


def _column_prompt_payload(
    profile: (
        NumericColumnProfileModel
        | DatetimeColumnProfileModel
        | BooleanColumnProfileModel
        | CategoricalColumnProfileModel
        | OtherColumnProfileModel
    ),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": str(profile.name).strip(),
        "kind": str(profile.inferred_kind),
        "dtype": profile.dtype,
        "missing_rate": profile.missing_rate,
        "distinct_count": profile.distinct_count,
    }

    if isinstance(profile, NumericColumnProfileModel):
        payload["range"] = {
            "min": profile.summary.min,
            "max": profile.summary.max,
        }
        return payload

    if isinstance(profile, DatetimeColumnProfileModel):
        payload["range"] = {
            "min": profile.summary.min,
            "max": profile.summary.max,
        }
        return payload

    if isinstance(profile, BooleanColumnProfileModel):
        payload["known_values"] = list(profile.summary.counts.keys())
        return payload

    if isinstance(profile, CategoricalColumnProfileModel):
        payload["top_values"] = [
            item.value for item in profile.summary.top_categories
        ]
        return payload

    if isinstance(profile, OtherColumnProfileModel):
        payload["sample_values"] = list(profile.summary.distinct_values_sample)
        return payload

    return payload


def _build_transform_plan_draft_schema(
    *,
    expected_role_by_column: dict[str, str],
) -> type[BaseModel]:
    from typing import Annotated

    covariate_columns = tuple(
        column
        for column, role in expected_role_by_column.items()
        if role == "covariate"
    )
    effect_modifier_columns = tuple(
        column
        for column, role in expected_role_by_column.items()
        if role == "effect_modifier"
    )
    variants: list[Any] = []
    if covariate_columns:
        variants.append(
            _build_role_specific_transform_plan_draft_column_type(
                columns=covariate_columns,
                role="covariate",
            )
        )
    if effect_modifier_columns:
        variants.append(
            _build_role_specific_transform_plan_draft_column_type(
                columns=effect_modifier_columns,
                role="effect_modifier",
            )
        )
    if not variants:
        raise ValueError("expected_role_by_column must not be empty")

    if len(variants) == 1:
        constrained_item_type = variants[0]
    else:
        union_type = variants[0]
        for variant in variants[1:]:
            union_type = union_type | variant
        constrained_item_type = Annotated[union_type, Field(discriminator="role")]

    return create_model(
        f"TransformPlanDraft_{len(expected_role_by_column)}",
        __module__=__name__,
        columns=(
            list[constrained_item_type],
            Field(
                ...,
                min_length=len(expected_role_by_column),
                max_length=len(expected_role_by_column),
            ),
        ),
    )


def _build_role_specific_transform_plan_draft_column_type(
    *,
    columns: Sequence[str],
    role: Literal["covariate", "effect_modifier"],
) -> type[_TransformPlanDraftColumn]:
    return create_model(
        f"TransformPlanDraftColumn_{role}_{len(columns)}",
        __base__=_TransformPlanDraftColumn,
        __module__=__name__,
        column=(Literal.__getitem__(tuple(columns)), ...),
        role=(Literal.__getitem__((role,)), ...),
    )


def _materialize_transform_plan_payload_from_draft(
    *,
    draft: BaseModel,
    validation_summary: DatasetSummaryModel,
) -> dict[str, Any]:
    profiles_by_name = {
        str(profile.name).strip(): profile for profile in validation_summary.profiles
    }
    draft_columns = getattr(draft, "columns")
    payload_columns: list[dict[str, Any]] = []
    for draft_column in draft_columns:
        column = str(draft_column.column).strip()
        profile = profiles_by_name.get(column)
        if profile is None:
            raise ValueError(f"Draft references unknown validation column: {column}")
        payload_columns.append(
            {
                "column": column,
                "role": str(draft_column.role),
                "encoding": _materialize_encoding_payload_from_draft_column(
                    draft_column=draft_column,
                    profile=profile,
                ),
            }
        )
    return {"columns": payload_columns}


def _materialize_encoding_payload_from_draft_column(
    *,
    draft_column: _TransformPlanDraftColumn,
    profile: (
        NumericColumnProfileModel
        | DatetimeColumnProfileModel
        | BooleanColumnProfileModel
        | CategoricalColumnProfileModel
        | OtherColumnProfileModel
    ),
) -> dict[str, Any]:
    match draft_column.preset:
        case "drop":
            return {"preset": "drop"}
        case "passthrough":
            return {"preset": "passthrough"}
        case "cat_onehot":
            return {
                "preset": "cat_onehot",
                "drop_first": False,
                "handle_unknown": "ignore",
                "missing": "impute_token",
                "missing_token": "__MISSING__",
            }
        case "num_standard":
            return {
                "preset": "num_standard",
                "impute": "median",
                "add_missing_indicator": True,
            }
        case "num_minmax":
            return {
                "preset": "num_minmax",
                "impute": "median",
                "add_missing_indicator": True,
                "eps": 1e-12,
            }
        case "num_log1p":
            return {
                "preset": "num_log1p",
                "impute": "median",
                "add_missing_indicator": True,
                "allow_negative": False,
                "then_scale": "none",
            }
        case "datetime_epoch_seconds":
            return {
                "preset": "datetime_epoch_seconds",
                "errors": "coerce",
                "unit": "s",
                "add_missing_indicator": True,
            }
        case "map_binary":
            mapping = draft_column.mapping
            if not mapping:
                raise ValueError(
                    f"map_binary requires a grounded mapping for column '{draft_column.column}'"
                )
            return {
                "preset": "map_binary",
                "mapping": mapping,
                "allow_unknown": True,
                "unknown_value": -1.0,
                "missing": "as_unknown",
            }
        case "map_ordinal":
            order = draft_column.order
            if not order:
                raise ValueError(
                    f"map_ordinal requires a grounded order for column '{draft_column.column}'"
                )
            return {
                "preset": "map_ordinal",
                "order": order,
                "start": 0,
                "allow_unknown": True,
                "unknown_value": -1,
                "missing": "as_unknown",
            }
        case _:
            raise ValueError(
                f"Unsupported draft preset '{draft_column.preset}' for column '{draft_column.column}'"
            )


def _exception_chain_text(exc: Exception) -> str:
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        message = str(current).strip() or current.__class__.__name__
        if message not in chain:
            chain.append(message)
        current = current.__cause__ or current.__context__
    return " | ".join(chain[:5])


def _build_review_summary_fallback(
    *,
    compiled_dataset_summary: DatasetSummaryModel,
    compiled_causal_spec: CausalSpec,
    transformation_plan: TransformPlan,
) -> str:
    transform_lines = [
        f"{column.column}: {column.encoding.preset}" for column in transformation_plan.columns
    ]
    retained_columns = [str(profile.name).strip() for profile in compiled_dataset_summary.profiles]
    return (
        "I prepared the data for causal modeling by narrowing the dataset to the "
        "protocol-scope columns and compiling the baseline transformation plan. "
        f"The compiled dataset has {compiled_dataset_summary.n_rows} rows and "
        f"{len(compiled_dataset_summary.profiles)} columns. "
        f"Retained columns: {', '.join(retained_columns) if retained_columns else 'None'}. "
        f"Treatment: {compiled_causal_spec.treatment_spec.column}. "
        f"Outcome: {compiled_causal_spec.outcome_spec.column}. "
        f"Covariates: {', '.join(compiled_causal_spec.covariates) if compiled_causal_spec.covariates else 'None'}. "
        f"Effect modifiers: {', '.join(compiled_causal_spec.effect_modifiers) if compiled_causal_spec.effect_modifiers else 'None'}. "
        f"Planned baseline transformations: {'; '.join(transform_lines)}. "
        "Please confirm this compiled dataset and transformation plan, or tell me exactly what should change."
    )


__all__ = ["DataCompilationNode"]
