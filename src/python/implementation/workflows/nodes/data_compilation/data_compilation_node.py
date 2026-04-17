from __future__ import annotations

import hashlib
import json
import numbers
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
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
    data_compilation_action_decision_prompt,
    data_compilation_causal_spec_prompt,
    data_compilation_cleaning_instructions_prompt,
    data_compilation_discrepancy_repair_prompt,
    data_compilation_locked_spec_revision_prompt,
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
from python.implementation.workflows.tools.causal.specs.causal_spec import (
    BinaryOutcomeSpecModel,
    CausalSpec,
    ContinuousOutcomeSpecModel,
)
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
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
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.implementation.workflows.utils.utils import safe_err

log = get_app_logger(__name__, component="data_compilation_node", log_type="node")


class _ReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class _ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["confirm", "revise", "clarify"]
    assistant_message: str = Field(..., min_length=1)


class _ActionRequiredDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal[
        "retry_transform",
        "retry_cleaning",
        "revise_spec_details",
        "revise_protocol",
        "clarify",
    ]
    assistant_message: str = Field(..., min_length=1)
    repair_request: str | None = None


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


@dataclass(frozen=True)
class _PreparedSourceArtifacts:
    dataframe: pd.DataFrame
    summary: DatasetSummaryModel
    actions: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class _CompiledDatasetArtifacts:
    dataframe: pd.DataFrame
    summary: DatasetSummaryModel
    actions: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class _ValidatedTransformPlanArtifacts:
    plan: TransformPlan | None
    warnings: list[str]
    issues: list[ValidationIssueModel]


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
        self._validation_tool = cast(
            ValidationBackdoorTool,
            tools_factory.get_tool(ValidationBackdoorTool.NAME),
        )
        self._data_manipulation_tool = cast(
            DataManipulationTool, tools_factory.get_tool(DataManipulationTool.NAME)
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
            protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
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
                    protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
                )
                return self._compile_pipeline(
                    request=request,
                    payload=payload,
                    source_df=source_df,
                    source_dataset_id=deps.dataset_id,
                    source_summary=source_summary,
                    protocol_discussion=deps.protocol_discussion,
                    protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
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
                source_df=source_df,
                source_summary=source_summary,
                protocol_discussion=deps.protocol_discussion,
                protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
            )

        if payload.phase == "ACTION_REQUIRED":
            if not self._action_payload_complete(payload):
                log.warning(
                    "data compilation action-required payload incomplete; recompiling",
                    conversation_id=str(request.conversation_id),
                    source_dataset_id=str(deps.dataset_id),
                )
                payload = payload.reset_for_recompile(
                    dataset_id=deps.dataset_id,
                    protocol_discussion=deps.protocol_discussion,
                    protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
                )
                return self._compile_pipeline(
                    request=request,
                    payload=payload,
                    source_df=source_df,
                    source_dataset_id=deps.dataset_id,
                    source_summary=source_summary,
                    protocol_discussion=deps.protocol_discussion,
                    protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
                    source_changed=False,
                )
            if latest_user_message is None:
                return self._needs_input_result(
                    request=request,
                    payload=payload,
                    user_message=payload.assistant_message
                    or "Validation found blocking issues. Tell me what to change next.",
                )
            return self._handle_action_required_response(
                request=request,
                payload=payload,
                latest_user_message=latest_user_message,
                source_df=source_df,
                source_summary=source_summary,
                protocol_discussion=deps.protocol_discussion,
                protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
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

        return self._compile_pipeline(
            request=request,
            payload=payload,
            source_df=source_df,
            source_dataset_id=deps.dataset_id,
            source_summary=source_summary,
            protocol_discussion=deps.protocol_discussion,
            protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
            source_changed=source_changed,
        )

    def _bind_payload_to_source(
        self,
        *,
        payload: DataCompilationPayloadModel,
        dataset_id: UUID,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
    ) -> tuple[DataCompilationPayloadModel, bool]:
        if (
            payload.source_dataset_id == dataset_id
            and payload.source_protocol_discussion == protocol_discussion
            and payload.source_protocol_cleaning_instructions == protocol_cleaning_instructions
        ):
            return payload, False

        if (
            payload.source_dataset_id is None
            and payload.source_protocol_discussion is None
            and payload.source_protocol_cleaning_instructions is None
            and payload.phase == "INIT"
        ):
            return payload.bind_source(
                dataset_id=dataset_id,
                protocol_discussion=protocol_discussion,
                protocol_cleaning_instructions=protocol_cleaning_instructions,
            ), False

        return payload.reset_for_recompile(
            dataset_id=dataset_id,
            protocol_discussion=protocol_discussion,
            protocol_cleaning_instructions=protocol_cleaning_instructions,
        ), True

    def _compile_pipeline(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        source_df: pd.DataFrame,
        source_dataset_id: UUID,
        source_summary: DatasetSummaryModel,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
        source_changed: bool,
    ) -> NodeExecutionResult:
        try:
            prepared_source = self._prepare_cleaned_source(
                request=request,
                source_df=source_df,
                source_summary=source_summary,
                protocol_discussion=protocol_discussion,
                protocol_cleaning_instructions=protocol_cleaning_instructions,
            )
        except Exception as exc:
            log.exception("data compilation first-pass cleaning failed", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not prepare the working dataset from the confirmed protocol "
                    "instructions. Please revise the protocol cleaning assumptions and try again."
                ),
                error_message=f"first-pass cleaning failed: {safe_err(exc)}",
            )

        try:
            causal_spec = self._compile_causal_spec(
                protocol_discussion=protocol_discussion,
                source_summary=prepared_source.summary,
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
            compiled_dataset = self._build_compiled_dataset(
                dataframe=prepared_source.dataframe,
                causal_spec=causal_spec,
            )
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
            transform_artifacts = self._build_validated_transform_plan(
                protocol_discussion=protocol_discussion,
                causal_spec=causal_spec,
                compiled_dataset_summary=compiled_dataset.summary,
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

        compilation_actions = [
            *prepared_source.actions,
            *compiled_dataset.actions,
        ]
        compilation_warnings = [
            *prepared_source.warnings,
            *compiled_dataset.warnings,
            *transform_artifacts.warnings,
        ]
        return self._finalize_compilation_attempt(
            request=request,
            payload=payload,
            source_dataset_id=source_dataset_id,
            protocol_discussion=protocol_discussion,
            protocol_cleaning_instructions=protocol_cleaning_instructions,
            causal_spec=causal_spec,
            compiled_dataset=compiled_dataset,
            transform_artifacts=transform_artifacts,
            compilation_actions=compilation_actions,
            compilation_warnings=compilation_warnings,
            source_changed=source_changed,
        )

    def _prepare_cleaned_source(
        self,
        *,
        request: NodeRequest,
        source_df: pd.DataFrame,
        source_summary: DatasetSummaryModel,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
    ) -> _PreparedSourceArtifacts:
        cleaning_instructions = self._build_first_pass_cleaning_instructions(
            protocol_discussion=protocol_discussion,
            protocol_cleaning_instructions=protocol_cleaning_instructions,
        )
        cleaned_df = self._run_data_manipulation_tool(
            dataframe=source_df,
            conversation_id=request.conversation_id,
            data_summary=source_summary,
            instructions=cleaning_instructions,
        )
        cleaned_summary = self._profile_dataset(cleaned_df)
        actions = _summarize_summary_delta_actions(
            before_summary=source_summary,
            after_summary=cleaned_summary,
            context="Applied first-pass protocol-driven cleaning before compilation.",
        )
        return _PreparedSourceArtifacts(
            dataframe=cleaned_df,
            summary=cleaned_summary,
            actions=actions,
            warnings=[],
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

    def _build_compiled_dataset(
        self,
        *,
        dataframe: pd.DataFrame,
        causal_spec: CausalSpec,
    ) -> _CompiledDatasetArtifacts:
        protocol_scope_columns = _protocol_scope_columns(causal_spec)
        missing_columns = [
            column for column in protocol_scope_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                "compiled causal spec references columns missing from the working dataset: "
                f"{missing_columns}"
            )

        scoped_df = dataframe.loc[:, protocol_scope_columns].copy()
        actions = [
            "Retained only protocol-scope columns for compilation: "
            + ", ".join(protocol_scope_columns)
        ]
        warnings: list[str] = []

        scoped_df, treatment_actions = _drop_rows_outside_binary_spec(
            dataframe=scoped_df,
            column=str(causal_spec.treatment_spec.column),
            allowed_values={
                str(causal_spec.treatment_spec.treated),
                str(causal_spec.treatment_spec.control),
            },
            label="treatment",
        )
        actions.extend(treatment_actions)

        if isinstance(causal_spec.outcome_spec, BinaryOutcomeSpecModel):
            scoped_df, outcome_actions = _drop_rows_outside_binary_spec(
                dataframe=scoped_df,
                column=str(causal_spec.outcome_spec.column),
                allowed_values={
                    str(causal_spec.outcome_spec.event),
                    str(causal_spec.outcome_spec.non_event),
                },
                label="outcome",
            )
            actions.extend(outcome_actions)

        if scoped_df.empty:
            raise ValueError("compiled dataset is empty after protocol-scope filtering")

        _ensure_binary_treatment_arms_present(scoped_df, causal_spec=causal_spec)
        _ensure_binary_outcome_classes_present(scoped_df, causal_spec=causal_spec)

        compiled_summary = self._profile_dataset(scoped_df)
        return _CompiledDatasetArtifacts(
            dataframe=scoped_df,
            summary=compiled_summary,
            actions=actions,
            warnings=warnings,
        )

    def _build_validated_transform_plan(
        self,
        *,
        protocol_discussion: str,
        causal_spec: CausalSpec,
        compiled_dataset_summary: DatasetSummaryModel,
        repair_request: str | None = None,
        validation_issues: Sequence[ValidationIssueModel] | None = None,
    ) -> _ValidatedTransformPlanArtifacts:
        expected_role_by_column = _protocol_scope_role_by_column(causal_spec)
        eligible_columns = list(expected_role_by_column.keys())
        context_payload = {
            "confirmed_protocol_discussion": protocol_discussion,
            "compiled_causal_specification": causal_spec.model_dump(mode="json"),
            "compiled_dataset_summary": _dataset_summary_prompt_payload(
                compiled_dataset_summary,
                include_columns=eligible_columns,
            ),
            "eligible_columns": eligible_columns,
            "expected_role_by_column": expected_role_by_column,
            "required_plan_column_count": len(expected_role_by_column),
            "repair_request": repair_request,
            "validation_issues": [
                issue.model_dump(mode="json", exclude_none=True)
                for issue in (validation_issues or [])
            ],
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
                protocol_discussion=protocol_discussion,
                causal_spec=causal_spec,
                expected_role_by_column=expected_role_by_column,
                validation_summary=validation_summary,
                repair_request=repair_request,
                validation_issues=validation_issues or [],
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
            return _ValidatedTransformPlanArtifacts(
                plan=None,
                warnings=[],
                issues=[
                    _fail_issue(
                        message="Transform-plan draft could not be materialized safely.",
                        evidence={"error": _exception_chain_text(exc)},
                        fix_hint=(
                            "Revise the transform plan choices while keeping the same "
                            "locked covariate and effect-modifier columns."
                        ),
                    )
                ],
            )

        model_dict, issues = self._encoding_plan_tool.validate_encoding_payload_structured(
            payload=transform_plan_payload,
            data_summary=validation_summary,
            covariate_columns=causal_spec.covariates,
            effect_modifier_columns=causal_spec.effect_modifiers,
        )
        if issues:
            return _ValidatedTransformPlanArtifacts(
                plan=None,
                warnings=[],
                issues=[
                    _encoding_validation_issue_to_validation_issue(issue) for issue in issues
                ],
            )

        if model_dict is None:
            raise ValueError("encoding validation returned no plan and no issues")

        validated_plan = self._encoding_plan_tool.validate_encoding_payload(
            payload=transform_plan_payload,
            data_summary=validation_summary,
            covariate_columns=causal_spec.covariates,
            effect_modifier_columns=causal_spec.effect_modifiers,
        )
        warnings = _summarize_transform_plan_warnings(
            plan=validated_plan,
            compiled_dataset_summary=compiled_dataset_summary,
        )
        return _ValidatedTransformPlanArtifacts(
            plan=validated_plan,
            warnings=warnings,
            issues=[],
        )

    def _repair_cleaned_source(
        self,
        *,
        request: NodeRequest,
        source_df: pd.DataFrame,
        source_summary: DatasetSummaryModel,
        compiled_dataset_summary: DatasetSummaryModel,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
        causal_spec: CausalSpec,
        validation_issues: Sequence[ValidationIssueModel],
        repair_request: str | None,
    ) -> _PreparedSourceArtifacts:
        repair_instructions = self._build_repair_cleaning_instructions(
            protocol_discussion=protocol_discussion,
            protocol_cleaning_instructions=protocol_cleaning_instructions,
            causal_spec=causal_spec,
            compiled_dataset_summary=compiled_dataset_summary,
            validation_issues=validation_issues,
            repair_request=repair_request,
        )
        repaired_df = self._run_data_manipulation_tool(
            dataframe=source_df,
            conversation_id=request.conversation_id,
            data_summary=source_summary,
            instructions=repair_instructions,
        )
        repaired_summary = self._profile_dataset(repaired_df)
        actions = _summarize_summary_delta_actions(
            before_summary=source_summary,
            after_summary=repaired_summary,
            context=(
                "Applied one user-directed corrective cleaning pass while keeping the "
                "locked protocol columns unchanged."
            ),
        )
        warnings = [
            "A same-lock corrective cleaning pass was applied after validation found blocking issues."
        ]
        return _PreparedSourceArtifacts(
            dataframe=repaired_df,
            summary=repaired_summary,
            actions=actions,
            warnings=warnings,
        )

    def _validate_compiled_setup(
        self,
        *,
        dataframe: pd.DataFrame,
        causal_spec: CausalSpec,
        transform_plan: TransformPlan | None,
        transform_issues: Sequence[ValidationIssueModel],
    ) -> tuple[list[ValidationIssueModel], ValidationStatus]:
        scope_issues = _validate_dataset_protocol_scope_columns(
            dataframe=dataframe,
            causal_spec=causal_spec,
        )
        validation_report = self._validation_tool.validate(
            causal_spec=causal_spec,
            dataframe=dataframe,
            transform_plan=transform_plan,
        )
        validation_issues = list(validation_report.issues)
        if transform_issues and transform_plan is None:
            validation_issues = [
                issue
                for issue in validation_issues
                if issue.message
                != "Transform plan is required when covariates or effect modifiers are present."
            ]

        issues = [*scope_issues, *transform_issues, *validation_issues]
        return issues, _validation_status(issues)

    def _finalize_compilation_attempt(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        source_dataset_id: UUID,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
        causal_spec: CausalSpec,
        compiled_dataset: _CompiledDatasetArtifacts,
        transform_artifacts: _ValidatedTransformPlanArtifacts,
        compilation_actions: Sequence[str],
        compilation_warnings: Sequence[str],
        source_changed: bool,
    ) -> NodeExecutionResult:
        validation_issues, validation_status = self._validate_compiled_setup(
            dataframe=compiled_dataset.dataframe,
            causal_spec=causal_spec,
            transform_plan=transform_artifacts.plan,
            transform_issues=transform_artifacts.issues,
        )

        compiled_dataset_id = uuid.uuid4()
        self._data_repo.save_csv_data(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            dataset_id=compiled_dataset_id,
            df=compiled_dataset.dataframe,
            overwrite=True,
            include_index=False,
        )

        base_update = {
            "source_dataset_id": source_dataset_id,
            "source_protocol_discussion": protocol_discussion,
            "source_protocol_cleaning_instructions": protocol_cleaning_instructions,
            "compiled_dataset_id": compiled_dataset_id,
            "compiled_dataset_summary": compiled_dataset.summary,
            "compiled_causal_spec": causal_spec,
            "transformation_plan": transform_artifacts.plan,
            "compilation_actions": list(compilation_actions),
            "compilation_warnings": list(compilation_warnings),
            "validation_issues": validation_issues,
            "validation_status": validation_status,
        }

        if validation_status == "FAIL":
            if _has_spec_breaking_issues(validation_issues):
                user_message = _build_protocol_revision_required_message(
                    causal_spec=causal_spec,
                    issues=validation_issues,
                )
                if source_changed:
                    user_message = (
                        "The active dataset or confirmed protocol changed, so I reran "
                        f"compilation and validation. {user_message}"
                    )
                failed_payload = payload.model_copy(
                    update={
                        **base_update,
                        "phase": "FAILED",
                        "assistant_message": user_message,
                        "system_message": "DATA_COMPILATION_PROTOCOL_REVISION_REQUIRED",
                        "error_message": "spec-breaking validation issues prevent confirmation",
                    }
                )
                return self._aborted_result(
                    request=request,
                    payload=failed_payload,
                    user_message=user_message,
                )

            user_message = _build_action_required_message(
                causal_spec=causal_spec,
                issues=validation_issues,
            )
            if source_changed:
                user_message = (
                    "The active dataset or confirmed protocol changed, so I reran "
                    f"compilation and validation. {user_message}"
                )
            action_payload = payload.model_copy(
                update={
                    **base_update,
                    "phase": "ACTION_REQUIRED",
                    "assistant_message": user_message,
                    "system_message": "DATA_COMPILATION_ACTION_REQUIRED",
                    "error_message": None,
                }
            )
            return self._needs_input_result(
                request=request,
                payload=action_payload,
                user_message=user_message,
            )

        if transform_artifacts.plan is None:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not produce a validated transformation plan for the locked "
                    "compiled dataset."
                ),
                error_message="validated transform plan missing",
            )

        try:
            review_message = self._build_review_summary_message(
                protocol_discussion=protocol_discussion,
                compiled_causal_spec=causal_spec,
                compiled_dataset_summary=compiled_dataset.summary,
                transformation_plan=transform_artifacts.plan,
                compilation_actions=compilation_actions,
                compilation_warnings=compilation_warnings,
                validation_status=validation_status,
                validation_issues=validation_issues,
                messages_history=request.read_only_messages_history,
            )
        except Exception as exc:
            log.exception("data compilation review summary failed", error=safe_err(exc))
            review_message = _build_review_summary_fallback(
                compiled_dataset_summary=compiled_dataset.summary,
                compiled_causal_spec=causal_spec,
                transformation_plan=transform_artifacts.plan,
                compilation_actions=compilation_actions,
                compilation_warnings=compilation_warnings,
                validation_status=validation_status,
                validation_issues=validation_issues,
            )

        if source_changed:
            review_message = (
                "The active dataset or confirmed protocol changed, so I recompiled, "
                f"revalidated, and rebuilt the transformation plan. {review_message}"
            )

        review_payload = payload.model_copy(
            update={
                **base_update,
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
                max_tokens=10000,
            ),
            history=None,
            max_attempts=3,
        )

    def _generate_columnwise_transform_plan_draft(
        self,
        *,
        protocol_discussion: str,
        causal_spec: CausalSpec,
        expected_role_by_column: dict[str, str],
        validation_summary: DatasetSummaryModel,
        repair_request: str | None,
        validation_issues: Sequence[ValidationIssueModel],
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
                "confirmed_protocol_discussion": protocol_discussion,
                "compiled_causal_specification": causal_spec.model_dump(mode="json"),
                "column_name": column,
                "expected_role": role,
                "column_profile": _column_prompt_payload(profile),
                "repair_request": repair_request,
                "validation_issues": [
                    issue.model_dump(mode="json", exclude_none=True)
                    for issue in validation_issues
                ],
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
            and payload.validation_status is not None
        )

    def _action_payload_complete(self, payload: DataCompilationPayloadModel) -> bool:
        return (
            payload.compiled_dataset_id is not None
            and payload.compiled_dataset_summary is not None
            and payload.compiled_causal_spec is not None
            and payload.validation_status is not None
        )

    def _handle_action_required_response(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        latest_user_message: str,
        source_df: pd.DataFrame,
        source_summary: DatasetSummaryModel,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
    ) -> NodeExecutionResult:
        if not self._action_payload_complete(payload):
            return self._failed_result(
                request=request,
                payload=DataCompilationPayloadModel(),
                user_message=(
                    "The stored compilation repair state is incomplete, so this step needs "
                    "to be recompiled from the latest dataset and confirmed protocol."
                ),
                error_message="action-required payload incomplete",
            )

        decision = self._llm.generate_json(
            schema=_ActionRequiredDecision,
            system_prompt=data_compilation_action_decision_prompt(),
            user_prompt=json.dumps(
                {
                    "compiled_causal_spec": payload.compiled_causal_spec.model_dump(
                        mode="json"
                    ),
                    "compiled_dataset_summary": payload.compiled_dataset_summary.model_dump(
                        mode="json"
                    ),
                    "transformation_plan": (
                        None
                        if payload.transformation_plan is None
                        else payload.transformation_plan.model_dump(mode="json")
                    ),
                    "validation_status": payload.validation_status,
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in payload.validation_issues
                    ],
                    "latest_user_message": latest_user_message,
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.0),
            history=None,
            max_attempts=3,
        )

        if decision.action == "clarify":
            clarified_payload = payload.model_copy(
                update={
                    "assistant_message": decision.assistant_message,
                    "system_message": "DATA_COMPILATION_ACTION_REQUIRED",
                    "error_message": None,
                }
            )
            return self._needs_input_result(
                request=request,
                payload=clarified_payload,
                user_message=decision.assistant_message,
            )

        if decision.action == "revise_protocol":
            failed_payload = payload.model_copy(
                update={
                    "phase": "FAILED",
                    "assistant_message": decision.assistant_message,
                    "system_message": "DATA_COMPILATION_PROTOCOL_REVISION_REQUIRED",
                    "error_message": "user requested upstream protocol revision",
                }
            )
            return self._aborted_result(
                request=request,
                payload=failed_payload,
                user_message=decision.assistant_message,
            )

        locked_spec = payload.compiled_causal_spec
        if locked_spec is None or payload.compiled_dataset_summary is None:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message="The locked compilation context is incomplete and must be rebuilt.",
                error_message="locked compilation context missing",
            )
        if payload.source_dataset_id is None:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message="The stored source dataset reference is missing, so compilation must be rebuilt.",
                error_message="source dataset id missing from action-required payload",
            )

        try:
            if decision.action == "retry_transform":
                compiled_df = self._data_repo.get_csv_data(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataset_id=payload.compiled_dataset_id,
                    limit=1_000_000,
                )
                compiled_dataset = _CompiledDatasetArtifacts(
                    dataframe=compiled_df,
                    summary=payload.compiled_dataset_summary,
                    actions=[],
                    warnings=[],
                )
                transform_artifacts = self._build_validated_transform_plan(
                    protocol_discussion=protocol_discussion,
                    causal_spec=locked_spec,
                    compiled_dataset_summary=payload.compiled_dataset_summary,
                    repair_request=decision.repair_request,
                    validation_issues=payload.validation_issues,
                )
                compilation_actions = list(payload.compilation_actions)
                if decision.repair_request:
                    compilation_actions.append(
                        f"Revised the transformation plan after user feedback: {decision.repair_request}"
                    )
                compilation_warnings = list(payload.compilation_warnings)
                compilation_warnings.extend(transform_artifacts.warnings)
                return self._finalize_compilation_attempt(
                    request=request,
                    payload=payload,
                    source_dataset_id=payload.source_dataset_id,
                    protocol_discussion=protocol_discussion,
                    protocol_cleaning_instructions=protocol_cleaning_instructions,
                    causal_spec=locked_spec,
                    compiled_dataset=compiled_dataset,
                    transform_artifacts=transform_artifacts,
                    compilation_actions=compilation_actions,
                    compilation_warnings=compilation_warnings,
                    source_changed=False,
                )

            if decision.action == "retry_cleaning":
                repaired_source = self._repair_cleaned_source(
                    request=request,
                    source_df=source_df,
                    source_summary=source_summary,
                    compiled_dataset_summary=payload.compiled_dataset_summary,
                    protocol_discussion=protocol_discussion,
                    protocol_cleaning_instructions=protocol_cleaning_instructions,
                    causal_spec=locked_spec,
                    validation_issues=payload.validation_issues,
                    repair_request=decision.repair_request,
                )
                compiled_dataset = self._build_compiled_dataset(
                    dataframe=repaired_source.dataframe,
                    causal_spec=locked_spec,
                )
                transform_artifacts = self._build_validated_transform_plan(
                    protocol_discussion=protocol_discussion,
                    causal_spec=locked_spec,
                    compiled_dataset_summary=compiled_dataset.summary,
                    repair_request=decision.repair_request,
                    validation_issues=payload.validation_issues,
                )
                return self._finalize_compilation_attempt(
                    request=request,
                    payload=payload,
                    source_dataset_id=payload.source_dataset_id,
                    protocol_discussion=protocol_discussion,
                    protocol_cleaning_instructions=protocol_cleaning_instructions,
                    causal_spec=locked_spec,
                    compiled_dataset=compiled_dataset,
                    transform_artifacts=transform_artifacts,
                    compilation_actions=[
                        *list(payload.compilation_actions),
                        *repaired_source.actions,
                        *compiled_dataset.actions,
                    ],
                    compilation_warnings=[
                        *list(payload.compilation_warnings),
                        *repaired_source.warnings,
                        *compiled_dataset.warnings,
                        *transform_artifacts.warnings,
                    ],
                    source_changed=False,
                )

            revised_spec = self._revise_locked_causal_spec(
                protocol_discussion=protocol_discussion,
                locked_causal_spec=locked_spec,
                source_summary=source_summary,
                compiled_dataset_summary=payload.compiled_dataset_summary,
                validation_issues=payload.validation_issues,
                repair_request=decision.repair_request,
            )
            prepared_source = self._prepare_cleaned_source(
                request=request,
                source_df=source_df,
                source_summary=source_summary,
                protocol_discussion=protocol_discussion,
                protocol_cleaning_instructions=protocol_cleaning_instructions,
            )
            compiled_dataset = self._build_compiled_dataset(
                dataframe=prepared_source.dataframe,
                causal_spec=revised_spec,
            )
            transform_artifacts = self._build_validated_transform_plan(
                protocol_discussion=protocol_discussion,
                causal_spec=revised_spec,
                compiled_dataset_summary=compiled_dataset.summary,
                repair_request=decision.repair_request,
                validation_issues=payload.validation_issues,
            )
            revision_actions = list(payload.compilation_actions)
            if decision.repair_request:
                revision_actions.append(
                    f"Revised locked causal-spec details after user feedback: {decision.repair_request}"
                )
            revision_actions.extend(prepared_source.actions)
            revision_actions.extend(compiled_dataset.actions)
            return self._finalize_compilation_attempt(
                request=request,
                payload=payload,
                source_dataset_id=payload.source_dataset_id,
                protocol_discussion=protocol_discussion,
                protocol_cleaning_instructions=protocol_cleaning_instructions,
                causal_spec=revised_spec,
                compiled_dataset=compiled_dataset,
                transform_artifacts=transform_artifacts,
                compilation_actions=revision_actions,
                compilation_warnings=[
                    *list(payload.compilation_warnings),
                    *prepared_source.warnings,
                    *compiled_dataset.warnings,
                    *transform_artifacts.warnings,
                ],
                source_changed=False,
            )
        except Exception as exc:
            log.exception("data compilation action-required repair failed", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not apply that requested repair while keeping the locked "
                    "compilation columns unchanged."
                ),
                error_message=f"action-required repair failed: {safe_err(exc)}",
            )

    def _handle_review_response(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        latest_user_message: str,
        source_df: pd.DataFrame,
        source_summary: DatasetSummaryModel,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
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
                    "validation_status": payload.validation_status,
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in payload.validation_issues
                    ],
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
                    "validation_issues": payload.validation_issues,
                    "is_validated": True,
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
            action_payload = payload.model_copy(
                update={
                    "phase": "ACTION_REQUIRED",
                    "assistant_message": decision.assistant_message,
                    "system_message": "DATA_COMPILATION_ACTION_REQUIRED",
                    "error_message": None,
                }
            )
            return self._handle_action_required_response(
                request=request,
                payload=action_payload,
                latest_user_message=latest_user_message,
                source_df=source_df,
                source_summary=source_summary,
                protocol_discussion=protocol_discussion,
                protocol_cleaning_instructions=protocol_cleaning_instructions,
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
        compilation_actions: Sequence[str],
        compilation_warnings: Sequence[str],
        validation_status: ValidationStatus,
        validation_issues: Sequence[ValidationIssueModel],
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
                    "compilation_actions": list(compilation_actions),
                    "compilation_warnings": list(compilation_warnings),
                    "validation_status": validation_status,
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in validation_issues
                    ],
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="mini", temperature=0.2),
            history=history,
            max_attempts=2,
        )
        return review_summary.assistant_message

    def _build_first_pass_cleaning_instructions(
        self,
        *,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
    ) -> str:
        parts = [
            data_compilation_cleaning_instructions_prompt(),
            "",
            "Confirmed protocol discussion:",
            protocol_discussion.strip(),
        ]
        if protocol_cleaning_instructions:
            parts.extend(
                [
                    "",
                    "Confirmed protocol cleaning instructions:",
                    protocol_cleaning_instructions.strip(),
                ]
            )
        return "\n".join(parts).strip()

    def _build_repair_cleaning_instructions(
        self,
        *,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
        causal_spec: CausalSpec,
        compiled_dataset_summary: DatasetSummaryModel,
        validation_issues: Sequence[ValidationIssueModel],
        repair_request: str | None,
    ) -> str:
        parts = [
            data_compilation_discrepancy_repair_prompt(),
            "",
            "Confirmed protocol discussion:",
            protocol_discussion.strip(),
        ]
        if protocol_cleaning_instructions:
            parts.extend(
                [
                    "",
                    "Confirmed protocol cleaning instructions:",
                    protocol_cleaning_instructions.strip(),
                ]
            )
        parts.extend(
            [
                "",
                "Compiled causal specification:",
                json.dumps(causal_spec.model_dump(mode="json"), ensure_ascii=False),
                "",
                "Compiled dataset summary:",
                json.dumps(
                    _dataset_summary_prompt_payload(compiled_dataset_summary),
                    ensure_ascii=False,
                ),
                "",
                "Validation issues:",
                json.dumps(
                    [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in validation_issues
                    ],
                    ensure_ascii=False,
                ),
            ]
        )
        if repair_request:
            parts.extend(["", "User-requested repair direction:", repair_request.strip()])
        return "\n".join(parts).strip()

    def _revise_locked_causal_spec(
        self,
        *,
        protocol_discussion: str,
        locked_causal_spec: CausalSpec,
        source_summary: DatasetSummaryModel,
        compiled_dataset_summary: DatasetSummaryModel,
        validation_issues: Sequence[ValidationIssueModel],
        repair_request: str | None,
    ) -> CausalSpec:
        revised_spec = self._llm.generate_json(
            schema=self._causal_specs_tool.build_backdoor_schema(data_summary=source_summary),
            system_prompt=data_compilation_locked_spec_revision_prompt(),
            user_prompt=json.dumps(
                {
                    "confirmed_protocol_discussion": protocol_discussion,
                    "locked_compiled_causal_specification": locked_causal_spec.model_dump(
                        mode="json"
                    ),
                    "compiled_dataset_summary": _dataset_summary_prompt_payload(
                        compiled_dataset_summary
                    ),
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in validation_issues
                    ],
                    "repair_request": repair_request,
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="pro", temperature=0.1),
            history=None,
            max_attempts=3,
        )
        validated_spec = self._causal_specs_tool.post_validate_backdoor_spec(
            causal_spec=revised_spec,
            data_summary=source_summary,
        )
        return _enforce_locked_causal_spec_identity(
            locked_spec=locked_causal_spec,
            revised_spec=validated_spec,
        )

    def _run_data_manipulation_tool(
        self,
        *,
        dataframe: pd.DataFrame,
        conversation_id: UUID,
        data_summary: DatasetSummaryModel,
        instructions: str,
    ) -> pd.DataFrame:
        return self._data_manipulation_tool.manipulate(
            dataframe=dataframe,
            table_name=_conversation_id_to_table_name(conversation_id),
            data_summary=self._profiling_tool.dataset_summary_to_json(data_summary),
            instructions=instructions,
            retry_attempts=3,
        )

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
    compilation_actions: Sequence[str],
    compilation_warnings: Sequence[str],
    validation_status: ValidationStatus,
    validation_issues: Sequence[ValidationIssueModel],
) -> str:
    transform_lines = [
        f"{column.column}: {column.encoding.preset}" for column in transformation_plan.columns
    ]
    retained_columns = [str(profile.name).strip() for profile in compiled_dataset_summary.profiles]
    actions_text = "; ".join(compilation_actions) if compilation_actions else "None"
    warnings_text = "; ".join(compilation_warnings) if compilation_warnings else "None"
    validation_text = (
        "None"
        if not validation_issues
        else "; ".join(
            f"{issue.severity}: {issue.message}" for issue in validation_issues
        )
    )
    return (
        "I prepared the data for causal modeling by narrowing the dataset to the "
        "protocol-scope columns, compiling the baseline transformation plan, and "
        "running validation on the locked setup. "
        f"The compiled dataset has {compiled_dataset_summary.n_rows} rows and "
        f"{len(compiled_dataset_summary.profiles)} columns. "
        f"Retained columns: {', '.join(retained_columns) if retained_columns else 'None'}. "
        f"Treatment: {compiled_causal_spec.treatment_spec.column}. "
        f"Outcome: {compiled_causal_spec.outcome_spec.column}. "
        f"Covariates: {', '.join(compiled_causal_spec.covariates) if compiled_causal_spec.covariates else 'None'}. "
        f"Effect modifiers: {', '.join(compiled_causal_spec.effect_modifiers) if compiled_causal_spec.effect_modifiers else 'None'}. "
        f"Compilation actions: {actions_text}. "
        f"Warnings: {warnings_text}. "
        f"Validation status: {validation_status}. "
        f"Validation details: {validation_text}. "
        f"Planned baseline transformations: {'; '.join(transform_lines)}. "
        "Please confirm this compiled setup, or tell me exactly what should change."
    )


def _drop_rows_outside_binary_spec(
    *,
    dataframe: pd.DataFrame,
    column: str,
    allowed_values: set[str],
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    allowed_keys = {_normalize_discrete_value(value) for value in allowed_values}
    series = dataframe[column]
    normalized_series = series.map(_normalize_discrete_value)
    invalid_mask = ~normalized_series.isin(list(allowed_keys))
    if not bool(invalid_mask.any()):
        return dataframe, []

    invalid_counts: dict[str, int] = {}
    for raw_value in series[invalid_mask].tolist():
        key = _normalize_discrete_value(raw_value)
        key_text = _discrete_key_text(key)
        invalid_counts[key_text] = invalid_counts.get(key_text, 0) + 1

    filtered_df = dataframe.loc[~invalid_mask].copy()
    actions = [
        f"Dropped {int(invalid_mask.sum())} row(s) with {label} values outside the compiled specification for '{column}'.",
        f"Removed invalid {label} values: {', '.join(f'{value} ({count})' for value, count in sorted(invalid_counts.items()))}.",
    ]
    return filtered_df, actions


def _ensure_binary_treatment_arms_present(
    dataframe: pd.DataFrame,
    *,
    causal_spec: CausalSpec,
) -> None:
    treatment_col = str(causal_spec.treatment_spec.column)
    allowed_keys = {
        _normalize_discrete_value(causal_spec.treatment_spec.treated),
        _normalize_discrete_value(causal_spec.treatment_spec.control),
    }
    observed = {
        _normalize_discrete_value(value)
        for value in dataframe[treatment_col].dropna().tolist()
        if _normalize_discrete_value(value) in allowed_keys
    }
    if observed != allowed_keys:
        raise ValueError("compiled dataset must retain both treatment arms after filtering")


def _ensure_binary_outcome_classes_present(
    dataframe: pd.DataFrame,
    *,
    causal_spec: CausalSpec,
) -> None:
    if not isinstance(causal_spec.outcome_spec, BinaryOutcomeSpecModel):
        return
    outcome_col = str(causal_spec.outcome_spec.column)
    allowed_keys = {
        _normalize_discrete_value(causal_spec.outcome_spec.event),
        _normalize_discrete_value(causal_spec.outcome_spec.non_event),
    }
    observed = {
        _normalize_discrete_value(value)
        for value in dataframe[outcome_col].dropna().tolist()
        if _normalize_discrete_value(value) in allowed_keys
    }
    if observed != allowed_keys:
        raise ValueError("compiled dataset must retain both binary outcome classes after filtering")


def _summarize_summary_delta_actions(
    *,
    before_summary: DatasetSummaryModel,
    after_summary: DatasetSummaryModel,
    context: str,
) -> list[str]:
    actions = [context]
    if before_summary.n_rows != after_summary.n_rows:
        actions.append(
            f"Row count changed from {before_summary.n_rows} to {after_summary.n_rows}."
        )

    before_columns = _summary_column_names(before_summary)
    after_columns = _summary_column_names(after_summary)
    removed_columns = [column for column in before_columns if column not in after_columns]
    added_columns = [column for column in after_columns if column not in before_columns]
    if removed_columns:
        actions.append(f"Removed columns: {', '.join(removed_columns)}.")
    if added_columns:
        actions.append(f"Added columns: {', '.join(added_columns)}.")
    return actions


def _summary_column_names(summary: DatasetSummaryModel) -> list[str]:
    return [str(profile.name).strip() for profile in summary.profiles if str(profile.name).strip()]


def _split_transform_validation_issues(
    issues: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repairable_issues: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    for issue in issues:
        message = str(issue.get("message", "")).lower()
        if (
            "column type and preset incompatibilities" in message
            or "mapping contains values not supported" in message
            or "order contains values not supported" in message
        ):
            repairable_issues.append(issue)
            continue
        blocking_issues.append(issue)
    return repairable_issues, blocking_issues


def _format_transform_validation_issues(issues: Sequence[dict[str, Any]]) -> str:
    return "; ".join(
        f"{issue.get('path', 'root')}: {issue.get('message', 'Invalid value')}"
        for issue in issues
    )


def _encoding_validation_issue_to_validation_issue(
    issue: dict[str, Any],
) -> ValidationIssueModel:
    path = str(issue.get("path", "root")).strip() or "root"
    message = str(issue.get("message", "Invalid transform-plan value")).strip()
    return _fail_issue(
        message=f"Transform-plan validation failed at '{path}': {message}",
        evidence=issue,
        fix_hint=(
            "Revise the transform plan while keeping the same locked covariate and "
            "effect-modifier columns and roles."
        ),
    )


def _fail_issue(
    *,
    message: str,
    evidence: dict[str, Any],
    fix_hint: str | None,
) -> ValidationIssueModel:
    return ValidationIssueModel(
        severity="FAIL",
        message=message,
        evidence=evidence,
        fix_hint=fix_hint,
    )


def _validation_status(issues: Sequence[ValidationIssueModel]) -> ValidationStatus:
    if any(issue.severity == "FAIL" for issue in issues):
        return "FAIL"
    if any(issue.severity == "WARN" for issue in issues):
        return "WARN"
    return "PASS"


def _has_spec_breaking_issues(issues: Sequence[ValidationIssueModel]) -> bool:
    return any(_is_spec_breaking_issue(issue) for issue in issues if issue.severity == "FAIL")


def _is_spec_breaking_issue(issue: ValidationIssueModel) -> bool:
    message = issue.message.lower()
    fix_hint = (issue.fix_hint or "").lower()
    spec_breaking_markers = (
        "treatment and outcome columns must be different",
        "causal spec contains duplicate covariates",
        "causal spec contains duplicate effect modifiers",
        "covariates and effect modifiers overlap",
        "covariates and effect modifiers must not include treatment or outcome columns",
        "observational studies require covariate",
    )
    return any(marker in message or marker in fix_hint for marker in spec_breaking_markers)


def _build_action_required_message(
    *,
    causal_spec: CausalSpec,
    issues: Sequence[ValidationIssueModel],
) -> str:
    lines = [
        "Compilation finished, but validation found hard errors that still look repairable without changing the locked protocol columns.",
        "",
        f"Locked treatment column: {causal_spec.treatment_spec.column}",
        f"Locked outcome column: {causal_spec.outcome_spec.column}",
        f"Locked covariates: {', '.join(causal_spec.covariates) if causal_spec.covariates else 'None'}",
        f"Locked effect modifiers: {', '.join(causal_spec.effect_modifiers) if causal_spec.effect_modifiers else 'None'}",
        "",
        "Inside this step I can still:",
        "- revise covariate/effect-modifier encodings",
        "- rerun same-column cleaning or value normalization",
        "- revise same-column treatment or outcome literals/details",
        "",
        "Inside this step I cannot:",
        "- change treatment, outcome, covariate, or effect-modifier column identity or role",
        "",
        "Hard errors:",
    ]
    for issue in issues:
        if issue.severity != "FAIL":
            continue
        lines.append(f"- {issue.message}")
        if issue.fix_hint:
            lines.append(f"  What to fix: {issue.fix_hint}")
    lines.extend(
        [
            "",
            "Tell me whether to revise transform encodings, rerun same-column cleaning, revise locked treatment/outcome details, or go back to protocol discussion.",
        ]
    )
    return "\n".join(lines)


def _build_protocol_revision_required_message(
    *,
    causal_spec: CausalSpec,
    issues: Sequence[ValidationIssueModel],
) -> str:
    lines = [
        "Compilation and validation found blocking issues that cannot be fixed safely without changing the locked protocol columns or roles.",
        "",
        f"Locked treatment column: {causal_spec.treatment_spec.column}",
        f"Locked outcome column: {causal_spec.outcome_spec.column}",
        "",
        "Blocking issues:",
    ]
    for issue in issues:
        if issue.severity != "FAIL":
            continue
        lines.append(f"- {issue.message}")
        if issue.fix_hint:
            lines.append(f"  What to revise upstream: {issue.fix_hint}")
    lines.extend(
        [
            "",
            "Please revise the protocol discussion if you want to change treatment, outcome, covariate, or effect-modifier column choices or roles.",
        ]
    )
    return "\n".join(lines)


def _enforce_locked_causal_spec_identity(
    *,
    locked_spec: CausalSpec,
    revised_spec: CausalSpec,
) -> CausalSpec:
    if str(locked_spec.treatment_spec.column) != str(revised_spec.treatment_spec.column):
        raise ValueError("Locked treatment column cannot change during compilation repair")
    if str(locked_spec.outcome_spec.column) != str(revised_spec.outcome_spec.column):
        raise ValueError("Locked outcome column cannot change during compilation repair")
    if [str(column) for column in locked_spec.covariates] != [
        str(column) for column in revised_spec.covariates
    ]:
        raise ValueError("Locked covariate columns cannot change during compilation repair")
    if [str(column) for column in locked_spec.effect_modifiers] != [
        str(column) for column in revised_spec.effect_modifiers
    ]:
        raise ValueError(
            "Locked effect-modifier columns cannot change during compilation repair"
        )
    if locked_spec.experiment_type != revised_spec.experiment_type:
        raise ValueError("Experiment type cannot change during compilation repair")

    if type(locked_spec.outcome_spec) is not type(revised_spec.outcome_spec):
        raise ValueError("Outcome kind cannot change during compilation repair")

    if isinstance(locked_spec.outcome_spec, BinaryOutcomeSpecModel):
        if not isinstance(revised_spec.outcome_spec, BinaryOutcomeSpecModel):
            raise ValueError("Outcome kind cannot change during compilation repair")
    if isinstance(locked_spec.outcome_spec, ContinuousOutcomeSpecModel):
        if not isinstance(revised_spec.outcome_spec, ContinuousOutcomeSpecModel):
            raise ValueError("Outcome kind cannot change during compilation repair")

    return revised_spec


def _summarize_transform_plan_warnings(
    *,
    plan: TransformPlan,
    compiled_dataset_summary: DatasetSummaryModel,
) -> list[str]:
    profiles_by_name = {
        str(profile.name).strip(): profile for profile in compiled_dataset_summary.profiles
    }
    warnings: list[str] = []
    for column_plan in plan.columns:
        column = str(column_plan.column).strip()
        preset = str(column_plan.encoding.preset)
        profile = profiles_by_name.get(column)
        if profile is None:
            continue
        if preset == "drop":
            warnings.append(
                f"Baseline feature '{column}' is dropped from the transformation plan."
            )
            continue
        if (
            str(profile.inferred_kind) == "NUMERIC"
            and profile.distinct_count is not None
            and profile.distinct_count <= 20
            and preset in {"num_standard", "num_minmax", "num_log1p"}
        ):
            warnings.append(
                f"Numeric column '{column}' has low cardinality but remains numeric with preset '{preset}'. Review whether it should stay numeric."
            )
    return warnings


def _normalize_discrete_value(value: Any) -> tuple[str, Any]:
    try:
        if pd.isna(value):
            return ("na", None)
    except Exception:
        pass

    if isinstance(value, bool):
        return ("bool", bool(value))
    if isinstance(value, numbers.Real):
        return ("num", float(value))
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered == "true":
            return ("bool", True)
        if lowered == "false":
            return ("bool", False)
        try:
            return ("num", float(lowered))
        except ValueError:
            return ("str", lowered)
    return ("str", str(value).strip().lower())


def _discrete_key_text(key: tuple[str, Any]) -> str:
    return f"{key[0]}:{key[1]!r}"


def _conversation_id_to_table_name(conversation_id: UUID) -> str:
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).hexdigest()
    return f"df_{digest[:16]}"


__all__ = ["DataCompilationNode"]
