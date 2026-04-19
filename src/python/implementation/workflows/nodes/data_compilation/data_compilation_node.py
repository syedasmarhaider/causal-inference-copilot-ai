from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, cast
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.data_compilation.data_compilation_cleaning import (
    cleaning,
    compile_causal_spec_from_cleaned_summary,
    MissingnessDecisionList,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_deps import (
    DataCompilationDeps,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_node_info,
    data_compilation_review_query_prompt,
    data_compilation_review_decision_prompt,
    data_compilation_review_summary_prompt,
    data_compilation_transformation_retry_guidance_prompt,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationPayloadModel,
    DataCompilationState,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_transformation import (
    ColumnTransformationSuggestionList,
    transform,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_valiation import (
    DataCompilationValidationResult,
    validate_data_compilation,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
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
from python.implementation.workflows.utils.utils import safe_err

log = get_app_logger(__name__, component="data_compilation_node", log_type="node")


class _ReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class _ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["confirm", "recompile", "answer_query", "reject", "clarify"]
    assistant_message: str = Field(..., min_length=1)
    recompile_request: str | None = None


@dataclass(frozen=True)
class _CompiledArtifacts:
    dataset_id: UUID
    dataframe: pd.DataFrame
    summary: DatasetSummaryModel
    causal_spec: CausalSpec
    missingness_decisions: MissingnessDecisionList
    actions: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class _ValidationRepairGuidance:
    issue: str
    fix_hint: str | None


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
                    "The compilation stage is missing the active dataset, confirmed "
                    "protocol, or causal draft. Please complete those earlier steps first."
                ),
            )

        payload, source_changed = self._bind_payload_to_source(payload=payload, deps=deps)
        latest_user_message = _latest_user_message(request.read_only_messages_history)

        if payload.phase == "REVIEW_READY":
            if not self._review_payload_complete(payload):
                log.warning("data compilation review payload incomplete; recompiling")
            elif latest_user_message is None:
                return self._needs_input_result(
                    request=request,
                    payload=payload,
                    user_message=payload.assistant_message
                    or "Please review the compiled dataset and confirm or revise it.",
                )
            else:
                return self._handle_review_response(
                    request=request,
                    payload=payload,
                    latest_user_message=latest_user_message,
                )

        if payload.phase == "CONFIRMED" and not source_changed:
            return self._done_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "The compiled setup is already confirmed.",
            )

        if payload.phase == "FAILED" and not source_changed:
            return self._aborted_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or (
                    "The compilation step hit a hard blocker and needs upstream revision."
                    if payload.hard_failure
                    else "The compilation step is blocked and needs upstream revision."
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

        return self._run_pipeline_from_source(
            request=request,
            payload=payload,
            deps=deps,
            source_df=source_df,
            source_changed=source_changed,
        )

    def _bind_payload_to_source(
        self,
        *,
        payload: DataCompilationPayloadModel,
        deps: DataCompilationDeps,
    ) -> tuple[DataCompilationPayloadModel, bool]:
        if (
            payload.source_dataset_id == deps.dataset_id
            and payload.source_protocol_discussion == deps.protocol_discussion
            and payload.source_protocol_cleaning_instructions
            == deps.protocol_cleaning_instructions
            and _same_draft(payload.source_causal_spec_draft, deps.causal_spec_draft)
        ):
            return payload, False

        if (
            payload.source_dataset_id is None
            and payload.source_protocol_discussion is None
            and payload.source_protocol_cleaning_instructions is None
            and payload.source_causal_spec_draft is None
            and payload.phase == "INIT"
        ):
            return payload.bind_source(
                dataset_id=deps.dataset_id,
                protocol_discussion=deps.protocol_discussion,
                protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
                causal_spec_draft=deps.causal_spec_draft,
            ), False

        return payload.reset_for_recompile(
            dataset_id=deps.dataset_id,
            protocol_discussion=deps.protocol_discussion,
            protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
            causal_spec_draft=deps.causal_spec_draft,
        ), True

    def _run_pipeline_from_source(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        deps: DataCompilationDeps,
        source_df: pd.DataFrame,
        source_changed: bool,
        review_recompile_request: str | None = None,
    ) -> NodeExecutionResult:
        try:
            compiled_artifacts = self._compile_from_source(
                request=request,
                deps=deps,
                source_df=source_df,
                review_recompile_request=review_recompile_request,
            )
        except Exception as exc:
            log.exception("data compilation full compile failed", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=_build_compile_failure_message(
                    error=safe_err(exc),
                ),
                error_message=f"full compile failed: {safe_err(exc)}",
            )

        compiled_payload = payload.model_copy(
            update={
                "source_dataset_id": deps.dataset_id,
                "source_protocol_discussion": deps.protocol_discussion,
                "source_protocol_cleaning_instructions": deps.protocol_cleaning_instructions,
                "source_causal_spec_draft": deps.causal_spec_draft,
                "compiled_dataset_id": compiled_artifacts.dataset_id,
                "compiled_dataset_summary": compiled_artifacts.summary,
                "compiled_causal_spec": compiled_artifacts.causal_spec,
                "missingness_decisions": compiled_artifacts.missingness_decisions,
                "transformation_plan": None,
                "transformation_suggestions": None,
                "compilation_actions": compiled_artifacts.actions,
                "compilation_warnings": compiled_artifacts.warnings,
                "validation_issues": [],
                "validation_status": None,
                "phase": "INIT",
                "hard_failure": False,
                "assistant_message": None,
                "system_message": None,
                "error_message": None,
                "retry_feedback": None,
            }
        )

        return self._run_pipeline_from_compiled_dataset(
            request=request,
            payload=compiled_payload,
            deps=deps,
            compiled_artifacts=compiled_artifacts,
            source_changed=source_changed,
        )

    def _compile_from_source(
        self,
        *,
        request: NodeRequest,
        deps: DataCompilationDeps,
        source_df: pd.DataFrame,
        review_recompile_request: str | None = None,
    ) -> _CompiledArtifacts:
        cleaning_result = cleaning(
            protocol_discussion=deps.protocol_discussion,
            cleaning_instructions=self._build_cleaning_instructions(
                protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
            ),
            review_recompile_request=review_recompile_request,
            draft_causal_spec=deps.causal_spec_draft,
            data_summary=deps.dataset_summary,
            to_clean_df=source_df,
            datasetProfilingTool=self._profiling_tool,
            dataManipulationTool=self._data_manipulation_tool,
            llm=self._llm,
        )

        compiled_dataset_id = uuid4()
        self._data_repo.save_csv_data(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            dataset_id=compiled_dataset_id,
            df=cleaning_result.pd_cleaned,
            overwrite=True,
            include_index=False,
        )

        return _CompiledArtifacts(
            dataset_id=compiled_dataset_id,
            dataframe=cleaning_result.pd_cleaned,
            summary=cleaning_result.cleaned_data_summary,
            causal_spec=cleaning_result.causal,
            missingness_decisions=cleaning_result.missingness_decisions,
            actions=_summarize_compile_actions(
                source_summary=deps.dataset_summary,
                cleaned_summary=cleaning_result.cleaned_data_summary,
                causal_spec=cleaning_result.causal,
                missingness_decisions=cleaning_result.missingness_decisions,
                review_recompile_request=review_recompile_request,
            ),
            warnings=[],
        )

    def _run_pipeline_from_compiled_dataset(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        deps: DataCompilationDeps,
        compiled_artifacts: _CompiledArtifacts,
        source_changed: bool,
    ) -> NodeExecutionResult:
        try:
            transformation_result = transform(
                transformation_instructions=self._build_transformation_instructions(
                    protocol_discussion=deps.protocol_discussion,
                    protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
                    retry_feedback=payload.retry_feedback,
                ),
                causal_spec=compiled_artifacts.causal_spec,
                data_summary=compiled_artifacts.summary,
                llm=self._llm,
            )
        except Exception as exc:
            log.exception("data compilation transformation failed", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I cleaned and compiled the dataset, but I could not build a safe "
                    f"baseline transformation plan. Error: {safe_err(exc)}"
                ),
                error_message=f"transformation failed: {safe_err(exc)}",
            )

        if transformation_result.transformation_plan is None:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not produce a usable baseline transformation plan for the "
                    "compiled dataset."
                ),
                error_message="transformation plan missing after successful transform stage",
            )

        transformation_suggestion_warnings = _summarize_transformation_suggestions(
            summary=compiled_artifacts.summary,
            transformation_suggestions=transformation_result.transformation_suggestions,
        )
        validation_result = validate_data_compilation(
            candidate_df=compiled_artifacts.dataframe,
            causal_spec=compiled_artifacts.causal_spec,
            transform_plan=transformation_result.transformation_plan,
        )
        validation_status = _validation_status(validation_result.validation_errors)
        validated_payload = payload.model_copy(
            update={
                "compiled_causal_spec": compiled_artifacts.causal_spec,
                "transformation_plan": transformation_result.transformation_plan,
                "transformation_suggestions": transformation_result.transformation_suggestions,
                "compilation_warnings": _merge_unique_text_items(
                    payload.compilation_warnings,
                    transformation_suggestion_warnings,
                ),
                "validation_issues": validation_result.validation_errors,
                "validation_status": validation_status,
            }
        )

        if validation_status == "FAIL":
            return self._handle_validation_failure(
                request=request,
                payload=validated_payload,
                deps=deps,
                compiled_artifacts=compiled_artifacts,
                validation_result=validation_result,
                source_changed=source_changed,
            )

        return self._build_review_ready_result(
            request=request,
            payload=validated_payload,
            compiled_artifacts=compiled_artifacts,
            transformation_plan=transformation_result.transformation_plan,
            transformation_suggestions=transformation_result.transformation_suggestions,
            validation_result=validation_result,
            source_changed=source_changed,
        )

    def _handle_validation_failure(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        deps: DataCompilationDeps,
        compiled_artifacts: _CompiledArtifacts,
        validation_result: DataCompilationValidationResult,
        source_changed: bool,
    ) -> NodeExecutionResult:
        if validation_result.user_suggestion_message is None:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=_build_hard_validation_message(
                    issues=validation_result.validation_errors
                ),
                error_message="hard validation failure without repairable guidance",
            )

        if payload.validation_retry_count >= 1:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=_build_validation_retry_exhausted_message(
                    validation_message=validation_result.user_suggestion_message
                ),
                error_message="repairable validation failure persisted after automatic retry",
            )

        retry_feedback = _build_validation_retry_feedback(
            validation_result.user_suggestion_message
        )
        retry_payload = payload.model_copy(
            update={
                "validation_retry_count": payload.validation_retry_count + 1,
                "retry_feedback": retry_feedback,
                "compilation_actions": [
                    *payload.compilation_actions,
                    "Applied one automatic retry after repairable validation feedback.",
                ],
                "compilation_warnings": [
                    *payload.compilation_warnings,
                    "A repairable validation issue triggered one automatic retry on the already cleaned compiled dataset.",
                ],
            }
        )
        return self._retry_from_compiled_dataset(
            request=request,
            payload=retry_payload,
            deps=deps,
            compiled_artifacts=compiled_artifacts,
            source_changed=source_changed,
            retry_feedback=retry_feedback,
        )

    def _retry_from_compiled_dataset(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        deps: DataCompilationDeps,
        compiled_artifacts: _CompiledArtifacts,
        source_changed: bool,
        retry_feedback: str,
    ) -> NodeExecutionResult:
        try:
            revised_causal_spec = compile_causal_spec_from_cleaned_summary(
                llm=self._llm,
                cleaned_summary=compiled_artifacts.summary,
                draft_causal_spec=deps.causal_spec_draft,
                protocol_discussion=deps.protocol_discussion,
                retry_feedback=retry_feedback,
            )
        except Exception as exc:
            log.exception(
                "data compilation validation retry causal spec recompilation failed",
                error=safe_err(exc),
            )
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "I tried one automatic repair pass after validation, but I could not "
                    f"recompile a consistent causal specification. Error: {safe_err(exc)}"
                ),
                error_message=f"validation retry causal spec recompilation failed: {safe_err(exc)}",
            )

        retried_artifacts = _CompiledArtifacts(
            dataset_id=compiled_artifacts.dataset_id,
            dataframe=compiled_artifacts.dataframe,
            summary=compiled_artifacts.summary,
            causal_spec=revised_causal_spec,
            missingness_decisions=compiled_artifacts.missingness_decisions,
            actions=payload.compilation_actions,
            warnings=payload.compilation_warnings,
        )
        retry_payload = payload.model_copy(update={"compiled_causal_spec": revised_causal_spec})
        return self._run_pipeline_from_compiled_dataset(
            request=request,
            payload=retry_payload,
            deps=deps,
            compiled_artifacts=retried_artifacts,
            source_changed=source_changed,
        )

    def _build_review_ready_result(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        compiled_artifacts: _CompiledArtifacts,
        transformation_plan: TransformPlan,
        transformation_suggestions: ColumnTransformationSuggestionList | None,
        validation_result: DataCompilationValidationResult,
        source_changed: bool,
    ) -> NodeExecutionResult:
        validation_status = payload.validation_status or _validation_status(
            validation_result.validation_errors
        )
        try:
            review_message = self._build_review_summary_message(
                protocol_discussion=payload.source_protocol_discussion
                or request.orchestrator_state.get("protocol_discussion")
                or "",
                compiled_causal_spec=compiled_artifacts.causal_spec,
                compiled_dataset_summary=compiled_artifacts.summary,
                missingness_decisions=compiled_artifacts.missingness_decisions,
                transformation_plan=transformation_plan,
                transformation_suggestions=transformation_suggestions,
                compilation_actions=payload.compilation_actions,
                compilation_warnings=payload.compilation_warnings,
                validation_status=validation_status,
                validation_issues=validation_result.validation_errors,
                messages_history=request.read_only_messages_history,
            )
        except Exception as exc:
            log.exception("data compilation review summary failed", error=safe_err(exc))
            review_message = _build_review_summary_fallback(
                compiled_dataset_summary=compiled_artifacts.summary,
                compiled_causal_spec=compiled_artifacts.causal_spec,
                missingness_decisions=compiled_artifacts.missingness_decisions,
                transformation_plan=transformation_plan,
                transformation_suggestions=transformation_suggestions,
                compilation_actions=payload.compilation_actions,
                compilation_warnings=payload.compilation_warnings,
                validation_status=validation_status,
                validation_issues=validation_result.validation_errors,
            )

        if source_changed:
            review_message = (
                "The active dataset or confirmed protocol changed, so I recompiled the "
                f"setup before this review. {review_message}"
            )

        review_payload = payload.model_copy(
            update={
                "compiled_causal_spec": compiled_artifacts.causal_spec,
                "compiled_dataset_summary": compiled_artifacts.summary,
                "missingness_decisions": compiled_artifacts.missingness_decisions,
                "transformation_plan": transformation_plan,
                "transformation_suggestions": transformation_suggestions,
                "validation_issues": validation_result.validation_errors,
                "validation_status": validation_status,
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
                    "The stored compilation review state is incomplete, so this step "
                    "needs to be recompiled from the latest dataset and confirmed protocol."
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
                    "missingness_decisions": payload.missingness_decisions.model_dump(
                        mode="json"
                    ),
                    "transformation_plan": payload.transformation_plan.model_dump(
                        mode="json"
                    ),
                    "transformation_suggestions": payload.transformation_suggestions.model_dump(
                        mode="json"
                    ),
                    "compilation_actions": list(payload.compilation_actions),
                    "compilation_warnings": list(payload.compilation_warnings),
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
                    "causal_spec_draft": payload.source_causal_spec_draft,
                    "causal_spec": payload.compiled_causal_spec,
                    "data_transformation_plan": payload.transformation_plan,
                    "working_dataset_frozen": True,
                    "validation_issues": payload.validation_issues,
                    "is_validated": True,
                },
            )
            confirmed_payload = payload.model_copy(
                update={
                "phase": "CONFIRMED",
                "hard_failure": False,
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

        if decision.action == "answer_query":
            try:
                answer_message = self._build_review_query_answer_message(
                    protocol_discussion=payload.source_protocol_discussion
                    or request.orchestrator_state.get("protocol_discussion")
                    or "",
                    compiled_causal_spec=payload.compiled_causal_spec,
                    compiled_dataset_summary=payload.compiled_dataset_summary,
                    missingness_decisions=payload.missingness_decisions,
                    transformation_plan=payload.transformation_plan,
                    transformation_suggestions=payload.transformation_suggestions,
                    compilation_actions=payload.compilation_actions,
                    compilation_warnings=payload.compilation_warnings,
                    validation_status=payload.validation_status or "WARN",
                    validation_issues=payload.validation_issues,
                    latest_user_message=latest_user_message,
                    messages_history=request.read_only_messages_history,
                )
            except Exception as exc:
                log.exception("data compilation review query answer failed", error=safe_err(exc))
                answer_message = _build_review_query_answer_fallback(
                    latest_user_message=latest_user_message,
                    compiled_dataset_summary=payload.compiled_dataset_summary,
                    compiled_causal_spec=payload.compiled_causal_spec,
                    missingness_decisions=payload.missingness_decisions,
                    transformation_plan=payload.transformation_plan,
                    compilation_actions=payload.compilation_actions,
                    validation_status=payload.validation_status or "WARN",
                )
            answered_payload = payload.model_copy(
                update={
                    "assistant_message": answer_message,
                    "hard_failure": False,
                    "system_message": None,
                    "error_message": None,
                }
            )
            return self._needs_input_result(
                request=request,
                payload=answered_payload,
                user_message=answer_message,
            )

        if decision.action == "recompile":
            normalized_recompile_request = _normalize_text(decision.recompile_request)
            if not normalized_recompile_request:
                clarified_payload = payload.model_copy(
                    update={
                        "assistant_message": (
                            "I understood that you want changes before accepting, but I "
                            "still need one clear sentence describing the same-column "
                            "cleaning or preprocessing change to apply."
                        ),
                        "hard_failure": False,
                        "system_message": None,
                        "error_message": None,
                    }
                )
                return self._needs_input_result(
                    request=request,
                    payload=clarified_payload,
                    user_message=clarified_payload.assistant_message or "",
                )

            try:
                deps = DataCompilationDeps.from_request(request)
                source_dataset_id = payload.source_dataset_id or deps.dataset_id
                source_df = self._data_repo.get_csv_data(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataset_id=source_dataset_id,
                    limit=1_000_000,
                )
            except Exception as exc:
                log.exception(
                    "failed to reload original source dataset for recompilation",
                    error=safe_err(exc),
                )
                return self._failed_result(
                    request=request,
                    payload=payload,
                    user_message=(
                        "I understood the requested same-column changes, but I could not "
                        "reload the original source dataset to recompile from scratch. "
                        f"Error: {safe_err(exc)}"
                    ),
                    error_message=(
                        "review-time recompilation source reload failed: "
                        f"{safe_err(exc)}"
                    ),
                )

            recompiling_payload = payload.reset_for_recompile(
                dataset_id=deps.dataset_id,
                protocol_discussion=deps.protocol_discussion,
                protocol_cleaning_instructions=deps.protocol_cleaning_instructions,
                causal_spec_draft=deps.causal_spec_draft,
            )
            return self._run_pipeline_from_source(
                request=request,
                payload=recompiling_payload,
                deps=deps,
                source_df=source_df,
                source_changed=False,
                review_recompile_request=normalized_recompile_request,
            )

        if decision.action == "reject":
            revised_payload = payload.model_copy(
                update={
                    "phase": "FAILED",
                    "hard_failure": False,
                    "assistant_message": decision.assistant_message,
                    "system_message": "DATA_COMPILATION_REVISION_REQUESTED",
                    "error_message": "user requested revision after review",
                }
            )
            return self._aborted_result(
                request=request,
                payload=revised_payload,
                user_message=decision.assistant_message,
            )

        clarified_payload = payload.model_copy(
            update={
                "assistant_message": decision.assistant_message,
                "hard_failure": False,
                "system_message": None,
                "error_message": None,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=clarified_payload,
            user_message=decision.assistant_message,
        )

    def _build_review_summary_message(
        self,
        *,
        protocol_discussion: str,
        compiled_causal_spec: CausalSpec,
        compiled_dataset_summary: DatasetSummaryModel,
        missingness_decisions: MissingnessDecisionList,
        transformation_plan: TransformPlan,
        transformation_suggestions: ColumnTransformationSuggestionList | None,
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
                    "compiled_dataset_summary": {
                        "n_rows": compiled_dataset_summary.n_rows,
                        "columns": [
                            str(profile.name).strip()
                            for profile in compiled_dataset_summary.profiles
                        ],
                    },
                    "missingness_decisions": missingness_decisions.model_dump(mode="json"),
                    "transformation_plan": transformation_plan.model_dump(mode="json"),
                    "transformation_suggestions": (
                        transformation_suggestions.model_dump(mode="json")
                        if transformation_suggestions is not None
                        else {"suggestions": []}
                    ),
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

    def _build_review_query_answer_message(
        self,
        *,
        protocol_discussion: str,
        compiled_causal_spec: CausalSpec,
        compiled_dataset_summary: DatasetSummaryModel,
        missingness_decisions: MissingnessDecisionList,
        transformation_plan: TransformPlan,
        transformation_suggestions: ColumnTransformationSuggestionList | None,
        compilation_actions: Sequence[str],
        compilation_warnings: Sequence[str],
        validation_status: ValidationStatus,
        validation_issues: Sequence[ValidationIssueModel],
        latest_user_message: str,
        messages_history: Sequence[ChatMessage] | None,
    ) -> str:
        history = list(messages_history[-4:]) if messages_history else None
        review_answer = self._llm.generate_json(
            schema=_ReviewSummary,
            system_prompt=data_compilation_review_query_prompt(),
            user_prompt=json.dumps(
                {
                    "protocol_discussion": protocol_discussion,
                    "compiled_causal_spec": compiled_causal_spec.model_dump(mode="json"),
                    "compiled_dataset_summary": {
                        "n_rows": compiled_dataset_summary.n_rows,
                        "columns": [
                            str(profile.name).strip()
                            for profile in compiled_dataset_summary.profiles
                        ],
                    },
                    "missingness_decisions": missingness_decisions.model_dump(mode="json"),
                    "transformation_plan": transformation_plan.model_dump(mode="json"),
                    "transformation_suggestions": (
                        transformation_suggestions.model_dump(mode="json")
                        if transformation_suggestions is not None
                        else {"suggestions": []}
                    ),
                    "compilation_actions": list(compilation_actions),
                    "compilation_warnings": list(compilation_warnings),
                    "validation_status": validation_status,
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in validation_issues
                    ],
                    "latest_user_message": latest_user_message,
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="mini", temperature=0.2),
            history=history,
            max_attempts=2,
        )
        return review_answer.assistant_message

    def _review_payload_complete(self, payload: DataCompilationPayloadModel) -> bool:
        return (
            payload.source_causal_spec_draft is not None
            and payload.compiled_dataset_id is not None
            and payload.compiled_dataset_summary is not None
            and payload.compiled_causal_spec is not None
            and payload.missingness_decisions is not None
            and payload.transformation_plan is not None
            and payload.transformation_suggestions is not None
            and payload.validation_status is not None
        )

    def _build_cleaning_instructions(
        self,
        *,
        protocol_cleaning_instructions: str | None,
    ) -> str:
        normalized_protocol_instructions = _normalize_text(protocol_cleaning_instructions)
        return normalized_protocol_instructions

    def _build_transformation_instructions(
        self,
        *,
        protocol_discussion: str,
        protocol_cleaning_instructions: str | None,
        retry_feedback: str | None,
    ) -> str:
        parts = [
            "Confirmed protocol discussion:",
            protocol_discussion.strip(),
        ]
        normalized_protocol_instructions = _normalize_text(protocol_cleaning_instructions)
        normalized_retry_feedback = _normalize_text(retry_feedback)
        if normalized_protocol_instructions:
            parts.extend(
                [
                    "",
                    "Confirmed protocol cleaning instructions:",
                    normalized_protocol_instructions,
                ]
            )
        if normalized_retry_feedback:
            parts.extend(
                [
                    "",
                    data_compilation_transformation_retry_guidance_prompt(),
                    normalized_retry_feedback,
                ]
            )
        return "\n".join(parts).strip()

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
                "hard_failure": True,
                "assistant_message": user_message,
                "system_message": "DATA_COMPILATION_HARD_FAILED",
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


def _same_draft(left: CausalSpecDraft | None, right: CausalSpecDraft | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _normalize_text(raw: str | None) -> str:
    if raw is None:
        return ""
    return raw.strip()


def _protocol_scope_columns(causal_spec: CausalSpec) -> list[str]:
    ordered_columns = [
        str(causal_spec.id_col).strip(),
        str(causal_spec.treatment_spec.column).strip(),
        str(causal_spec.outcome_spec.column).strip(),
        *(str(column).strip() for column in causal_spec.covariates),
        *(str(column).strip() for column in causal_spec.effect_modifiers),
    ]
    deduped: list[str] = []
    for column in ordered_columns:
        if column and column not in deduped:
            deduped.append(column)
    return deduped


def _summary_column_names(summary: DatasetSummaryModel) -> list[str]:
    return [str(profile.name).strip() for profile in summary.profiles if str(profile.name).strip()]


def _summarize_compile_actions(
    *,
    source_summary: DatasetSummaryModel,
    cleaned_summary: DatasetSummaryModel,
    causal_spec: CausalSpec,
    missingness_decisions: MissingnessDecisionList,
    review_recompile_request: str | None,
) -> list[str]:
    actions = [
        "Retained only the confirmed protocol-scope columns for compilation: "
        + ", ".join(_protocol_scope_columns(causal_spec))
    ]
    actions.extend(_summarize_missingness_decisions(missingness_decisions))
    if source_summary.n_rows != cleaned_summary.n_rows:
        actions.append(
            f"Row count changed from {source_summary.n_rows} to {cleaned_summary.n_rows} during cleaning."
        )

    source_columns = _summary_column_names(source_summary)
    cleaned_columns = _summary_column_names(cleaned_summary)
    removed_columns = [column for column in source_columns if column not in cleaned_columns]
    if removed_columns:
        actions.append(
            "Removed columns outside the confirmed draft scope: "
            + ", ".join(removed_columns)
        )

    normalized_review_recompile_request = _normalize_text(review_recompile_request)
    if normalized_review_recompile_request:
        actions.append(
            "Applied a review-time recompilation request on the original working dataset: "
            + normalized_review_recompile_request
        )

    actions.append("Recompiled the causal specification on the cleaned dataset summary.")
    return actions


def _summarize_missingness_decisions(
    missingness_decisions: MissingnessDecisionList,
) -> list[str]:
    affected = [
        decision
        for decision in missingness_decisions.decisions
        if decision.missing_count_before > 0
    ]
    if not affected:
        return ["No protocol-scope missingness was detected before cleaning."]

    return [
        (
            f"Resolved missingness for '{decision.column}' ({decision.role}) via "
            f"{decision.resolution}; before={decision.missing_count_before}, "
            f"after={decision.missing_count_after}."
        )
        for decision in affected
    ]


def _merge_unique_text_items(
    existing_items: Sequence[str],
    new_items: Sequence[str],
) -> list[str]:
    merged = list(existing_items)
    for item in new_items:
        if item not in merged:
            merged.append(item)
    return merged


def _summarize_transformation_suggestions(
    *,
    summary: DatasetSummaryModel,
    transformation_suggestions: ColumnTransformationSuggestionList | None,
) -> list[str]:
    if transformation_suggestions is None:
        return []

    current_kind_by_column = {
        str(profile.name).strip(): str(profile.inferred_kind)
        for profile in summary.profiles
        if str(profile.name).strip()
    }
    warnings: list[str] = []
    for suggestion in transformation_suggestions.suggestions:
        current_kind = current_kind_by_column.get(suggestion.column)
        if current_kind is None or suggestion.preferred_type == current_kind:
            continue
        warnings.append(
            f"Column '{suggestion.column}' is currently stored as {current_kind}; "
            f"preferred future raw type is {suggestion.preferred_type}. "
            f"{suggestion.preferred_type_reason}"
        )
    return warnings


def _build_validation_retry_feedback(validation_message: str) -> str:
    return "\n\n".join(
        [
            data_compilation_transformation_retry_guidance_prompt(),
            validation_message.strip(),
        ]
    ).strip()


def _validation_status(issues: Sequence[ValidationIssueModel]) -> ValidationStatus:
    if any(issue.severity == "FAIL" for issue in issues):
        return "FAIL"
    if any(issue.severity == "WARN" for issue in issues):
        return "WARN"
    return "PASS"


def _format_issue_lines(issues: Sequence[ValidationIssueModel]) -> list[str]:
    lines: list[str] = []
    for issue in issues:
        lines.append(f"- {issue.severity}: {issue.message}")
        if issue.fix_hint:
            lines.append(f"  Suggested fix: {issue.fix_hint}")
    return lines


def _build_compile_failure_message(
    *,
    error: str,
) -> str:
    return (
        "I could not clean and compile the current dataset into a stable causal setup. "
        f"Error: {error}"
    )


def _build_validation_retry_exhausted_message(
    *,
    validation_message: str,
) -> str:
    guidance = _parse_validation_retry_guidance(validation_message)
    if not guidance:
        return "\n".join(
            [
                "I applied one automatic repair retry after validation, but the compiled setup still has a remaining transformation or encoding problem.",
                "",
                "Please adjust the dataset preprocessing or encoding choices and rerun compilation. You only need to revise the protocol if you want different variables or roles.",
            ]
        ).strip()

    if len(guidance) == 1:
        item = guidance[0]
        lines = [
            "I applied one automatic repair retry after validation, but one transformation or encoding issue still remains.",
            "",
            f"Remaining issue: {item.issue}",
        ]
        if item.fix_hint:
            lines.extend(["", f"Most direct fix: {item.fix_hint}"])
        lines.extend(
            [
                "",
                "Please keep the same locked treatment, outcome, covariate, and effect-modifier roles, adjust the preprocessing or encoding accordingly, and rerun compilation.",
            ]
        )
        return "\n".join(lines).strip()

    lines = [
        "I applied one automatic repair retry after validation, but a few transformation or encoding issues still remain.",
        "",
        "Remaining issues:",
    ]
    for item in guidance:
        lines.append(f"- {item.issue}")
        if item.fix_hint:
            lines.append(f"  Most direct fix: {item.fix_hint}")
    lines.extend(
        [
            "",
            "Please keep the same locked treatment, outcome, covariate, and effect-modifier roles, adjust the preprocessing or encoding accordingly, and rerun compilation.",
        ]
    )
    return "\n".join(lines).strip()


def _build_hard_validation_message(
    *,
    issues: Sequence[ValidationIssueModel],
) -> str:
    lines = [
        "Validation found blocking problems that cannot be repaired automatically in this step.",
        "",
        "Blocking issues:",
        *_format_issue_lines([issue for issue in issues if issue.severity == "FAIL"]),
    ]
    lines.extend(
        [
            "",
            "Please revise the protocol discussion or the upstream dataset preparation before trying again.",
        ]
    )
    return "\n".join(lines).strip()


def _build_review_summary_fallback(
    *,
    compiled_dataset_summary: DatasetSummaryModel,
    compiled_causal_spec: CausalSpec,
    missingness_decisions: MissingnessDecisionList,
    transformation_plan: TransformPlan,
    transformation_suggestions: ColumnTransformationSuggestionList | None,
    compilation_actions: Sequence[str],
    compilation_warnings: Sequence[str],
    validation_status: ValidationStatus,
    validation_issues: Sequence[ValidationIssueModel],
) -> str:
    retained_columns = [str(profile.name).strip() for profile in compiled_dataset_summary.profiles]
    preferred_type_notes = _preferred_type_note_by_column(
        summary=compiled_dataset_summary,
        transformation_suggestions=transformation_suggestions,
    )
    transform_lines = []
    for column in transformation_plan.columns:
        line = f"{column.column}: {column.encoding.preset}"
        note = preferred_type_notes.get(column.column)
        if note:
            line = f"{line} ({note})"
        transform_lines.append(line)
    missingness_text = "; ".join(_summarize_missingness_decisions(missingness_decisions))
    warning_text = (
        "No non-blocking warnings remain."
        if not compilation_warnings and not validation_issues
        else "; ".join(
            [
                *list(compilation_warnings),
                *[
                    f"{issue.severity}: {issue.message}"
                    for issue in validation_issues
                    if issue.severity == "WARN"
                ],
            ]
        )
    )
    recommendation = (
        "I recommend accepting this setup now."
        if validation_status == "PASS"
        else "I think this setup is usable, but only if you are comfortable with the cautions listed below."
    )
    return (
        "I prepared the dataset for causal modeling by narrowing it to the confirmed "
        "protocol columns and checking that the cleaned data still matches the agreed "
        "clinical question. "
        f"The compiled dataset now has {compiled_dataset_summary.n_rows} rows and "
        f"{len(compiled_dataset_summary.profiles)} columns: "
        f"{', '.join(retained_columns) if retained_columns else 'none'}. "
        f"Missingness handling: {missingness_text}. "
        f"The treatment column is {compiled_causal_spec.treatment_spec.column}, the "
        f"outcome column is {compiled_causal_spec.outcome_spec.column}, the baseline "
        f"covariates are {', '.join(compiled_causal_spec.covariates) if compiled_causal_spec.covariates else 'none'}, "
        f"and the effect modifiers are {', '.join(compiled_causal_spec.effect_modifiers) if compiled_causal_spec.effect_modifiers else 'none'}. "
        f"Data preparation steps: {'; '.join(compilation_actions) if compilation_actions else 'none recorded'}. "
        f"Planned baseline transformations: {'; '.join(transform_lines)}. "
        f"Validation status: {validation_status}. "
        f"Warnings and cautions: {warning_text}. "
        f"{recommendation} Please confirm this compiled setup, or tell me what should change."
    )


def _build_review_query_answer_fallback(
    *,
    latest_user_message: str,
    compiled_dataset_summary: DatasetSummaryModel,
    compiled_causal_spec: CausalSpec,
    missingness_decisions: MissingnessDecisionList,
    transformation_plan: TransformPlan,
    compilation_actions: Sequence[str],
    validation_status: ValidationStatus,
) -> str:
    retained_columns = [
        str(profile.name).strip() for profile in compiled_dataset_summary.profiles
    ]
    transform_text = "; ".join(
        f"{column.column}: {column.encoding.preset}" for column in transformation_plan.columns
    )
    missingness_text = "; ".join(_summarize_missingness_decisions(missingness_decisions))
    return (
        f"Your question was: {latest_user_message.strip()} "
        f"The compiled dataset currently has {compiled_dataset_summary.n_rows} rows and "
        f"{len(retained_columns)} columns: {', '.join(retained_columns) if retained_columns else 'none'}. "
        f"Treatment is {compiled_causal_spec.treatment_spec.column}, outcome is "
        f"{compiled_causal_spec.outcome_spec.column}, covariates are "
        f"{', '.join(compiled_causal_spec.covariates) if compiled_causal_spec.covariates else 'none'}, "
        f"and effect modifiers are "
        f"{', '.join(compiled_causal_spec.effect_modifiers) if compiled_causal_spec.effect_modifiers else 'none'}. "
        f"Missingness handling: {missingness_text}. "
        f"Planned transformations: {transform_text}. "
        f"Data preparation steps: {'; '.join(compilation_actions) if compilation_actions else 'none recorded'}. "
        f"Validation status is {validation_status}. Please confirm this setup or tell me what should change."
    )


def _preferred_type_note_by_column(
    *,
    summary: DatasetSummaryModel,
    transformation_suggestions: ColumnTransformationSuggestionList | None,
) -> dict[str, str]:
    if transformation_suggestions is None:
        return {}

    current_kind_by_column = {
        str(profile.name).strip(): str(profile.inferred_kind)
        for profile in summary.profiles
        if str(profile.name).strip()
    }
    notes: dict[str, str] = {}
    for suggestion in transformation_suggestions.suggestions:
        current_kind = current_kind_by_column.get(suggestion.column)
        if current_kind is None or suggestion.preferred_type == current_kind:
            continue
        notes[suggestion.column] = (
            f"preferred future raw type {suggestion.preferred_type}: "
            f"{suggestion.preferred_type_reason}"
        )
    return notes


def _parse_validation_retry_guidance(
    validation_message: str,
) -> list[_ValidationRepairGuidance]:
    guidance: list[_ValidationRepairGuidance] = []
    current_issue: str | None = None
    current_fix_hint: str | None = None
    in_repair_section = False

    for raw_line in validation_message.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "Repairable validation errors:":
            in_repair_section = True
            continue

        if not in_repair_section:
            continue

        if line.startswith("- "):
            if current_issue is not None:
                guidance.append(
                    _ValidationRepairGuidance(
                        issue=current_issue,
                        fix_hint=current_fix_hint,
                    )
                )
            current_issue = line[2:].strip()
            current_fix_hint = None
            continue

        if line.startswith("What to fix:") and current_issue is not None:
            current_fix_hint = line.removeprefix("What to fix:").strip() or None

    if current_issue is not None:
        guidance.append(
            _ValidationRepairGuidance(
                issue=current_issue,
                fix_hint=current_fix_hint,
            )
        )

    return guidance
