from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.data_compilation.data_compilation_cleaning import clean
from python.implementation.workflows.nodes.data_compilation.data_compilation_deps import (
    DataCompilationDeps,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_cannot_confirm_message_prompt,
    data_compilation_clarify_fallback_message_prompt,
    data_compilation_hard_failure_message_prompt,
    data_compilation_message_generation_repair_prompt,
    data_compilation_node_info,
    data_compilation_review_decision_prompt,
    data_compilation_review_query_prompt,
    data_compilation_review_summary_prompt,
    data_compilation_transformation_retry_guidance_prompt,
    data_compilation_validation_failure_message_prompt,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationPayloadModel,
    DataCompilationState,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_transformation import (
    transform,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_valiation import (
    validate_data_compilation,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
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
        latest_user_message = _latest_user_message(request.read_only_messages_history)

        try:
            deps = DataCompilationDeps.from_request(request)
        except Exception as exc:
            error = safe_err(exc)
            log.exception("data compilation dependencies missing", error=error)
            message = self._generate_user_message(
                "hard_failure",
                {
                    "step": "dependency loading",
                    "error": error,
                    "hard_failure": True,
                    "payload": payload,
                    "messages_history": request.read_only_messages_history,
                },
                latest_user_message=latest_user_message,
            )
            blocked_payload = payload.model_copy(
                update={
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": f"dependency loading failed: {error}",
                    "last_handled_user_message_fingerprint": (
                        latest_user_message["fingerprint"] if latest_user_message else None
                    ),
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        source_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "dataset_id": str(deps.dataset_id),
                    "dataset_summary": deps.dataset_summary.model_dump(mode="json"),
                    "causal_spec_draft": deps.causal_spec_draft.model_dump(mode="json"),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        active_dataset_id = deps.dataset_id
        active_is_source = (
            payload.source_dataset_id is not None
            and active_dataset_id == payload.source_dataset_id
        )
        active_is_preview = (
            payload.compiled_dataset_id is not None
            and active_dataset_id == payload.compiled_dataset_id
        )
        active_is_unknown = not active_is_source and not active_is_preview
        new_user_message = (
            latest_user_message is not None
            and latest_user_message["fingerprint"]
            != payload.last_handled_user_message_fingerprint
        )

        source_fields_exist = (
            payload.source_dataset_id is not None
            and payload.source_dataset_summary is not None
            and payload.source_causal_spec_draft is not None
        )
        source_changed = False
        if not source_fields_exist:
            source_changed = True
        elif active_is_preview:
            source_changed = not _same_draft(
                payload.source_causal_spec_draft, deps.causal_spec_draft
            )
        elif active_is_source:
            source_changed = (
                payload.source_fingerprint not in {None, source_fingerprint}
                or not _same_draft(payload.source_causal_spec_draft, deps.causal_spec_draft)
            )
            if payload.source_fingerprint is None:
                source_changed = (
                    payload.source_dataset_id != deps.dataset_id
                    or payload.source_dataset_summary.model_dump(mode="json")
                    != deps.dataset_summary.model_dump(mode="json")
                    or not _same_draft(payload.source_causal_spec_draft, deps.causal_spec_draft)
                )
        elif payload.phase != "INIT":
            source_changed = True

        compiled_review_complete = (
            payload.hard_failure
            or (
                payload.compiled_dataset_id is not None
                and payload.compiled_dataset_summary is not None
                and payload.compiled_causal_spec is not None
                and payload.transformation_plan is not None
                and payload.validation_status is not None
            )
        )
        if compiled_review_complete and not payload.hard_failure:
            compiled_review_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "compiled_dataset_id": str(payload.compiled_dataset_id),
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
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            compiled_review_complete = (
                payload.compiled_fingerprint == compiled_review_fingerprint
            )
        current_review_state = (
            payload.phase == "REVIEW_READY"
            and source_fields_exist
            and (active_is_source or active_is_preview)
            and not source_changed
            and compiled_review_complete
        )
        if current_review_state:
            if new_user_message and latest_user_message is not None:
                return self._handle_review_message(
                    request=request,
                    payload=payload,
                    deps=deps,
                    latest_user_message=latest_user_message,
                )
            message = (
                payload.assistant_message
                or "Please review the compiled dataset preview and confirm or request changes."
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        confirmed_complete = (
            payload.compiled_dataset_id is not None
            and payload.compiled_dataset_summary is not None
            and payload.compiled_causal_spec is not None
            and payload.transformation_plan is not None
            and payload.validation_status in {"PASS", "WARN"}
        )
        if (
            payload.phase == "CONFIRMED"
            and active_is_preview
            and confirmed_complete
            and not source_changed
        ):
            message = payload.assistant_message or "The compiled setup is already confirmed."
            return NodeExecutionResult(
                new_node_state=DataCompilationState(payload),
                new_orchestrator_state=request.orchestrator_state,
                status="DONE",
                action="NONE",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        compile_deps = deps
        compile_payload = payload.model_copy(
            update={
                "source_dataset_id": deps.dataset_id,
                "source_dataset_summary": deps.dataset_summary,
                "source_causal_spec_draft": deps.causal_spec_draft,
                "source_fingerprint": source_fingerprint,
                "compiled_dataset_id": None,
                "compiled_dataset_summary": None,
                "compiled_causal_spec": None,
                "effective_causal_spec_draft": None,
                "compiled_fingerprint": None,
                "cleaning_summary": None,
                "transformation_plan": None,
                "transformation_suggestions": None,
                "compilation_actions": [],
                "compilation_warnings": [],
                "validation_issues": [],
                "validation_status": None,
                "retry_count": 0,
                "retry_reason": None,
                "phase": "INIT",
                "hard_failure": False,
                "assistant_message": None,
                "system_message": None,
                "error_message": None,
            }
        )
        if active_is_preview and source_fields_exist:
            compile_deps = DataCompilationDeps(
                dataset_id=payload.source_dataset_id,
                dataset_summary=payload.source_dataset_summary,
                causal_spec_draft=deps.causal_spec_draft,
            )
            compile_source_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "dataset_id": str(compile_deps.dataset_id),
                        "dataset_summary": compile_deps.dataset_summary.model_dump(mode="json"),
                        "causal_spec_draft": compile_deps.causal_spec_draft.model_dump(
                            mode="json"
                        ),
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            compile_payload = payload.model_copy(
                update={
                    "source_dataset_id": compile_deps.dataset_id,
                    "source_dataset_summary": compile_deps.dataset_summary,
                    "source_causal_spec_draft": compile_deps.causal_spec_draft,
                    "source_fingerprint": compile_source_fingerprint,
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": None,
                    "compiled_causal_spec": None,
                    "effective_causal_spec_draft": None,
                    "compiled_fingerprint": None,
                    "cleaning_summary": None,
                    "transformation_plan": None,
                    "transformation_suggestions": None,
                    "compilation_actions": [],
                    "compilation_warnings": [],
                    "validation_issues": [],
                    "validation_status": None,
                    "retry_count": 0,
                    "retry_reason": None,
                    "phase": "INIT",
                    "hard_failure": False,
                    "assistant_message": None,
                    "system_message": None,
                    "error_message": None,
                }
            )
            try:
                request.orchestrator_state.set(
                    request.node_state.name(),
                    {
                        "working_dataset_id": compile_deps.dataset_id,
                        "latest_dataset_summary": compile_deps.dataset_summary,
                        "revert_request": True,
                    },
                )
            except Exception as exc:
                error = safe_err(exc)
                log.exception("failed to restore data compilation source", error=error)
                message = self._generate_user_message(
                    "hard_failure",
                    {
                        "step": "source restore before recompilation",
                        "error": error,
                        "hard_failure": True,
                        "source_dataset_id": compile_deps.dataset_id,
                        "messages_history": request.read_only_messages_history,
                    },
                    latest_user_message=latest_user_message,
                )
                blocked_payload = compile_payload.model_copy(
                    update={
                        "phase": "REVIEW_READY",
                        "hard_failure": True,
                        "assistant_message": message,
                        "system_message": "DATA_COMPILATION_BLOCKED",
                        "error_message": f"source restore failed: {error}",
                        "last_handled_user_message_fingerprint": (
                            latest_user_message["fingerprint"]
                            if latest_user_message is not None
                            else payload.last_handled_user_message_fingerprint
                        ),
                    }
                )
                return NodeExecutionResult(
                    new_node_state=DataCompilationState(blocked_payload),
                    new_orchestrator_state=request.orchestrator_state,
                    status="PENDING",
                    action="NEEDS_INPUT",
                    response_messages=[ChatMessage(role="assistant", content=message)],
                )
        elif active_is_unknown and payload.phase != "INIT":
            log.warning("data compilation active dataset is neither source nor preview")

        try:
            source_df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=compile_deps.dataset_id,
                limit=1_000_000,
            )
        except Exception as exc:
            error = safe_err(exc)
            log.exception(
                "failed to load data compilation source dataset",
                dataset_id=str(compile_deps.dataset_id),
                error=error,
            )
            message = self._generate_user_message(
                "hard_failure",
                {
                    "step": "source dataset loading",
                    "error": error,
                    "source_dataset_id": compile_deps.dataset_id,
                    "hard_failure": True,
                    "messages_history": request.read_only_messages_history,
                },
                latest_user_message=latest_user_message,
            )
            blocked_payload = compile_payload.model_copy(
                update={
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": f"source load failed: {error}",
                    "last_handled_user_message_fingerprint": (
                        latest_user_message["fingerprint"]
                        if latest_user_message is not None
                        else payload.last_handled_user_message_fingerprint
                    ),
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        return self._compile_pipeline(
            request=request,
            payload=compile_payload,
            deps=compile_deps,
            source_df=source_df,
            retry_count=0,
            recompile_instruction=None,
            trigger_message_fingerprint=(
                latest_user_message["fingerprint"] if latest_user_message else None
            ),
        )

    def _compile_pipeline(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        deps: DataCompilationDeps,
        source_df: pd.DataFrame,
        retry_count: int,
        recompile_instruction: str | None,
        trigger_message_fingerprint: str | None = None,
    ) -> NodeExecutionResult:
        source_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "dataset_id": str(deps.dataset_id),
                    "dataset_summary": deps.dataset_summary.model_dump(mode="json"),
                    "causal_spec_draft": deps.causal_spec_draft.model_dump(mode="json"),
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        base_payload = payload.model_copy(
            update={
                "source_dataset_id": deps.dataset_id,
                "source_dataset_summary": deps.dataset_summary,
                "source_causal_spec_draft": deps.causal_spec_draft,
                "source_fingerprint": source_fingerprint,
                "retry_count": retry_count,
                "retry_reason": recompile_instruction,
            }
        )

        try:
            cleaning_result = clean(
                data=source_df,
                data_summary=deps.dataset_summary,
                draft=deps.causal_spec_draft,
                data_maupulation_tools=self._data_manipulation_tool,
                data_profiling_tools=self._profiling_tool,
                llm=self._llm,
                revised_instructions=recompile_instruction,
            )
            cleaned_df = cleaning_result.pd_cleaned
            cleaned_summary = cleaning_result.cleaned_data_summary
            causal_spec = cleaning_result.causal
            effective_draft = cleaning_result.effective_draft or deps.causal_spec_draft
            cleaning_summary = cleaning_result.summary_str
            cleaning_notes = list(cleaning_result.cleaning_notes)
        except Exception as exc:
            error = safe_err(exc)
            log.exception("data compilation clean stage failed", error=error)
            message = self._generate_user_message(
                "hard_failure",
                {
                    "step": "cleaning",
                    "error": error,
                    "source_dataset_id": deps.dataset_id,
                    "source_dataset_summary": deps.dataset_summary,
                    "hard_failure": True,
                    "retry_count": retry_count,
                    "retry_reason": recompile_instruction,
                    "messages_history": request.read_only_messages_history,
                },
            )
            blocked_payload = base_payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": None,
                    "compiled_causal_spec": None,
                    "effective_causal_spec_draft": None,
                    "compiled_fingerprint": None,
                    "cleaning_summary": None,
                    "transformation_plan": None,
                    "transformation_suggestions": None,
                    "validation_status": None,
                    "validation_issues": [],
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": f"cleaning failed: {error}",
                    "last_handled_user_message_fingerprint": trigger_message_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        guard_error: str | None = None
        if not isinstance(cleaned_df, pd.DataFrame):
            guard_error = "cleaning did not return a dataframe"
        elif cleaned_df.empty:
            guard_error = "cleaning produced an empty dataframe"
        else:
            treatment_column = str(causal_spec.treatment_spec.column).strip()
            outcome_column = str(causal_spec.outcome_spec.column).strip()
            missing_required_columns = [
                column
                for column in [treatment_column, outcome_column]
                if column and column not in cleaned_df.columns
            ]
            if missing_required_columns:
                guard_error = (
                    "cleaning output is missing required causal column(s): "
                    + ", ".join(missing_required_columns)
                )

        if guard_error is not None:
            message = self._generate_user_message(
                "hard_failure",
                {
                    "step": "cleaning output guard",
                    "error": guard_error,
                    "source_dataset_id": deps.dataset_id,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "cleaning_summary": cleaning_summary,
                    "hard_failure": True,
                    "retry_count": retry_count,
                    "retry_reason": recompile_instruction,
                    "messages_history": request.read_only_messages_history,
                },
            )
            blocked_payload = base_payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "effective_causal_spec_draft": effective_draft,
                    "compiled_fingerprint": None,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": None,
                    "transformation_suggestions": None,
                    "validation_status": "FAIL",
                    "validation_issues": [
                        ValidationIssueModel(
                            severity="FAIL",
                            message=guard_error,
                            fix_hint=(
                                "Revise cleaning or preprocessing so the compiled dataset "
                                "contains rows and the locked treatment and outcome columns."
                            ),
                        )
                    ],
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": guard_error,
                    "last_handled_user_message_fingerprint": trigger_message_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        try:
            transformation_result = transform(
                transformation_instructions=(recompile_instruction or "").strip(),
                causal_spec=causal_spec,
                data_summary=cleaned_summary,
                llm=self._llm,
                encoding_plan_tool=self._encoding_plan_tool,
            )
        except Exception as exc:
            error = safe_err(exc)
            log.exception("data compilation transform stage failed", error=error)
            message = self._generate_user_message(
                "hard_failure",
                {
                    "step": "transformation planning",
                    "error": error,
                    "source_dataset_id": deps.dataset_id,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "cleaning_summary": cleaning_summary,
                    "hard_failure": True,
                    "retry_count": retry_count,
                    "retry_reason": recompile_instruction,
                    "messages_history": request.read_only_messages_history,
                },
            )
            blocked_payload = base_payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "effective_causal_spec_draft": effective_draft,
                    "compiled_fingerprint": None,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": None,
                    "transformation_suggestions": None,
                    "validation_status": None,
                    "validation_issues": [],
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": f"transformation failed: {error}",
                    "last_handled_user_message_fingerprint": trigger_message_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        transformation_plan = transformation_result.transformation_plan
        transformation_suggestions = transformation_result.transformation_suggestions
        transform_guard_issues: list[ValidationIssueModel] = []
        if transformation_plan is None:
            transform_guard_issues.append(
                ValidationIssueModel(
                    severity="FAIL",
                    message="No usable baseline transformation plan was produced.",
                    fix_hint="Rebuild the transformation plan for the compiled covariates.",
                )
            )
        else:
            seen_columns: set[str] = set()
            duplicate_columns: set[str] = set()
            missing_columns: list[str] = []
            for column_plan in transformation_plan.columns:
                column = str(column_plan.column).strip()
                if column in seen_columns:
                    duplicate_columns.add(column)
                seen_columns.add(column)
                if column not in cleaned_df.columns:
                    missing_columns.append(column)
            if duplicate_columns:
                transform_guard_issues.append(
                    ValidationIssueModel(
                        severity="FAIL",
                        message=(
                            "Transformation plan contains duplicate column entries: "
                            + ", ".join(sorted(duplicate_columns))
                        ),
                        fix_hint="Include each transform column exactly once.",
                    )
                )
            if missing_columns:
                transform_guard_issues.append(
                    ValidationIssueModel(
                        severity="FAIL",
                        message=(
                            "Transformation plan references missing compiled column(s): "
                            + ", ".join(sorted(set(missing_columns)))
                        ),
                        fix_hint=(
                            "Revise the transformation plan so it only references columns "
                            "present in the cleaned dataset."
                        ),
                    )
                )

        if transform_guard_issues:
            message = self._generate_user_message(
                "validation_failure",
                {
                    "step": "transformation output guard",
                    "source_dataset_id": deps.dataset_id,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "transformation_suggestions": transformation_suggestions,
                    "validation_status": "FAIL",
                    "validation_issues": transform_guard_issues,
                    "hard_failure": True,
                    "retry_count": retry_count,
                    "retry_reason": recompile_instruction,
                    "messages_history": request.read_only_messages_history,
                },
            )
            blocked_payload = base_payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "effective_causal_spec_draft": effective_draft,
                    "compiled_fingerprint": None,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "transformation_suggestions": transformation_suggestions,
                    "validation_status": "FAIL",
                    "validation_issues": transform_guard_issues,
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": "transformation output guard failed",
                    "last_handled_user_message_fingerprint": trigger_message_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        validation_result = validate_data_compilation(
            candidate_df=cleaned_df,
            causal_spec=causal_spec,
            transform_plan=transformation_plan,
        )
        validation_issues = validation_result.validation_errors
        validation_status: ValidationStatus = "PASS"
        if any(issue.severity == "FAIL" for issue in validation_issues):
            validation_status = "FAIL"
        elif any(issue.severity == "WARN" for issue in validation_issues):
            validation_status = "WARN"

        if (
            validation_status == "FAIL"
            and retry_count == 0
            and validation_result.user_suggestion_message
        ):
            retry_instruction = "\n\n".join(
                [
                    data_compilation_transformation_retry_guidance_prompt(),
                    validation_result.user_suggestion_message.strip(),
                ]
            ).strip()
            return self._compile_pipeline(
                request=request,
                payload=base_payload,
                deps=deps,
                source_df=source_df,
                retry_count=1,
                recompile_instruction=retry_instruction,
                trigger_message_fingerprint=trigger_message_fingerprint,
            )

        if validation_status == "FAIL":
            message = self._generate_user_message(
                "validation_failure",
                {
                    "step": "validation",
                    "source_dataset_id": deps.dataset_id,
                    "source_dataset_summary": deps.dataset_summary,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "effective_causal_spec_draft": effective_draft,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "transformation_suggestions": transformation_suggestions,
                    "validation_status": validation_status,
                    "validation_issues": validation_issues,
                    "hard_failure": True,
                    "retry_count": retry_count,
                    "retry_reason": recompile_instruction,
                    "messages_history": request.read_only_messages_history,
                },
            )
            blocked_payload = base_payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "effective_causal_spec_draft": effective_draft,
                    "compiled_fingerprint": None,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "transformation_suggestions": transformation_suggestions,
                    "validation_status": validation_status,
                    "validation_issues": validation_issues,
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": "validation failed",
                    "last_handled_user_message_fingerprint": trigger_message_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        preview_dataset_id = uuid4()
        try:
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=preview_dataset_id,
                df=cleaned_df,
                overwrite=True,
                include_index=False,
            )
        except Exception as exc:
            error = safe_err(exc)
            log.exception("data compilation preview save failed", error=error)
            message = self._generate_user_message(
                "hard_failure",
                {
                    "step": "preview dataset save",
                    "error": error,
                    "source_dataset_id": deps.dataset_id,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "validation_status": validation_status,
                    "validation_issues": validation_issues,
                    "hard_failure": True,
                    "retry_count": retry_count,
                    "retry_reason": recompile_instruction,
                    "messages_history": request.read_only_messages_history,
                },
            )
            blocked_payload = base_payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "effective_causal_spec_draft": effective_draft,
                    "compiled_fingerprint": None,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "transformation_suggestions": transformation_suggestions,
                    "validation_status": validation_status,
                    "validation_issues": validation_issues,
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": f"preview save failed: {error}",
                    "last_handled_user_message_fingerprint": trigger_message_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        try:
            request.orchestrator_state.set(
                request.node_state.name(),
                {
                    "working_dataset_id": preview_dataset_id,
                    "latest_dataset_summary": cleaned_summary,
                },
            )
        except Exception as exc:
            error = safe_err(exc)
            log.exception("data compilation preview activation failed", error=error)
            message = self._generate_user_message(
                "hard_failure",
                {
                    "step": "preview activation",
                    "error": error,
                    "source_dataset_id": deps.dataset_id,
                    "compiled_dataset_id": preview_dataset_id,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "validation_status": validation_status,
                    "validation_issues": validation_issues,
                    "hard_failure": True,
                    "retry_count": retry_count,
                    "retry_reason": recompile_instruction,
                    "messages_history": request.read_only_messages_history,
                },
            )
            blocked_payload = base_payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": cleaned_summary,
                    "compiled_causal_spec": causal_spec,
                    "effective_causal_spec_draft": effective_draft,
                    "compiled_fingerprint": None,
                    "cleaning_summary": cleaning_summary,
                    "transformation_plan": transformation_plan,
                    "transformation_suggestions": transformation_suggestions,
                    "validation_status": validation_status,
                    "validation_issues": validation_issues,
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": message,
                    "system_message": "DATA_COMPILATION_BLOCKED",
                    "error_message": f"preview activation failed: {error}",
                    "last_handled_user_message_fingerprint": trigger_message_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(blocked_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        compiled_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "compiled_dataset_id": str(preview_dataset_id),
                    "compiled_dataset_summary": cleaned_summary.model_dump(mode="json"),
                    "compiled_causal_spec": causal_spec.model_dump(mode="json"),
                    "transformation_plan": transformation_plan.model_dump(mode="json"),
                    "validation_status": validation_status,
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in validation_issues
                    ],
                },
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        compilation_actions = [
            "Published a compiled dataset preview after cleaning, transformation planning, and validation.",
            f"Row count changed from {deps.dataset_summary.n_rows} to {cleaned_summary.n_rows}.",
        ]
        if cleaning_summary:
            compilation_actions.append(cleaning_summary)
        compilation_actions.extend(note for note in cleaning_notes if note)
        if recompile_instruction:
            compilation_actions.append(
                "Applied recompilation or retry instruction: " + recompile_instruction
            )

        review_context = {
            "source_dataset_id": deps.dataset_id,
            "compiled_dataset_id": preview_dataset_id,
            "source_dataset_summary": deps.dataset_summary,
            "compiled_dataset_summary": cleaned_summary,
            "compiled_causal_spec": causal_spec,
            "effective_causal_spec_draft": effective_draft,
            "cleaning_summary": cleaning_summary,
            "transformation_plan": transformation_plan,
            "transformation_suggestions": transformation_suggestions,
            "compilation_actions": compilation_actions,
            "compilation_warnings": [],
            "validation_status": validation_status,
            "validation_issues": validation_issues,
            "retry_count": retry_count,
            "retry_reason": recompile_instruction,
            "messages_history": request.read_only_messages_history,
        }
        review_message = self._generate_user_message("review_summary", review_context)
        review_payload = base_payload.model_copy(
            update={
                "compiled_dataset_id": preview_dataset_id,
                "compiled_dataset_summary": cleaned_summary,
                "compiled_causal_spec": causal_spec,
                "effective_causal_spec_draft": effective_draft,
                "compiled_fingerprint": compiled_fingerprint,
                "cleaning_summary": cleaning_summary,
                "transformation_plan": transformation_plan,
                "transformation_suggestions": transformation_suggestions,
                "compilation_actions": compilation_actions,
                "compilation_warnings": [],
                "validation_status": validation_status,
                "validation_issues": validation_issues,
                "phase": "REVIEW_READY",
                "hard_failure": False,
                "assistant_message": review_message,
                "system_message": None,
                "error_message": None,
                "last_handled_user_message_fingerprint": trigger_message_fingerprint,
            }
        )
        return NodeExecutionResult(
            new_node_state=DataCompilationState(review_payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[ChatMessage(role="assistant", content=review_message)],
        )

    def _handle_review_message(
        self,
        *,
        request: NodeRequest,
        payload: DataCompilationPayloadModel,
        deps: DataCompilationDeps,
        latest_user_message: dict[str, Any],
    ) -> NodeExecutionResult:
        latest_fingerprint = str(latest_user_message["fingerprint"])
        try:
            decision = self._llm.generate_json(
                schema=_ReviewDecision,
                system_prompt=data_compilation_review_decision_prompt(),
                user_prompt=json.dumps(
                    {
                        "latest_user_message": latest_user_message["content"],
                        "hard_failure": payload.hard_failure,
                        "validation_status": payload.validation_status,
                        "validation_issues": [
                            issue.model_dump(mode="json", exclude_none=True)
                            for issue in payload.validation_issues
                        ],
                        "compiled_dataset_summary": (
                            payload.compiled_dataset_summary.model_dump(mode="json")
                            if payload.compiled_dataset_summary is not None
                            else None
                        ),
                        "compiled_causal_spec": (
                            payload.compiled_causal_spec.model_dump(mode="json")
                            if payload.compiled_causal_spec is not None
                            else None
                        ),
                        "cleaning_summary": payload.cleaning_summary,
                        "transformation_plan": (
                            payload.transformation_plan.model_dump(mode="json")
                            if payload.transformation_plan is not None
                            else None
                        ),
                        "confirmation_eligibility_facts": {
                            "active_dataset_id": str(deps.dataset_id),
                            "compiled_dataset_id": (
                                str(payload.compiled_dataset_id)
                                if payload.compiled_dataset_id is not None
                                else None
                            ),
                            "hard_failure": payload.hard_failure,
                            "validation_status": payload.validation_status,
                            "source_draft_unchanged": _same_draft(
                                payload.source_causal_spec_draft, deps.causal_spec_draft
                            ),
                        },
                    },
                    ensure_ascii=False,
                ),
                config=LLMConfig(model="basic", temperature=0.7),
                history=None,
                max_attempts=3,
            )
        except Exception as exc:
            error = safe_err(exc)
            log.exception("data compilation review decision failed", error=error)
            message = self._generate_user_message(
                "clarify_fallback",
                {"error": error, "latest_user_message": latest_user_message["content"]},
                latest_user_message=latest_user_message,
            )
            clarified_payload = payload.model_copy(
                update={
                    "assistant_message": message,
                    "system_message": None,
                    "error_message": f"review decision failed: {error}",
                    "last_handled_user_message_fingerprint": latest_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(clarified_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        if decision.action == "confirm":
            blockers: list[str] = []
            if payload.hard_failure:
                blockers.append("the current review state is blocked by a hard failure")
            if payload.compiled_dataset_id is None:
                blockers.append("there is no active compiled preview dataset")
            if payload.compiled_dataset_summary is None:
                blockers.append("the compiled dataset summary is missing")
            if payload.compiled_causal_spec is None:
                blockers.append("the compiled causal specification is missing")
            if payload.transformation_plan is None:
                blockers.append("the transformation plan is missing")
            if payload.validation_status not in {"PASS", "WARN"}:
                blockers.append("validation has not passed or produced only warnings")
            if payload.compiled_dataset_id is not None and deps.dataset_id != payload.compiled_dataset_id:
                blockers.append("the active dataset is not the compiled preview")
            if not _same_draft(payload.source_causal_spec_draft, deps.causal_spec_draft):
                blockers.append("the upstream causal draft changed after this preview")
            if payload.compiled_fingerprint is None:
                blockers.append("the compiled preview fingerprint is missing")
            elif (
                payload.compiled_dataset_id is not None
                and payload.compiled_dataset_summary is not None
                and payload.compiled_causal_spec is not None
                and payload.transformation_plan is not None
                and payload.validation_status is not None
            ):
                current_compiled_fingerprint = hashlib.sha256(
                    json.dumps(
                        {
                            "compiled_dataset_id": str(payload.compiled_dataset_id),
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
                        },
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                if current_compiled_fingerprint != payload.compiled_fingerprint:
                    blockers.append("the compiled preview state fingerprint no longer matches")

            if blockers:
                message = self._generate_user_message(
                    "cannot_confirm",
                    {
                        "confirmation_blockers": blockers,
                        "validation_status": payload.validation_status,
                        "validation_issues": payload.validation_issues,
                        "hard_failure": payload.hard_failure,
                    },
                    latest_user_message=latest_user_message,
                )
                blocked_payload = payload.model_copy(
                    update={
                        "assistant_message": message,
                        "system_message": None,
                        "error_message": "confirmation blocked: " + "; ".join(blockers),
                        "last_handled_user_message_fingerprint": latest_fingerprint,
                    }
                )
                return NodeExecutionResult(
                    new_node_state=DataCompilationState(blocked_payload),
                    new_orchestrator_state=request.orchestrator_state,
                    status="PENDING",
                    action="NEEDS_INPUT",
                    response_messages=[ChatMessage(role="assistant", content=message)],
                )

            try:
                request.orchestrator_state.set(
                    request.node_state.name(),
                    {
                        "working_dataset_id": payload.compiled_dataset_id,
                        "latest_dataset_summary": payload.compiled_dataset_summary,
                        "causal_spec_draft": (
                            payload.effective_causal_spec_draft
                            or payload.source_causal_spec_draft
                        ),
                        "causal_spec": payload.compiled_causal_spec,
                        "data_transformation_plan": payload.transformation_plan,
                        "working_dataset_frozen": True,
                        "validation_issues": payload.validation_issues,
                        "is_validated": True,
                    },
                )
            except Exception as exc:
                error = safe_err(exc)
                log.exception("data compilation acceptance publish failed", error=error)
                message = self._generate_user_message(
                    "hard_failure",
                    {
                        "step": "acceptance update",
                        "error": error,
                        "compiled_dataset_id": payload.compiled_dataset_id,
                        "validation_status": payload.validation_status,
                        "validation_issues": payload.validation_issues,
                        "hard_failure": True,
                    },
                    latest_user_message=latest_user_message,
                )
                blocked_payload = payload.model_copy(
                    update={
                        "phase": "REVIEW_READY",
                        "hard_failure": True,
                        "assistant_message": message,
                        "system_message": "DATA_COMPILATION_BLOCKED",
                        "error_message": f"acceptance update failed: {error}",
                        "last_handled_user_message_fingerprint": latest_fingerprint,
                    }
                )
                return NodeExecutionResult(
                    new_node_state=DataCompilationState(blocked_payload),
                    new_orchestrator_state=request.orchestrator_state,
                    status="PENDING",
                    action="NEEDS_INPUT",
                    response_messages=[ChatMessage(role="assistant", content=message)],
                )

            confirmed_payload = payload.model_copy(
                update={
                    "phase": "CONFIRMED",
                    "hard_failure": False,
                    "assistant_message": decision.assistant_message,
                    "system_message": None,
                    "error_message": None,
                    "last_handled_user_message_fingerprint": latest_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(confirmed_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="DONE",
                action="NONE",
                response_messages=[
                    ChatMessage(role="assistant", content=decision.assistant_message)
                ],
            )

        if decision.action == "answer_query":
            message = self._generate_user_message(
                "answer_query",
                {
                    "compiled_dataset_summary": payload.compiled_dataset_summary,
                    "compiled_causal_spec": payload.compiled_causal_spec,
                    "cleaning_summary": payload.cleaning_summary,
                    "transformation_plan": payload.transformation_plan,
                    "transformation_suggestions": payload.transformation_suggestions,
                    "compilation_actions": payload.compilation_actions,
                    "compilation_warnings": payload.compilation_warnings,
                    "validation_status": payload.validation_status,
                    "validation_issues": payload.validation_issues,
                    "hard_failure": payload.hard_failure,
                    "latest_user_message": latest_user_message["content"],
                    "messages_history": request.read_only_messages_history,
                },
                latest_user_message=latest_user_message,
            )
            answered_payload = payload.model_copy(
                update={
                    "assistant_message": message,
                    "system_message": None,
                    "error_message": None,
                    "last_handled_user_message_fingerprint": latest_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(answered_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=message)],
            )

        if decision.action == "recompile":
            recompile_request = (decision.recompile_request or "").strip()
            if not recompile_request:
                message = (
                    "I understood that you want changes before accepting, but I need one "
                    "clear sentence describing the same-column cleaning, filtering, "
                    "missingness, preprocessing, or encoding change to apply."
                )
                clarified_payload = payload.model_copy(
                    update={
                        "assistant_message": message,
                        "system_message": None,
                        "error_message": None,
                        "last_handled_user_message_fingerprint": latest_fingerprint,
                    }
                )
                return NodeExecutionResult(
                    new_node_state=DataCompilationState(clarified_payload),
                    new_orchestrator_state=request.orchestrator_state,
                    status="PENDING",
                    action="NEEDS_INPUT",
                    response_messages=[ChatMessage(role="assistant", content=message)],
                )

            if (
                payload.source_dataset_id is None
                or payload.source_dataset_summary is None
                or payload.source_causal_spec_draft is None
            ):
                message = self._generate_user_message(
                    "hard_failure",
                    {
                        "step": "review-time source reload",
                        "error": "source dataset state is missing",
                        "hard_failure": True,
                    },
                    latest_user_message=latest_user_message,
                )
                blocked_payload = payload.model_copy(
                    update={
                        "phase": "REVIEW_READY",
                        "hard_failure": True,
                        "assistant_message": message,
                        "system_message": "DATA_COMPILATION_BLOCKED",
                        "error_message": "source dataset state is missing for recompile",
                        "last_handled_user_message_fingerprint": latest_fingerprint,
                    }
                )
                return NodeExecutionResult(
                    new_node_state=DataCompilationState(blocked_payload),
                    new_orchestrator_state=request.orchestrator_state,
                    status="PENDING",
                    action="NEEDS_INPUT",
                    response_messages=[ChatMessage(role="assistant", content=message)],
                )

            try:
                active_dataset_id = request.orchestrator_state.get("working_dataset_id")
                if active_dataset_id != payload.source_dataset_id:
                    request.orchestrator_state.set(
                        request.node_state.name(),
                        {
                            "working_dataset_id": payload.source_dataset_id,
                            "latest_dataset_summary": payload.source_dataset_summary,
                            "revert_request": True,
                        },
                    )
                source_df = self._data_repo.get_csv_data(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataset_id=payload.source_dataset_id,
                    limit=1_000_000,
                )
            except Exception as exc:
                error = safe_err(exc)
                log.exception("review-time data compilation recompile setup failed", error=error)
                message = self._generate_user_message(
                    "hard_failure",
                    {
                        "step": "review-time source restore/reload",
                        "error": error,
                        "source_dataset_id": payload.source_dataset_id,
                        "hard_failure": True,
                    },
                    latest_user_message=latest_user_message,
                )
                blocked_payload = payload.model_copy(
                    update={
                        "phase": "REVIEW_READY",
                        "hard_failure": True,
                        "assistant_message": message,
                        "system_message": "DATA_COMPILATION_BLOCKED",
                        "error_message": f"review-time recompile setup failed: {error}",
                        "last_handled_user_message_fingerprint": latest_fingerprint,
                    }
                )
                return NodeExecutionResult(
                    new_node_state=DataCompilationState(blocked_payload),
                    new_orchestrator_state=request.orchestrator_state,
                    status="PENDING",
                    action="NEEDS_INPUT",
                    response_messages=[ChatMessage(role="assistant", content=message)],
                )

            source_deps = DataCompilationDeps(
                dataset_id=payload.source_dataset_id,
                dataset_summary=payload.source_dataset_summary,
                causal_spec_draft=payload.source_causal_spec_draft,
            )
            reset_payload = payload.model_copy(
                update={
                    "compiled_dataset_id": None,
                    "compiled_dataset_summary": None,
                    "compiled_causal_spec": None,
                    "effective_causal_spec_draft": None,
                    "compiled_fingerprint": None,
                    "cleaning_summary": None,
                    "transformation_plan": None,
                    "transformation_suggestions": None,
                    "compilation_actions": [],
                    "compilation_warnings": [],
                    "validation_issues": [],
                    "validation_status": None,
                    "retry_count": 0,
                    "retry_reason": recompile_request,
                    "phase": "INIT",
                    "hard_failure": False,
                    "assistant_message": None,
                    "system_message": None,
                    "error_message": None,
                }
            )
            return self._compile_pipeline(
                request=request,
                payload=reset_payload,
                deps=source_deps,
                source_df=source_df,
                retry_count=0,
                recompile_instruction=recompile_request,
                trigger_message_fingerprint=latest_fingerprint,
            )

        if decision.action == "reject":
            try:
                if (
                    payload.source_dataset_id is not None
                    and payload.source_dataset_summary is not None
                    and request.orchestrator_state.get("working_dataset_id")
                    != payload.source_dataset_id
                ):
                    request.orchestrator_state.set(
                        request.node_state.name(),
                        {
                            "working_dataset_id": payload.source_dataset_id,
                            "latest_dataset_summary": payload.source_dataset_summary,
                            "revert_request": True,
                        },
                    )
            except Exception as exc:
                error = safe_err(exc)
                log.exception("data compilation reject source restore failed", error=error)
                decision_message = (
                    "I understood that you do not accept this preview, but I could not "
                    f"restore the source dataset. Error: {error}"
                )
            else:
                decision_message = decision.assistant_message

            rejected_payload = payload.model_copy(
                update={
                    "phase": "REVIEW_READY",
                    "hard_failure": True,
                    "assistant_message": decision_message,
                    "system_message": "DATA_COMPILATION_REJECTED",
                    "error_message": "user rejected compiled preview",
                    "last_handled_user_message_fingerprint": latest_fingerprint,
                }
            )
            return NodeExecutionResult(
                new_node_state=DataCompilationState(rejected_payload),
                new_orchestrator_state=request.orchestrator_state,
                status="PENDING",
                action="NEEDS_INPUT",
                response_messages=[ChatMessage(role="assistant", content=decision_message)],
            )

        clarified_payload = payload.model_copy(
            update={
                "assistant_message": decision.assistant_message,
                "system_message": None,
                "error_message": None,
                "last_handled_user_message_fingerprint": latest_fingerprint,
            }
        )
        return NodeExecutionResult(
            new_node_state=DataCompilationState(clarified_payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[ChatMessage(role="assistant", content=decision.assistant_message)],
        )

    def _generate_user_message(
        self,
        mode: str,
        context: dict[str, Any],
        latest_user_message: dict[str, Any] | None = None,
    ) -> str:
        history_raw = context.get("messages_history")
        history = list(history_raw[-4:]) if history_raw else None

        prompt_context: dict[str, Any] = {}
        for key, value in context.items():
            if key == "messages_history":
                continue
            if hasattr(value, "model_dump"):
                prompt_context[key] = value.model_dump(mode="json")
            elif isinstance(value, list):
                prompt_context[key] = [
                    item.model_dump(mode="json", exclude_none=True)
                    if hasattr(item, "model_dump")
                    else item
                    for item in value
                ]
            else:
                prompt_context[key] = value
        if latest_user_message is not None:
            prompt_context["latest_user_message"] = latest_user_message["content"]

        if mode == "review_summary":
            system_prompt = data_compilation_review_summary_prompt()
            config = LLMConfig(model="mini", temperature=0.6)
        elif mode == "answer_query":
            system_prompt = data_compilation_review_query_prompt()
            config = LLMConfig(model="mini", temperature=0.6)
        elif mode == "validation_failure":
            system_prompt = data_compilation_validation_failure_message_prompt()
            config = LLMConfig(model="mini", temperature=0.5)
        elif mode == "hard_failure":
            system_prompt = data_compilation_hard_failure_message_prompt()
            config = LLMConfig(model="mini", temperature=0.5)
        elif mode == "cannot_confirm":
            system_prompt = data_compilation_cannot_confirm_message_prompt()
            config = LLMConfig(model="mini", temperature=0.4)
        elif mode == "clarify_fallback":
            system_prompt = data_compilation_clarify_fallback_message_prompt()
            config = LLMConfig(model="mini", temperature=0.5)
        else:
            system_prompt = data_compilation_message_generation_repair_prompt()
            config = LLMConfig(model="mini", temperature=0.5)

        try:
            message = self._llm.generate_json(
                schema=_ReviewSummary,
                system_prompt=system_prompt,
                user_prompt=json.dumps(prompt_context, ensure_ascii=False, default=str),
                config=config,
                history=history,
                max_attempts=2,
            )
            return message.assistant_message
        except Exception as exc:
            first_error = safe_err(exc)
            log.exception("data compilation user message generation failed", error=first_error)
            repair_context = {
                "mode": mode,
                "step": prompt_context.get("step"),
                "error": prompt_context.get("error") or first_error,
                "validation_status": prompt_context.get("validation_status"),
                "validation_issues": prompt_context.get("validation_issues"),
                "hard_failure": prompt_context.get("hard_failure"),
            }
            repair_message = self._llm.generate_json(
                schema=_ReviewSummary,
                system_prompt=data_compilation_message_generation_repair_prompt(),
                user_prompt=json.dumps(repair_context, ensure_ascii=False, default=str),
                config=LLMConfig(model="mini", temperature=0.3),
                history=None,
                max_attempts=2,
            )
            return repair_message.assistant_message


def _latest_user_message(messages_history: Sequence[ChatMessage] | None) -> dict[str, Any] | None:
    if not messages_history:
        return None
    for index in range(len(messages_history) - 1, -1, -1):
        message = messages_history[index]
        if message.role != "user":
            continue
        content = message.content.strip()
        if not content:
            continue
        if message.id:
            fingerprint = f"id:{message.id}"
        else:
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "index": index,
                        "role": message.role,
                        "content": content,
                        "created_at_utc": message.created_at_utc,
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        return {
            "content": content,
            "message_id": message.id,
            "created_at_utc": message.created_at_utc,
            "index": index,
            "fingerprint": fingerprint,
        }
    return None


def _same_draft(left: CausalSpecDraft | None, right: CausalSpecDraft | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.model_dump(mode="json") == right.model_dump(mode="json")
