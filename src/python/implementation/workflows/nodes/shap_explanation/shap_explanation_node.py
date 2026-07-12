from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from python.domain.models.models import ArtifactRef, ChatMessage
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.shap_explanation.shap_explanation_deps import (
    ShapExplanationDeps,
)
from python.implementation.workflows.nodes.shap_explanation.shap_explanation_prompts import (
    SHAP_EXPLANATION_SUMMARY_SYSTEM_PROMPT,
    SHAP_EXPLANATION_SUMMARY_USER_PROMPT_TEMPLATE,
    get_shap_explanation_node_info,
)
from python.implementation.workflows.nodes.shap_explanation.shap_explanation_state import (
    ShapExplanationPayloadModel,
    ShapExplanationState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.inference.shap_cache import (
    build_shap_values_dataframe,
    serialize_econml_shap_values_for_effect_modifiers,
    summarize_shap_values_dataframe,
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

log = get_app_logger(__name__, component="shap_explanation_node", log_type="node")

_ARTIFACT_KIND_SHAP_VALUES = "shap_values"
_ARTIFACT_KIND_SHAP_QUERY_RESULT = "shap_query_result"
_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16


@dataclass(frozen=True)
class _ResolvedShapContext:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    selected_model: str
    trained_model_id: UUID
    inference_ready_spec: InferenceReadyCausalSpec
    shap_values_dataset_id: UUID | None = None
    shap_values_summary: dict[str, Any] | None = None
    shap_values_source_signature: str | None = None


@dataclass(frozen=True)
class _ShapDataset:
    dataframe: pd.DataFrame
    dataset_id: UUID
    summary: dict[str, Any]
    warnings: list[str]
    computed: bool


class ShapExplanationNode(Node):
    NAME: ClassVar[str] = ShapExplanationState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        data_repo: DataRepo,
        models_repo: ModelsRepo,
        tools_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        self._data_repo = data_repo
        self._models_repo = models_repo
        self._data_manipulation_tool = cast(
            DataManipulationTool,
            tools_factory.get_tool(DataManipulationTool.NAME),
        )
        self._profiling_tool = cast(
            DatasetProfilingTool,
            tools_factory.get_tool(DatasetProfilingTool.NAME),
        )

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_shap_explanation_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, ShapExplanationState):
            raise TypeError(
                f"{self.name}: expected ShapExplanationState, got "
                f"{type(request.node_state).__name__}"
            )

        payload = request.node_state.payload.model_copy(deep=True)
        try:
            deps = ShapExplanationDeps.from_request(request)
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "SHAP feature-importance analysis needs a trained causal model first."
                ),
                error_message=f"SHAP deps invalid: {safe_err(exc)}",
            )

        try:
            dataframe = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=None,
            )
        except Exception as exc:
            log.exception("SHAP_EXPLANATION failed to load dataset", error=exc)
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the compiled dataset needed for SHAP analysis. "
                    "Please retry after the dataset is available."
                ),
            )

        if dataframe.empty:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="The compiled dataset is empty, so SHAP values cannot be calculated.",
                error_message="compiled dataset is empty",
            )

        try:
            inference_ready_spec = InferenceReadyCausalSpec(
                causal_spec=deps.causal_spec,
                transformation_plan=deps.transformation_plan,
                data_summary=deps.dataset_summary,
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The compiled dataset, causal specification, and transformation plan are "
                    "not consistent enough for SHAP analysis yet."
                ),
                error_message=f"inference-ready spec invalid: {safe_err(exc)}",
            )

        resolved = _ResolvedShapContext(
            dataset_id=deps.dataset_id,
            dataset_summary=deps.dataset_summary,
            selected_model=deps.selected_model,
            trained_model_id=deps.trained_model_id,
            inference_ready_spec=inference_ready_spec,
            shap_values_dataset_id=deps.shap_values_dataset_id,
            shap_values_summary=deps.shap_values_summary,
            shap_values_source_signature=deps.shap_values_source_signature,
        )
        source_signature = _source_signature(resolved=resolved)
        if payload.source_signature != source_signature:
            payload = payload.reset_for_signature(source_signature=source_signature)

        effect_modifier_columns = inference_ready_spec.get_effect_modifiers_order()
        if not effect_modifier_columns:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "SHAP feature importance is not available because the confirmed protocol "
                    "has no effect modifiers."
                ),
                error_message="no effect modifiers",
            )

        latest_user_message = _latest_user_message(request.read_only_messages_history)
        shap_dataset = self._load_or_compute_shap_dataset(
            request=request,
            payload=payload,
            dataframe=dataframe,
            resolved=resolved,
            source_signature=source_signature,
            effect_modifier_columns=effect_modifier_columns,
        )
        if isinstance(shap_dataset, NodeExecutionResult):
            return shap_dataset

        query_result_df, query_error = self._query_shap_dataset(
            dataframe=shap_dataset.dataframe,
            conversation_id=request.conversation_id,
            request_summary=latest_user_message,
            resolved=resolved,
            shap_summary=shap_dataset.summary,
        )

        artifact_refs = [
            _build_data_artifact_ref(
                artifact_id=shap_dataset.dataset_id,
                artifact_kind=_ARTIFACT_KIND_SHAP_VALUES,
            )
        ]
        query_payload: dict[str, Any] = {
            "analysis_kind": "shap_feature_importance",
            "request_summary": latest_user_message,
            "shap_values_dataset_id": str(shap_dataset.dataset_id),
            "shap_summary": shap_dataset.summary,
            "query_result": _dataframe_preview(query_result_df),
        }
        if query_error:
            query_payload["query_error"] = query_error

        if not query_result_df.empty:
            query_result_id = uuid4()
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=query_result_id,
                df=query_result_df,
                overwrite=True,
                include_index=False,
            )
            artifact_refs.append(
                _build_data_artifact_ref(
                    artifact_id=query_result_id,
                    artifact_kind=_ARTIFACT_KIND_SHAP_QUERY_RESULT,
                )
            )

        assistant_message = self._generate_shap_answer(
            request_summary=latest_user_message,
            shap_summary=shap_dataset.summary,
            query_result_df=query_result_df,
            computed=shap_dataset.computed,
            query_error=query_error,
            history=request.read_only_messages_history,
        )

        next_payload = payload.model_copy(
            update={
                "source_signature": source_signature,
                "shap_values_dataset_id": shap_dataset.dataset_id,
                "shap_values_summary": shap_dataset.summary,
                "latest_query_result_raw_json_str": _dumps(query_payload),
                "latest_request_summary": latest_user_message,
                "assistant_message": assistant_message,
                "message_artifact_refs": artifact_refs,
                "error_message": query_error,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=next_payload,
            user_message=assistant_message,
            artifact_refs=artifact_refs,
            error_message=query_error,
        )

    def _load_or_compute_shap_dataset(
        self,
        *,
        request: NodeRequest,
        payload: ShapExplanationPayloadModel,
        dataframe: pd.DataFrame,
        resolved: _ResolvedShapContext,
        source_signature: str,
        effect_modifier_columns: Sequence[str],
    ) -> _ShapDataset | NodeExecutionResult:
        if (
            resolved.shap_values_dataset_id is not None
            and resolved.shap_values_source_signature == source_signature
        ):
            try:
                cached_df = self._data_repo.get_csv_data(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataset_id=resolved.shap_values_dataset_id,
                    limit=None,
                )
                if not cached_df.empty:
                    summary = resolved.shap_values_summary or summarize_shap_values_dataframe(
                        dataframe=cached_df,
                        dataset_id=resolved.shap_values_dataset_id,
                        identifier_column=str(resolved.inference_ready_spec.causal_spec.id_col),
                        effect_modifier_columns=effect_modifier_columns,
                        selected_model=resolved.selected_model,
                    )
                    return _ShapDataset(
                        dataframe=cached_df,
                        dataset_id=resolved.shap_values_dataset_id,
                        summary=summary,
                        warnings=[],
                        computed=False,
                    )
            except Exception as exc:
                log.warning(
                    "cached SHAP dataset load failed; recalculating",
                    dataset_id=str(resolved.shap_values_dataset_id),
                    error=safe_err(exc),
                )

        return self._compute_shap_dataset(
            request=request,
            payload=payload,
            dataframe=dataframe,
            resolved=resolved,
            source_signature=source_signature,
            effect_modifier_columns=effect_modifier_columns,
        )

    def _compute_shap_dataset(
        self,
        *,
        request: NodeRequest,
        payload: ShapExplanationPayloadModel,
        dataframe: pd.DataFrame,
        resolved: _ResolvedShapContext,
        source_signature: str,
        effect_modifier_columns: Sequence[str],
    ) -> _ShapDataset | NodeExecutionResult:
        model_record = self._models_repo.load_model(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            model_id=resolved.trained_model_id,
        )
        if model_record is None:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="The trained model artifact is not available for SHAP analysis.",
                error_message=f"model not found: {resolved.trained_model_id}",
            )

        try:
            x_query = _prepare_x_query(
                dataframe=dataframe,
                effect_modifier_columns=effect_modifier_columns,
                selected_model=resolved.selected_model,
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The effect-modifier columns could not be prepared for SHAP analysis."
                ),
                error_message=f"SHAP X preparation failed: {safe_err(exc)}",
            )

        log.info(
            "SHAP computation started",
            selected_model=resolved.selected_model,
            trained_model_id=str(resolved.trained_model_id),
            row_count=int(len(dataframe)),
            effect_modifier_columns=list(effect_modifier_columns),
        )
        shap_payload, shap_warnings = serialize_econml_shap_values_for_effect_modifiers(
            model_record.model,
            X=x_query,
            feature_names=effect_modifier_columns,
        )
        if shap_payload is None:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=("SHAP values are not available for the trained EconML estimator."),
                error_message="; ".join(shap_warnings) or "SHAP_NOT_AVAILABLE",
            )

        try:
            shap_df = build_shap_values_dataframe(
                dataframe=dataframe,
                identifier_column=str(resolved.inference_ready_spec.causal_spec.id_col),
                effect_modifier_columns=effect_modifier_columns,
                shap_values=cast(np.ndarray, shap_payload["values"]),
                shap_feature_names=cast(Sequence[str], shap_payload["feature_names"]),
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="SHAP values were calculated but could not be serialized to CSV.",
                error_message=f"SHAP dataframe build failed: {safe_err(exc)}",
            )

        dataset_id = uuid4()
        try:
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=dataset_id,
                df=shap_df,
                overwrite=True,
                include_index=False,
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="SHAP values were calculated but the CSV artifact could not be saved.",
                error_message=f"SHAP dataset save failed: {safe_err(exc)}",
            )

        summary = summarize_shap_values_dataframe(
            dataframe=shap_df,
            dataset_id=dataset_id,
            identifier_column=str(resolved.inference_ready_spec.causal_spec.id_col),
            effect_modifier_columns=effect_modifier_columns,
            selected_model=resolved.selected_model,
            warnings=shap_warnings,
        )
        request.orchestrator_state.set(
            ShapExplanationState.NAME,
            {
                "shap_values_dataset_id": dataset_id,
                "shap_values_summary": summary,
                "shap_values_source_signature": source_signature,
            },
        )
        log.info(
            "SHAP computation completed",
            selected_model=resolved.selected_model,
            trained_model_id=str(resolved.trained_model_id),
            dataset_id=str(dataset_id),
            row_count=int(len(shap_df)),
            shap_columns=summary.get("shap_columns"),
            warning_count=len(shap_warnings),
        )
        return _ShapDataset(
            dataframe=shap_df,
            dataset_id=dataset_id,
            summary=summary,
            warnings=list(shap_warnings),
            computed=True,
        )

    def _query_shap_dataset(
        self,
        *,
        dataframe: pd.DataFrame,
        conversation_id: UUID,
        request_summary: str,
        resolved: _ResolvedShapContext,
        shap_summary: Mapping[str, Any],
    ) -> tuple[pd.DataFrame, str | None]:
        try:
            summary_model = self._profiling_tool.extract_dataset_summary(
                dataframe,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )
            result_df = self._run_data_manipulation_tool(
                dataframe=dataframe,
                conversation_id=conversation_id,
                summary_json=self._profiling_tool.dataset_summary_to_json(summary_model),
                instructions=_build_shap_query_instructions(
                    request_summary=request_summary,
                    identifier_column=str(resolved.inference_ready_spec.causal_spec.id_col),
                    effect_modifier_columns=(
                        resolved.inference_ready_spec.get_effect_modifiers_order()
                    ),
                    shap_summary=shap_summary,
                ),
            )
            return result_df, None
        except Exception as exc:
            return _feature_importance_dataframe(shap_summary=shap_summary), (
                f"SHAP query failed, returned global importance fallback: {safe_err(exc)}"
            )

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
                "data manipulation tool must accept either 'table_name' or 'conversation_id'"
            )
        return manipulate(**kwargs)

    def _generate_shap_answer(
        self,
        *,
        request_summary: str,
        shap_summary: Mapping[str, Any],
        query_result_df: pd.DataFrame,
        computed: bool,
        query_error: str | None,
        history: Sequence[ChatMessage] | None,
    ) -> str:
        fallback = _build_shap_answer(
            shap_summary=shap_summary,
            query_result_df=query_result_df,
            computed=computed,
            query_error=query_error,
        )
        shap_context = {
            "artifact_status": "calculated" if computed else "reused",
            "shap_summary": dict(shap_summary),
            "query_result": _dataframe_preview(query_result_df),
            "query_error": query_error,
            "interpretation_constraints": {
                "shap_scope": "effect modifiers only",
                "global_importance_metric": "mean_abs_shap",
                "signed_values": (
                    "direction on the model's estimated treatment-effect scale, "
                    "not standalone causal proof"
                ),
            },
        }
        try:
            response = self._llm.generate(
                system_prompt=SHAP_EXPLANATION_SUMMARY_SYSTEM_PROMPT,
                user_prompt=SHAP_EXPLANATION_SUMMARY_USER_PROMPT_TEMPLATE.format(
                    request_summary=request_summary,
                    shap_context_json=_dumps(shap_context),
                ),
                config=LLMConfig(model="basic", temperature=0.2, max_tokens=2500),
                history=history,
            )
        except Exception as exc:
            log.warning("SHAP clinician summary generation failed", error=safe_err(exc))
            return fallback

        content = response.content.strip()
        return content or fallback

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: ShapExplanationPayloadModel,
        user_message: str,
        artifact_refs: Sequence[ArtifactRef] | None = None,
        error_message: str | None = None,
    ) -> NodeExecutionResult:
        updated_payload = payload.model_copy(
            update={
                "assistant_message": user_message,
                "message_artifact_refs": list(artifact_refs or []),
                "error_message": error_message,
            }
        )
        return NodeExecutionResult(
            new_node_state=ShapExplanationState(updated_payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[
                ChatMessage(
                    role="assistant",
                    content=user_message,
                    artifact_refs=list(artifact_refs or []) or None,
                )
            ],
        )

    def _needs_data_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ShapExplanationState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )


def _prepare_x_query(
    *,
    dataframe: pd.DataFrame,
    effect_modifier_columns: Sequence[str],
    selected_model: str,
) -> Any:
    missing = [str(column) for column in effect_modifier_columns if str(column) not in dataframe]
    if missing:
        raise ValueError(f"missing effect modifier columns: {missing}")
    x_df = dataframe.loc[:, [str(column) for column in effect_modifier_columns]].copy()
    if selected_model == "econml.dml.KernelDML":
        for column in x_df.columns:
            if not pd.api.types.is_numeric_dtype(x_df[column]):
                raise ValueError(f"KernelDML SHAP requires numeric effect modifier {column!r}")
        return x_df.to_numpy()
    return x_df


def _source_signature(*, resolved: _ResolvedShapContext) -> str:
    signature_payload = {
        "dataset_id": str(resolved.dataset_id),
        "dataset_summary": resolved.dataset_summary.model_dump(mode="json", exclude_none=True),
        "causal_spec": resolved.inference_ready_spec.causal_spec.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "transformation_plan": resolved.inference_ready_spec.transformation_plan.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "selected_model": resolved.selected_model,
        "trained_model_id": str(resolved.trained_model_id),
    }
    encoded = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_shap_query_instructions(
    *,
    request_summary: str,
    identifier_column: str,
    effect_modifier_columns: Sequence[str],
    shap_summary: Mapping[str, Any],
) -> str:
    shap_columns = ", ".join(str(column) for column in shap_summary.get("shap_columns", []))
    quoted_effect_modifiers = ", ".join(str(column) for column in effect_modifier_columns)
    return (
        "Use DuckDB SQL over the provided row-level SHAP dataframe. The dataframe is a "
        "separate feature-importance artifact and is not the CATE CSV. It contains the "
        f"identifier column `{identifier_column}`, original effect modifier columns "
        f"({quoted_effect_modifiers}), and SHAP attribution columns ({shap_columns}). "
        "SHAP values are row-level effect-modifier attributions for the trained EconML "
        "treatment-effect model. For global feature-importance questions, rank features by "
        "mean absolute SHAP. For local or patient-specific questions, include the identifier "
        "and the requested shap_ columns. Preserve the sign of SHAP values when the user asks "
        "about direction; use absolute values only for importance strength. "
        f"Answer this request with a compact result table: {request_summary}. "
        f"SHAP summary JSON: {_dumps(dict(shap_summary))}"
    )


def _build_shap_answer(
    *,
    shap_summary: Mapping[str, Any],
    query_result_df: pd.DataFrame,
    computed: bool,
    query_error: str | None,
) -> str:
    action = "calculated" if computed else "reused"
    top_features = _top_feature_lines(shap_summary=shap_summary, limit=5)
    parts = [
        f"I {action} a separate row-level SHAP CSV for the trained EconML model.",
    ]
    if top_features:
        parts.append(
            "The strongest effect modifiers by mean absolute SHAP are "
            + "; ".join(top_features)
            + "."
        )
    if not query_result_df.empty:
        parts.append(_compact_records_text(query_result_df) + ".")
    if query_error:
        parts.append(query_error)
    return " ".join(part for part in parts if part).strip()


def _top_feature_lines(*, shap_summary: Mapping[str, Any], limit: int) -> list[str]:
    importance = shap_summary.get("feature_importance")
    if not isinstance(importance, Mapping):
        return []
    ranked = importance.get("ranked")
    if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes, bytearray)):
        return []
    lines: list[str] = []
    for item in ranked[:limit]:
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column", "feature"))
        value = item.get("mean_abs_shap")
        if isinstance(value, (int, float)) and np.isfinite(value):
            lines.append(f"{column}={float(value):.4g}")
    return lines


def _feature_importance_dataframe(*, shap_summary: Mapping[str, Any]) -> pd.DataFrame:
    importance = shap_summary.get("feature_importance")
    if not isinstance(importance, Mapping):
        return pd.DataFrame()
    ranked = importance.get("ranked")
    if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes, bytearray)):
        return pd.DataFrame()
    records = [dict(item) for item in ranked if isinstance(item, Mapping)]
    return pd.DataFrame.from_records(records)


def _dataframe_preview(dataframe: pd.DataFrame, *, max_rows: int = 10) -> dict[str, Any]:
    preview_df = dataframe.head(max_rows).copy()
    for column in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[column]):
            preview_df[column] = preview_df[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    return {
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "preview_rows": preview_df.to_dict(orient="records"),
    }


def _compact_records_text(dataframe: pd.DataFrame, *, max_rows: int = 3) -> str:
    preview = _dataframe_preview(dataframe, max_rows=max_rows)
    records = preview["preview_rows"]
    if not records:
        return "no rows"
    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        pairs = [f"{key}: {value}" for key, value in record.items()]
        lines.append(f"row {index} ({'; '.join(pairs)})")
    return "The query result includes " + "; ".join(lines)


def _latest_user_message(history: Sequence[ChatMessage] | None) -> str:
    for message in reversed(list(history or [])):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return "Show global SHAP feature importance."


def _conversation_id_to_table_name(conversation_id: UUID) -> str:
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).hexdigest()
    return f"{_WORKING_TABLE_PREFIX}{digest[:_WORKING_TABLE_HASH_HEX_LEN]}"


def _build_data_artifact_ref(
    *,
    artifact_id: UUID,
    artifact_kind: str,
) -> ArtifactRef:
    return {
        "id": artifact_id,
        "kind": "data",
        "format": "csv",
        "artifact_meta": {"kind": artifact_kind},
    }


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, default=str)


__all__ = ["ShapExplanationNode"]
