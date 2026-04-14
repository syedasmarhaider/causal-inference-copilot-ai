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

from python.domain.models.models import ArtifactRef, get_chat_messages_role_and_message_json
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.data_dashboard.data_dashboard_prompts import (
    ANALYTICS_INTERPRETATION_SYSTEM_PROMPT,
    FINAL_RESPONSE_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    MISSING_DATA_SYSTEM_PROMPT,
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    prev_state_revert_message,
)
from python.implementation.workflows.data_dashboard.data_dashboard_state import (
    DataDashboardPayloadModel,
    DataDashboardState,
)
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_tool import (
    AdvancedAnalyticsTool,
    AnalyticsResultModel,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
    DatasetSummaryModel,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool
from python.implementation.workflows.utils.utils import JSONDict, safe_err

log = get_app_logger(__name__, component="data_dashboard_node", log_type="node")

_MANIPULATION_RETRY = 3
_TABLE_PREFIX = "df_"
_TABLE_HASH_LEN = 16
_ARTIFACT_ANALYTICAL = "analytical_result"
_ARTIFACT_ANALYTICS = "analytics_result"
_ARTIFACT_CHART = "chart_spec"
_SUMMARY_PREVIEW_COLS = 8

_READY = (
    "Dashboard dataset is ready. You can ask questions, run statistical analyses, "
    "apply transformations, or generate charts."
)
_OUT_OF_SCOPE = (
    "This dashboard handles data exploration, analytics, transformations, and charts. "
    "Causal inference, DAG construction, and model training belong to other workflow "
    "stages — please navigate there for those tasks."
)
_OUT_OF_SCOPE_REPEAT = (
    "As noted, this dashboard focuses on data exploration and analytics. "
    "For causal inference or model training, use the appropriate workflow stage."
)


# ---------------------------------------------------------------------------
# Intent model — produced by the LLM
# ---------------------------------------------------------------------------


class DashboardIntentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent_data_question: bool = False
    intent_data_question_brief: str = ""
    intent_manipulation: bool = False
    intent_manipulation_brief: str = ""
    intent_manipulation_is_analytical_query: bool = False
    intent_analytics: bool = False
    intent_analytics_brief: str = ""
    intent_chart: bool = False
    intent_chart_brief: str = ""
    intent_out_of_scope: bool = False

    @field_validator(
        "intent_data_question_brief",
        "intent_manipulation_brief",
        "intent_analytics_brief",
        "intent_chart_brief",
        mode="before",
    )
    @classmethod
    def _norm(cls, v: Any) -> str:
        return str(v).strip() if v else ""

    @model_validator(mode="after")
    def _check(self) -> DashboardIntentModel:
        if self.intent_manipulation_is_analytical_query and not self.intent_manipulation:
            raise ValueError("analytical_query requires manipulation=true")
        if self.intent_data_question and not self.intent_data_question_brief:
            raise ValueError("data_question requires brief")
        if self.intent_manipulation and not self.intent_manipulation_brief:
            raise ValueError("manipulation requires brief")
        if self.intent_analytics and not self.intent_analytics_brief:
            raise ValueError("analytics requires brief")
        if self.intent_chart and not self.intent_chart_brief:
            raise ValueError("chart requires brief")
        return self

    def has_any(self) -> bool:
        return (
            self.intent_data_question
            or self.intent_manipulation
            or self.intent_analytics
            or self.intent_chart
        )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class DataDashboardNode(Node):
    NAME: ClassVar[str] = DataDashboardState.NAME

    def __init__(
        self,
        *,
        data_repo: DataRepo,
        llm: LLMService,
        tools_factory: ToolFactory,
    ) -> None:
        self._data_repo = data_repo
        self._llm = llm
        self._manipulation = cast(
            DataManipulationTool, tools_factory.get_tool(DataManipulationTool.NAME)
        )
        self._plot = cast(PlotTool, tools_factory.get_tool(PlotTool.NAME))
        self._profiling = cast(
            DatasetProfilingTool, tools_factory.get_tool(DatasetProfilingTool.NAME)
        )
        self._analytics = cast(
            AdvancedAnalyticsTool, tools_factory.get_tool(AdvancedAnalyticsTool.NAME)
        )

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return (
            "Data dashboard orchestrator. Interprets plain-language requests and dispatches "
            "to data profiling, SQL manipulation, advanced analytics (regression, propensity "
            "scores, t-tests, …), and chart generation tools."
        )

    # =====================================================================
    # run
    # =====================================================================

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        readonly_orchestrator_state: ReadOnlyOchestratorState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        if not isinstance(state, DataDashboardState):
            raise TypeError(f"{self.name}: expected DataDashboardState, got {type(state).__name__}")

        iterations = list(state.payload.dataset_iterations)
        cached_summary = (
            state.payload.latest_summary.model_copy(deep=True)
            if state.payload.latest_summary
            else None
        )

        # -- revert -------------------------------------------------------
        if _is_revert(messages_history):
            return self._revert(
                user_id=user_id,
                conversation_id=conversation_id,
                iterations=iterations,
                cached_summary=cached_summary,
            )

        # -- load dataset --------------------------------------------------
        df, summary, summary_json, iterations, loaded = self._load_dataset(
            user_id=user_id,
            conversation_id=conversation_id,
            iterations=iterations,
            cached_summary=cached_summary,
            messages_history=messages_history,
        )
        if df is None:
            # _load_dataset returns (None, …) when no data is available
            return self._state(user_message=cast(str, summary))

        persisted_summary: DatasetSummaryModel = summary  # type: ignore[assignment]

        # -- no user message yet -------------------------------------------
        user_msg = _last_user_msg(messages_history)
        if not user_msg:
            return self._state(
                iterations=iterations,
                summary=persisted_summary,
                user_message=(
                    self._loaded_msg(persisted_summary) if loaded else _READY
                ),
            )

        # -- classify intent -----------------------------------------------
        history_text = _recent_history_text(messages_history)
        try:
            intent = self._classify(user_msg, history_text, cast(str, summary_json))
        except Exception as exc:
            log.exception("intent classification failed", error=safe_err(exc))
            return self._state(
                iterations=iterations,
                summary=persisted_summary,
                user_message="I couldn't understand that request — please rephrase.",
            )

        # -- out-of-scope --------------------------------------------------
        if intent.intent_out_of_scope:
            prev_asst = _last_assistant_msg(messages_history)
            return self._state(
                iterations=iterations,
                summary=persisted_summary,
                user_message=(
                    _OUT_OF_SCOPE_REPEAT
                    if prev_asst and _is_scope_msg(prev_asst)
                    else _OUT_OF_SCOPE
                ),
            )

        if not intent.has_any():
            if loaded:
                return self._state(
                    iterations=iterations,
                    summary=persisted_summary,
                    user_message=self._loaded_msg(persisted_summary),
                )
            return self._state(
                iterations=iterations,
                summary=persisted_summary,
                user_message=_OUT_OF_SCOPE,
            )

        # -- execute intents -----------------------------------------------
        summary_answer: str | None = None
        manip_result: JSONDict | None = None
        analytics_result: JSONDict | None = None
        chart_result: JSONDict | None = None
        manip_artifacts: list[ArtifactRef] = []
        analytics_artifacts: list[ArtifactRef] = []
        chart_artifacts: list[ArtifactRef] = []

        work_df: pd.DataFrame = df
        work_summary: DatasetSummaryModel = summary  # type: ignore[assignment]
        work_json: str = summary_json  # type: ignore[assignment]

        # 1) data question
        if intent.intent_data_question:
            summary_answer = self._answer_question(
                intent.intent_data_question_brief or user_msg,
                work_json,
                history_text,
            )

        # 2) manipulation
        if intent.intent_manipulation:
            try:
                (
                    manip_result,
                    manip_artifacts,
                    iterations,
                    work_df,
                    work_summary,
                    work_json,
                    persisted_summary,
                ) = self._manipulate(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    iterations=iterations,
                    df=work_df,
                    summary=work_summary,
                    summary_json=work_json,
                    instructions=intent.intent_manipulation_brief or user_msg,
                    analytical=intent.intent_manipulation_is_analytical_query,
                    chart_follows=intent.intent_chart,
                )
            except Exception as exc:
                log.exception("manipulation failed", error=safe_err(exc))
                return self._state(
                    iterations=iterations,
                    summary=persisted_summary,
                    user_message="Data manipulation failed — try rephrasing.",
                )

        # 3) analytics
        if intent.intent_analytics:
            try:
                analytics_result, analytics_artifacts = self._run_analytics(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    df=work_df,
                    summary=work_summary,
                    request=intent.intent_analytics_brief or user_msg,
                )
            except Exception as exc:
                log.exception("analytics failed", error=safe_err(exc))
                analytics_result = {"status": "error", "detail": safe_err(exc)}

        # 4) chart
        if intent.intent_chart:
            try:
                chart_result, chart_artifacts = self._run_chart(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    df=work_df,
                    summary=work_summary,
                    instructions=intent.intent_chart_brief or user_msg,
                )
            except Exception as exc:
                log.exception("chart generation failed", error=safe_err(exc))
                chart_result = {"status": "error", "detail": safe_err(exc)}

        # -- final message -------------------------------------------------
        try:
            final = self._final_message(
                summary_answer=summary_answer,
                manipulation_result=manip_result,
                analytics_result=analytics_result,
                chart_result=chart_result,
                context={
                    "loaded_this_turn": loaded,
                    "original_message": user_msg,
                    "intents": {
                        "data_question": intent.intent_data_question,
                        "manipulation": intent.intent_manipulation,
                        "analytics": intent.intent_analytics,
                        "chart": intent.intent_chart,
                    },
                    "rows": int(len(work_df)),
                    "columns": [str(c) for c in work_df.columns],
                },
            )
        except Exception as exc:
            log.exception("final message build failed", error=safe_err(exc))
            final = self._fallback_message(
                summary_answer, manip_result, analytics_result, chart_result
            )

        all_refs = [*manip_artifacts, *analytics_artifacts, *chart_artifacts]
        return self._state(
            iterations=iterations,
            summary=persisted_summary,
            user_message=final,
            artifact_refs=all_refs or None,
        )

    # =====================================================================
    # State builder
    # =====================================================================

    def _state(
        self,
        *,
        user_message: str,
        iterations: Sequence[UUID] | None = None,
        summary: DatasetSummaryModel | None = None,
        artifact_refs: Sequence[ArtifactRef] | None = None,
    ) -> DataDashboardState:
        return DataDashboardState(
            DataDashboardPayloadModel(
                dataset_iterations=list(iterations or []),
                latest_summary=summary.model_copy(deep=True) if summary else None,
                user_message=user_message,
            ),
            response_message_artifact_refs=list(artifact_refs or []),
        )

    # =====================================================================
    # Dataset loading
    # =====================================================================

    def _load_dataset(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        iterations: list[UUID],
        cached_summary: DatasetSummaryModel | None,
        messages_history: Sequence[ChatMessage] | None,
    ) -> tuple[
        pd.DataFrame | None,
        DatasetSummaryModel | str,
        str | None,
        list[UUID],
        bool,
    ]:
        """Returns (df, summary_or_message, summary_json, iterations, loaded_this_turn).

        When no data is available, df is None and summary_or_message is a user-facing string.
        """
        if iterations:
            latest = iterations[-1]
            try:
                df = self._data_repo.get_csv_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=latest,
                    limit=1_000_000,
                )
            except Exception as exc:
                log.exception("load failed", dataset_id=str(latest), error=safe_err(exc))
                return None, "Could not load the working dataset — please re-upload.", None, iterations, False

            summary = cached_summary or self._profile(df)
            return df, summary, self._profiling.dataset_summary_to_json(summary), iterations, False

        # first load from INIT_DATA_ID
        try:
            df = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=DataDashboardState.INIT_DATA_ID,
                limit=1_000_000,
            )
        except Exception:
            msg = self._missing_data_msg(messages_history)
            return None, msg, None, iterations, False

        summary = self._profile(df)
        iterations.append(DataDashboardState.INIT_DATA_ID)
        return df, summary, self._profiling.dataset_summary_to_json(summary), iterations, True

    def _profile(self, df: pd.DataFrame) -> DatasetSummaryModel:
        return self._profiling.extract_dataset_summary(
            df, max_categories=200, sample_distinct=200, compute_quantiles=False, strict=True,
        )

    # =====================================================================
    # Revert
    # =====================================================================

    def _revert(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        iterations: list[UUID],
        cached_summary: DatasetSummaryModel | None,
    ) -> DataDashboardState:
        if not iterations:
            return self._state(user_message="No dataset to revert.")
        if len(iterations) == 1:
            return self._state(
                iterations=iterations, summary=cached_summary,
                user_message="Already at the original dataset — nothing to revert.",
            )
        reverted = iterations[:-1]
        summary = self._reload_summary(user_id, conversation_id, reverted[-1])
        return self._state(
            iterations=reverted, summary=summary,
            user_message="Reverted to previous dataset version.",
        )

    def _reload_summary(self, user_id: UUID, conv_id: UUID, ds_id: UUID) -> DatasetSummaryModel | None:
        try:
            df = self._data_repo.get_csv_data(
                user_id=user_id, conversation_id=conv_id, dataset_id=ds_id, limit=1_000_000,
            )
            return self._profile(df)
        except Exception as exc:
            log.exception("reload summary failed", error=safe_err(exc))
            return None

    # =====================================================================
    # LLM helpers
    # =====================================================================

    def _missing_data_msg(self, history: Sequence[ChatMessage] | None) -> str:
        resp = self._llm.generate(
            system_prompt=MISSING_DATA_SYSTEM_PROMPT,
            user_prompt="The user has no dataset loaded.",
            config=LLMConfig(model="mini", temperature=0.4),
            history=history,
        )
        return resp.content.strip()

    def _loaded_msg(self, summary: DatasetSummaryModel) -> str:
        profiles = list(summary.profiles)
        shown = profiles[:_SUMMARY_PREVIEW_COLS]
        preview = ", ".join(f"{p.name} ({p.inferred_kind.lower()})" for p in shown)
        extra = len(profiles) - _SUMMARY_PREVIEW_COLS
        if extra > 0:
            preview += f", +{extra} more"
        return (
            f"Dashboard dataset loaded — {summary.n_rows} rows, {len(profiles)} columns: "
            f"{preview}. Ask questions, run stats, transform, or chart."
        )

    def _classify(self, user_msg: str, history: str | None, summary_json: str) -> DashboardIntentModel:
        payload: JSONDict = {
            "latest_user_message": user_msg,
            "chat_history": history,
            "dataset_summary": summary_json,
        }
        return self._llm.generate_json(
            schema=DashboardIntentModel,
            system_prompt=INTENT_CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.0, top_p=1.0),
            history=None,
            max_attempts=2,
        )

    def _answer_question(self, brief: str, summary_json: str, history: str | None) -> str:
        payload: JSONDict = {
            "user_intent_brief": brief,
            "dataset_summary": summary_json,
            "chat_history": history,
        }
        try:
            resp = self._llm.generate(
                system_prompt=SUMMARY_ANSWER_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                config=LLMConfig(model="basic", temperature=0.2),
                history=None,
            )
            return resp.content.strip() or "Couldn't determine an answer from the summary."
        except Exception as exc:
            log.exception("summary answer failed", error=safe_err(exc))
            return "Couldn't answer from the summary."

    def _interpret_analytics(self, result: AnalyticsResultModel, user_request: str) -> str:
        payload: JSONDict = {
            "analysis_type": result.analysis_type,
            "summary": result.summary,
            "tables": result.tables,
            "metrics": result.metrics,
            "user_request": user_request,
        }
        try:
            resp = self._llm.generate(
                system_prompt=ANALYTICS_INTERPRETATION_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
                config=LLMConfig(model="basic", temperature=0.3),
                history=None,
            )
            return resp.content.strip()
        except Exception:
            return result.summary

    def _final_message(
        self,
        *,
        summary_answer: str | None,
        manipulation_result: JSONDict | None,
        analytics_result: JSONDict | None,
        chart_result: JSONDict | None,
        context: JSONDict,
    ) -> str:
        payload: JSONDict = {
            "summary_answer": summary_answer,
            "manipulation_result": manipulation_result,
            "analytics_result": analytics_result,
            "chart_result": chart_result,
            "dataset_context": context,
        }
        resp = self._llm.generate(
            system_prompt=FINAL_RESPONSE_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            config=LLMConfig(model="basic", temperature=0.3),
            history=None,
        )
        return resp.content

    def _fallback_message(
        self,
        summary_answer: str | None,
        manip: JSONDict | None,
        analytics: JSONDict | None,
        chart: JSONDict | None,
    ) -> str:
        parts: list[str] = []
        if summary_answer:
            parts.append(summary_answer.strip())
        if manip:
            s = str(manip.get("status", ""))
            if s == "dataset_updated":
                parts.append("Saved updated dataset version.")
            elif s == "analytical_query":
                parts.append("Ran analytical query.")
        if analytics:
            parts.append(str(analytics.get("interpretation", analytics.get("summary", ""))))
        if chart:
            n = int(chart.get("saved_chart_count", 0) or 0)
            parts.append(f"Generated {n} chart(s)." if n else "Generated chart output.")
        return " ".join(parts) if parts else "Completed the dashboard request."

    # =====================================================================
    # Manipulation intent
    # =====================================================================

    def _manipulate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        iterations: list[UUID],
        df: pd.DataFrame,
        summary: DatasetSummaryModel,
        summary_json: str,
        instructions: str,
        analytical: bool,
        chart_follows: bool,
    ) -> tuple[JSONDict, list[ArtifactRef], list[UUID], pd.DataFrame, DatasetSummaryModel, str, DatasetSummaryModel]:
        result_df = self._exec_manipulation(df, conversation_id, summary_json, instructions)

        if analytical:
            rid = uuid.uuid4()
            self._data_repo.save_csv_data(
                user_id=user_id, conversation_id=conversation_id,
                dataset_id=rid, df=result_df, overwrite=True, include_index=False,
            )
            ref = _artifact_ref(rid, "csv", _ARTIFACT_ANALYTICAL)
            if chart_follows:
                new_sum = self._profile(result_df)
                new_json = self._profiling.dataset_summary_to_json(new_sum)
                return (
                    {"status": "analytical_query", "instruction": instructions,
                     "result": _preview(result_df)},
                    [ref], iterations, result_df, new_sum, new_json, summary,
                )
            return (
                {"status": "analytical_query", "instruction": instructions,
                 "result": _preview(result_df)},
                [ref], iterations, df, summary, summary_json, summary,
            )

        new_id = uuid.uuid4()
        self._data_repo.save_csv_data(
            user_id=user_id, conversation_id=conversation_id,
            dataset_id=new_id, df=result_df, overwrite=True, include_index=False,
        )
        new_sum = self._profile(result_df)
        iterations.append(new_id)
        new_json = self._profiling.dataset_summary_to_json(new_sum)
        return (
            {"status": "dataset_updated", "instruction": instructions,
             "new_dataset_id": str(new_id), "result": _preview(result_df)},
            [], iterations, result_df, new_sum, new_json, new_sum,
        )

    def _exec_manipulation(
        self, df: pd.DataFrame, conv_id: UUID, summary_json: str, instructions: str,
    ) -> pd.DataFrame:
        sig = inspect.signature(self._manipulation.manipulate)
        kwargs: dict[str, Any] = {
            "dataframe": df,
            "data_summary": summary_json,
            "instructions": instructions,
        }
        if "table_name" in sig.parameters:
            kwargs["table_name"] = _table_name(conv_id)
        elif "conversation_id" in sig.parameters:
            kwargs["conversation_id"] = str(conv_id)
        else:
            raise TypeError("manipulation tool must accept table_name or conversation_id")
        if "retry_attempts" in sig.parameters:
            kwargs["retry_attempts"] = _MANIPULATION_RETRY
        return self._manipulation.manipulate(**kwargs)

    # =====================================================================
    # Analytics intent
    # =====================================================================

    def _run_analytics(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        df: pd.DataFrame,
        summary: DatasetSummaryModel,
        request: str,
    ) -> tuple[JSONDict, list[ArtifactRef]]:
        result: AnalyticsResultModel = self._analytics.analyze(
            dataframe=df, data_summary=summary, user_request=request,
        )
        interpretation = self._interpret_analytics(result, request)

        # persist result as JSON artifact
        rid = uuid.uuid4()
        payload = result.model_dump(mode="json")
        payload["interpretation"] = interpretation
        self._data_repo.save_json_data(
            user_id=user_id, conversation_id=conversation_id,
            dataset_id=rid,
            json_data=json.dumps(payload, ensure_ascii=False, default=str),
            overwrite=True,
        )
        ref = _artifact_ref(rid, "json", _ARTIFACT_ANALYTICS)
        return (
            {
                "status": "analytics_complete",
                "analysis_type": result.analysis_type,
                "summary": result.summary,
                "interpretation": interpretation,
                "metrics": result.metrics,
            },
            [ref],
        )

    # =====================================================================
    # Chart intent
    # =====================================================================

    def _run_chart(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        df: pd.DataFrame,
        summary: DatasetSummaryModel,
        instructions: str,
    ) -> tuple[JSONDict, list[ArtifactRef]]:
        specs = self._plot.generate_specs(
            dataframe=df, data_summary=summary, user_intent=instructions,
        )
        refs: list[ArtifactRef] = []
        for spec in specs:
            sid = uuid.uuid4()
            self._data_repo.save_json_data(
                user_id=user_id, conversation_id=conversation_id,
                dataset_id=sid,
                json_data=json.dumps(spec, ensure_ascii=False),
                overwrite=True,
            )
            refs.append(_artifact_ref(sid, "json", _ARTIFACT_CHART))
        return (
            {
                "status": "charts_saved",
                "instruction": instructions,
                "saved_chart_count": len(refs),
            },
            refs,
        )


# =========================================================================
# Module-level helpers
# =========================================================================


def _last_user_msg(history: Sequence[ChatMessage] | None) -> str | None:
    if not history:
        return None
    for m in reversed(history):
        if m.role == "user" and m.content.strip():
            return m.content.strip()
    return None


def _last_assistant_msg(history: Sequence[ChatMessage] | None) -> str | None:
    if not history:
        return None
    for m in reversed(history):
        if m.role == "assistant" and m.content.strip():
            return m.content.strip()
    return None


def _recent_history_text(history: Sequence[ChatMessage] | None) -> str | None:
    if not history or len(history) <= 1:
        return None
    return get_chat_messages_role_and_message_json(history[-5:-1])


def _is_revert(history: Sequence[ChatMessage] | None) -> bool:
    if not history:
        return False
    last = history[-1]
    return last.role == "user" and last.content.strip().lower() == prev_state_revert_message


def _is_scope_msg(value: str) -> bool:
    norm = " ".join(value.strip().casefold().split())
    return "dashboard" in norm and ("exploration" in norm or "analytics" in norm)


def _preview(df: pd.DataFrame, limit: int = 10) -> JSONDict:
    head = df.head(limit).copy()
    for c in head.columns:
        if pd.api.types.is_datetime64_any_dtype(head[c]):
            head[c] = head[c].dt.strftime("%Y-%m-%dT%H:%M:%S")  # pyright: ignore[reportAttributeAccessIssue]
    head = head.where(pd.notnull(head), None)  # pyright: ignore[reportArgumentType]
    return {
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "preview_rows": head.to_dict(orient="records"),
    }


def _table_name(conv_id: UUID) -> str:
    digest = hashlib.sha256(str(conv_id).encode("ascii")).hexdigest()
    return f"{_TABLE_PREFIX}{digest[:_TABLE_HASH_LEN]}"


def _artifact_ref(aid: UUID, fmt: str, kind: str) -> ArtifactRef:
    return {
        "id": aid,
        "kind": "data",
        "format": cast(Any, fmt),
        "artifact_meta": {"kind": kind},
    }


__all__ = ["DashboardIntentModel", "DataDashboardNode"]
