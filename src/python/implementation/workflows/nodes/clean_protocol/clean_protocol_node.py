from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd
import pandas.api.types as ptypes
from pydantic import BaseModel, ConfigDict

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.clean_protocol.clean_protocol_deps import CleanProtocolDeps
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_prompts import (
    CLEANING_MESSAGE_TEMPLATE,
    get_clean_protocol_node_info,
)
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CleanProtocolPayloadModel,
    CleanProtocolState,
)
from python.implementation.workflows.tools.causal.causal_spec import BinaryOutcomeSpecModel, BinaryTreatmentSpecModel, CausalSpec, ContinuousOutcomeSpecModel
from python.implementation.workflows.tools.data_processing.data_processing_tool import (
    DataProcessingTool,
    ExclusionRulesModel,
)
from python.implementation.workflows.tools.data_profiling.causal_data_profiling_tool import (
    CausalDataProfilingTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.utils.utils import BOOL_FALSE, BOOL_TRUE


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanProtocolNode(Node):
    """
    Clean dataset into an inference-ready dataset artifact:

      1) Keep only required columns from protocol + exclusion rules
      2) Normalize missing sentinels
      3) Drop rows missing key modeling columns
      4) Apply exclusions through DataProcessingTool
      5) Keep treatment/outcome domains
      6) Run feasibility checks
      7) Save cleaned dataset + summary + graphs
    """

    NAME: ClassVar[str] = CleanProtocolState.NAME

    data_repo: DataRepo
    llm: LLMService

    strict_required_cols: bool = True
    missing_sentinels: Tuple[str, ...] = ("na", "nan", "null")

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
                DataProcessingTool,
                tool_factory.get_tool(DataProcessingTool.NAME),
            )

            deps = CleanProtocolDeps.from_loaded(previous_state_dependencies)

            dataset_id = deps.load_dataset.payload.id
            if dataset_id is None:
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error="LOAD_DATASET.id is missing; cannot load data.",
                        user_message="Dataset id missing. Re-run LOAD_DATASET.",
                    )
                )

            compiled_causal_specs = deps.compile_protocol.payload.causal_specs
            if compiled_causal_specs is None:
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error="COMPILE_PROTOCOL produced no causal specs.",
                        user_message="Causal specs missing. Re-run COMPILE_PROTOCOL.",
                    )
                )

            # Robust fallback: treat missing exclusion object as “no exclusions”
            compiled_exclusion = deps.compile_protocol.payload.exclusion
            if compiled_exclusion is None:
                compiled_exclusion = ExclusionRulesModel(exclusion_rules=[])

            df = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=None,
            )
            if df.empty:
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error=f"Dataset is empty (dataset_id={dataset_id}).",
                        user_message="Dataset is empty; cannot prepare inference-ready data.",
                    )
                )

            n_rows_0 = int(df.shape[0])
            n_cols_0 = int(df.shape[1])

            # 1) Keep only required columns from protocol + exclusions
            df1, drop_summary = edit_df_drop_cols_expect_required(
                df=df,
                compiled_causal_specs=compiled_causal_specs,
                exclusions=compiled_exclusion,
                keep_all_original=False,
                strict=self.strict_required_cols,
            )

            # 2) Normalize missing sentinels
            df2 = _normalize_missing_sentinels(
                df1,
                missing_sentinels=self.missing_sentinels,
            )

            # 3) Drop rows missing key modeling columns
            df3, null_summary = apply_key_null_purge(
                df2,
                causal_specs=compiled_causal_specs,
            )

            # 4) Apply exclusions through shared tool
            df4, exclusion_summary = apply_exclusions_via_tool(
                df=df3,
                exclusions=compiled_exclusion,
                data_processing_tool=data_processing_tool,
            )

            # 5) Treatment/outcome domain keep
            df5, domain_summary = apply_treatment_outcome_domain_keep(
                df4,
                compiled_causal_specs,
                keep_treatment_domain=True,
                keep_outcome_domain=True,
                dropna_on_domain_cols=False,
            )

            # 6) Feasibility checks
            feas_err = _feasibility_error(df5, compiled_causal_specs)
            if feas_err is not None:
                msg = _render_failure_message(
                    cleaning_error=feas_err,
                    n_rows_0=n_rows_0,
                    n_cols_0=n_cols_0,
                    drop_summary=drop_summary,
                    null_summary=null_summary,
                    exclusion_summary=exclusion_summary,
                    domain_summary=domain_summary,
                )
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error=feas_err,
                        user_message=msg,
                    )
                )

            # 7) Save cleaned dataset
            clean_id = uuid4()
            self.data_repo.save_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_id,
                df=df5,
                overwrite=True,
                include_index=False,
            )

            summary = data_profiling_tool.extract_dataset_summary(
                df5,
                max_categories=1000,
                sample_distinct=1000,
                compute_quantiles=True,
                strict=False,
            )

            artifact_ids: List[UUID] = []
            if (
                compiled_causal_specs.treatment_spec.kind == "binary"
                and (len(compiled_causal_specs.covariates) + len(compiled_causal_specs.effect_modifiers)) > 0
            ):
                graphs_list: List[GraphImage] = [
                    causal_data_profiling_tool.generate_causal_missingness_by_group_graph(
                        df=df5,
                        protocol=compiled_causal_specs,
                    ),
                    causal_data_profiling_tool.generate_comparability_overlap_histogram(
                        df=df5,
                        protocol=compiled_causal_specs,
                    ),
                ]
                graphs_list.extend(
                    causal_data_profiling_tool.generate_propensity_vs_top_confounders_graphs(
                        df=df5,
                        protocol=compiled_causal_specs,
                    )
                )

                for graph in graphs_list:
                    artifact_id = uuid4()
                    self.data_repo.save_artifact(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        artifact_id=artifact_id,
                        content=graph.content,
                        mime=graph.mime,
                        overwrite=True,
                    )
                    artifact_ids.append(artifact_id)

            message = _render_success_message(
                llm=self.llm,
                chat_history=messages_history,
                n_rows_0=n_rows_0,
                n_cols_0=n_cols_0,
                df_clean=df5,
                drop_summary=drop_summary,
                null_summary=null_summary,
                exclusion_summary=exclusion_summary,
                domain_summary=domain_summary,
            )

            return CleanProtocolState(
                payload=CleanProtocolPayloadModel(
                    clean_dataset_id=clean_id,
                    cleaning_error=None,
                    graph_picture_ids=artifact_ids,
                    summary=summary,
                    user_acceptance=message.user_acceptance,
                    user_message=message.message_for_user,
                )
            )

        except Exception as e:
            log.exception("CleanProtocolNode failed")
            return CleanProtocolState(
                payload=CleanProtocolPayloadModel(
                    clean_dataset_id=None,
                    cleaning_error=f"Clean protocol failed: {e!r}",
                    user_message=f"Clean protocol failed: {e!r}",
                )
            )


# =============================================================================
# Feasibility checks
# =============================================================================

def _feasibility_error(df: pd.DataFrame, causal_specs: CausalSpec) -> Optional[str]:
    if df.empty:
        return "Cleaned dataset has zero rows after preprocessing."

    if int(df.shape[1]) == 0:
        return "Cleaned dataset has zero columns after dropping to required columns."

    tcol = str(causal_specs.treatment_spec.column)
    ys = causal_specs.outcome_spec
    needed = {tcol, str(ys.column)}

    missing = [c for c in needed if c not in df.columns]
    if missing:
        return f"Cleaned dataset is missing required modeling columns: {missing}"

    ts = causal_specs.treatment_spec
    if isinstance(ts, BinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        nunq = int(df[tcol].nunique(dropna=True))
        if nunq < 2:
            return f"Binary treatment column '{tcol}' has <2 unique values after filtering."
    else:
        return f"Unsupported treatment spec kind: {getattr(ts, 'kind', None)!r}"

    if isinstance(ys, BinaryOutcomeSpecModel):
        ycol = str(ys.column)
        nunq = int(df[ycol].nunique(dropna=True))
        if nunq < 2:
            return f"Binary outcome column '{ycol}' has <2 unique values after filtering."
    elif isinstance(ys, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        ycol = str(ys.column)
        nunq = int(df[ycol].nunique(dropna=True))
        if nunq < 2:
            return f"Continuous outcome column '{ycol}' has <=1 unique value after filtering."
    else:
        return f"Unsupported outcome spec kind: {getattr(ys, 'kind', None)!r}"

    return None


# =============================================================================
# Message rendering
# =============================================================================

class CleaningMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message_for_user: str
    user_acceptance: Optional[bool] = None


def _render_success_message(
    *,
    llm: LLMService,
    chat_history: Optional[Sequence[ChatMessage]],
    n_rows_0: int,
    n_cols_0: int,
    df_clean: pd.DataFrame,
    drop_summary: "DropColsSummary",
    null_summary: "NullPurgeSummary",
    exclusion_summary: "ExclusionSummary",
    domain_summary: "TreatmentOutcomeDomainSummary",
) -> CleaningMessage:
    prompt_payload = { # pyright: ignore[reportUnknownVariableType]
        "initial_stats": {"rows": n_rows_0, "cols": n_cols_0},
        "final_stats": {"rows": len(df_clean), "cols": len(df_clean.columns)},
        "column_drops": asdict(drop_summary),
        "null_purge": asdict(null_summary),
        "exclusions": asdict(exclusion_summary),
        "domain_insights": asdict(domain_summary),
    }

    config = LLMConfig(model="basic", temperature=0.7)
    recent_history = chat_history[-12:] if chat_history else None

    return llm.generate_json(
        config=config,
        system_prompt=CLEANING_MESSAGE_TEMPLATE,
        user_prompt=json.dumps(prompt_payload),
        history=recent_history,
        schema=CleaningMessage,
        max_attempts=2,
    )


def _render_failure_message(
    *,
    cleaning_error: str,
    n_rows_0: int,
    n_cols_0: int,
    drop_summary: "DropColsSummary",
    null_summary: "NullPurgeSummary",
    exclusion_summary: "ExclusionSummary",
    domain_summary: "TreatmentOutcomeDomainSummary",
) -> str:
    parts: List[str] = []
    parts.append("Failed to prepare inference-ready dataset.")
    parts.append(f"Error: {cleaning_error}")
    parts.append("")
    parts.append("Diagnostics:")
    parts.append(f"- rows_before: {n_rows_0}, cols_before: {n_cols_0}")
    parts.append(
        f"- kept_cols: {len(drop_summary.kept_cols)}, dropped_cols: {len(drop_summary.dropped_cols)}"
    )
    if drop_summary.missing_required:
        parts.append(f"- missing_required_cols: {drop_summary.missing_required}")

    parts.append(f"- rows_after_key_null_purge: {null_summary.n_rows_after}")
    parts.append(f"- rows_after_exclusions: {exclusion_summary.n_rows_after}")
    parts.append(f"- rows_after_domain_keep: {domain_summary.n_rows_after}")
    parts.append(f"- exclusion_rules_count: {exclusion_summary.n_rules}")

    return "\n".join(parts)


# =============================================================================
# Summaries
# =============================================================================

@dataclass(frozen=True)
class DropColsSummary:
    kept_cols: List[str]
    dropped_cols: List[str]
    missing_required: List[str]
    required_cols: List[str]


@dataclass(frozen=True)
class NullPurgeSummary:
    required_nonnull_cols: List[str]
    n_rows_before: int
    n_rows_after: int
    n_removed: int


@dataclass(frozen=True)
class ExclusionSummary:
    n_rules: int
    n_rows_before: int
    n_rows_after: int
    n_removed: int


@dataclass(frozen=True)
class TreatmentOutcomeDomainSummary:
    n_rows_before: int
    n_rows_after: int
    total_removed: int
    treatment: Optional[Dict[str, Any]]
    outcome: Optional[Dict[str, Any]]


# =============================================================================
# Column projection
# =============================================================================

def edit_df_drop_cols_expect_required(
    df: pd.DataFrame,
    compiled_causal_specs: CausalSpec,
    exclusions: ExclusionRulesModel,
    *,
    keep_all_original: bool = False,
    strict: bool = True,
) -> Tuple[pd.DataFrame, DropColsSummary]:
    """
    Keep only columns required by protocol + exclusions.

    Required columns:
      - treatment_spec.column
      - outcome_spec.column
      - covariates
      - effect_modifiers
      - time_zero if time_zero_type == "COLUMN"
      - any exclusion rule columns
    """
    required: Set[str] = set()

    if compiled_causal_specs.time_zero_type == "COLUMN":
        required.add(str(compiled_causal_specs.time_zero))

    required.add(str(compiled_causal_specs.treatment_spec.column))

    ys = compiled_causal_specs.outcome_spec
    if ys.kind in ("binary", "continuous"):
        required.add(str(ys.column))
    else:
        raise ValueError(f"Unsupported outcome_spec kind: {getattr(ys, 'kind', None)!r}")

    required.update(str(c) for c in compiled_causal_specs.covariates)
    required.update(str(c) for c in compiled_causal_specs.effect_modifiers)

    for ex in exclusions.exclusion_rules:
        required.add(str(ex.column))

    required = {c.strip() for c in required if c.strip()}

    df_cols = [str(c) for c in df.columns]
    df_col_set = set(df_cols)

    missing = sorted([c for c in required if c not in df_col_set])
    required_sorted = sorted(required)

    if strict and missing:
        raise ValueError(
            f"edit_df_drop_cols_expect_required: missing required columns: {missing}"
        )

    if keep_all_original:
        kept = df_cols
        dropped: List[str] = []
        out = df.copy()
    else:
        kept = [c for c in df_cols if c in required]
        dropped = [c for c in df_cols if c not in required]
        out = df.loc[:, kept].copy()

    summary = DropColsSummary(
        kept_cols=kept,
        dropped_cols=dropped,
        missing_required=missing,
        required_cols=required_sorted,
    )
    return out, summary


# =============================================================================
# Missing normalization + null purge
# =============================================================================

def _normalize_missing_sentinels(
    df: pd.DataFrame,
    *,
    missing_sentinels: Sequence[str],
) -> pd.DataFrame:
    sent = {s.strip().casefold() for s in missing_sentinels if s.strip()}
    if not sent:
        return df.copy()

    out = df.copy()
    for c in out.columns:
        s = out[c]
        dt = s.dtype

        is_obj_or_str = ptypes.is_object_dtype(dt) or ptypes.is_string_dtype(dt)
        is_cat = isinstance(dt, pd.CategoricalDtype)

        if is_obj_or_str or is_cat:
            out[c] = s.map(
                lambda x: pd.NA
                if isinstance(x, str) and x.strip().casefold() in sent
                else x
            )

    return out


def apply_key_null_purge(
    df: pd.DataFrame,
    causal_specs: CausalSpec,
) -> Tuple[pd.DataFrame, NullPurgeSummary]:
    """
    Drop rows with nulls in key modeling columns only.
    """
    required_nonnull: List[str] = [
        str(causal_specs.treatment_spec.column),
        str(causal_specs.outcome_spec.column),
    ]

    if causal_specs.time_zero_type == "COLUMN":
        required_nonnull.append(str(causal_specs.time_zero))

    missing_req = [c for c in required_nonnull if c not in df.columns]
    if missing_req:
        raise KeyError(f"Required non-null columns missing in df: {missing_req}")

    n_before = int(df.shape[0])
    out = df.dropna(axis=0, how="any", subset=required_nonnull).copy() # pyright: ignore[reportUnknownMemberType]
    n_after = int(out.shape[0])

    summary = NullPurgeSummary(
        required_nonnull_cols=required_nonnull,
        n_rows_before=n_before,
        n_rows_after=n_after,
        n_removed=n_before - n_after,
    )
    return out, summary


# =============================================================================
# Exclusions via shared tool
# =============================================================================

def apply_exclusions_via_tool(
    *,
    df: pd.DataFrame,
    exclusions: ExclusionRulesModel,
    data_processing_tool: DataProcessingTool,
) -> Tuple[pd.DataFrame, ExclusionSummary]:
    n_before = int(df.shape[0])

    out = data_processing_tool.apply_exclusion_model(
        df=df,
        model=exclusions,
        copy=True,
        deep_copy=True,
    )

    n_after = int(out.shape[0])

    summary = ExclusionSummary(
        n_rules=len(exclusions.exclusion_rules),
        n_rows_before=n_before,
        n_rows_after=n_after,
        n_removed=n_before - n_after,
    )
    return out, summary


# =============================================================================
# Treatment / outcome domain keep
# =============================================================================

def _parse_bool_token_strict(v: str) -> Optional[bool]:
    s = v.strip().casefold()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    return None


def _coerce_literals_for_series(s: pd.Series, values: Sequence[str]) -> List[Any]:
    if ptypes.is_bool_dtype(s.dtype):
        out: List[bool] = []
        for raw in values:
            b = _parse_bool_token_strict(raw)
            if b is None:
                raise ValueError(
                    f"Invalid boolean literal {raw!r}. Allowed: {sorted(BOOL_TRUE | BOOL_FALSE)}"
                )
            out.append(b)
        return out

    if ptypes.is_numeric_dtype(s.dtype):
        outn: List[float] = []
        for raw in values:
            outn.append(float(raw))
        return outn

    if ptypes.is_datetime64_any_dtype(s.dtype):
        outd: List[pd.Timestamp] = []
        for raw in values:
            outd.append(pd.Timestamp(raw))
        return outd

    return [str(v) for v in values]


def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.casefold()


def _mask_keep_in_domain(s: pd.Series, allowed_literals: Sequence[str]) -> pd.Series:
    if not allowed_literals:
        return pd.Series([False] * len(s), index=s.index)

    if (
        ptypes.is_bool_dtype(s.dtype)
        or ptypes.is_numeric_dtype(s.dtype)
        or ptypes.is_datetime64_any_dtype(s.dtype)
    ):
        coerced = _coerce_literals_for_series(s, allowed_literals)
        return s.isin(coerced)

    s_norm = _norm_str_series(s)
    allowed_norm = [str(x).strip().casefold() for x in allowed_literals]
    return s_norm.isin(allowed_norm)


def apply_treatment_outcome_domain_keep(
    df: pd.DataFrame,
    compiled_causal_specs: CausalSpec,
    *,
    keep_treatment_domain: bool = True,
    keep_outcome_domain: bool = True,
    dropna_on_domain_cols: bool = False,
) -> Tuple[pd.DataFrame, TreatmentOutcomeDomainSummary]:
    cur = df.copy()
    n0 = int(cur.shape[0])

    t_summary: Optional[Dict[str, Any]] = None
    y_summary: Optional[Dict[str, Any]] = None

    if keep_treatment_domain:
        ts = compiled_causal_specs.treatment_spec
        tcol = str(ts.column)

        if tcol not in cur.columns:
            raise KeyError(f"Treatment column not found in df: {tcol!r}")

        if isinstance(ts, BinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
            allowed_t = [ts.treated, ts.control]
        else:
            raise ValueError(f"Unknown treatment_spec kind: {getattr(ts, 'kind', None)!r}")

        n_before = int(cur.shape[0])
        if dropna_on_domain_cols:
            cur = cur.dropna(axis=0, how="any", subset=[tcol]).copy() # pyright: ignore[reportUnknownMemberType]

        mask_keep = _mask_keep_in_domain(cur[tcol], allowed_t)
        cur = cur.loc[mask_keep].copy()
        n_after = int(cur.shape[0])

        t_summary = {
            "column": tcol,
            "allowed": allowed_t,
            "n_rows_before": n_before,
            "n_rows_after": n_after,
            "n_removed": n_before - n_after,
        }

    if keep_outcome_domain:
        ys = compiled_causal_specs.outcome_spec
        ycol = str(ys.column)

        if ycol not in cur.columns:
            raise KeyError(f"Outcome column not found in df: {ycol!r}")

        allowed_y: Optional[List[str]] = None
        if isinstance(ys, BinaryOutcomeSpecModel):
            allowed_y = [ys.event, ys.non_event]
        elif isinstance(ys, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
            allowed_y = None
        else:
            raise ValueError(f"Unknown outcome_spec kind: {getattr(ys, 'kind', None)!r}")

        if allowed_y is not None:
            n_before = int(cur.shape[0])
            if dropna_on_domain_cols:
                cur = cur.dropna(axis=0, how="any", subset=[ycol]).copy() # pyright: ignore[reportUnknownMemberType]

            mask_keep = _mask_keep_in_domain(cur[ycol], allowed_y)
            cur = cur.loc[mask_keep].copy()
            n_after = int(cur.shape[0])

            y_summary = {
                "kind": getattr(ys, "kind", "unknown"),
                "column": ycol,
                "allowed": allowed_y,
                "n_rows_before": n_before,
                "n_rows_after": n_after,
                "n_removed": n_before - n_after,
            }

    n_final = int(cur.shape[0])
    summary = TreatmentOutcomeDomainSummary(
        n_rows_before=n0,
        n_rows_after=n_final,
        total_removed=n0 - n_final,
        treatment=t_summary,
        outcome=y_summary,
    )
    return cur, summary