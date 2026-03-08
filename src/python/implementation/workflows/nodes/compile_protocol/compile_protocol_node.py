from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import UUID

from pydantic import ValidationError

from python.domain.service.llm_service import (
    AvailableModelsKey,
    ChatMessage,
    LLMConfig,
    LLMService,
)
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol import compile_protocol_prompt
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_deps import (
    CompileProtocolDeps,
)
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import (
    CompileProtocolPayloadModel,
    CompileProtocolState,
)

from python.implementation.workflows.tools.causal.causal_spec import BinaryOutcomeSpecModel, BinaryTreatmentSpecModel, CausalSpec, ContinuousOutcomeSpecModel
from python.implementation.workflows.tools.data_processing.data_processing_tool import (
    ExclusionRulesModel,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetSummaryModel,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileProtocolNode(Node):
    NAME: ClassVar[str] = CompileProtocolState.NAME

    llm: LLMService
    protocol_model_name: AvailableModelsKey = "basic"
    exclusion_model_name: AvailableModelsKey = "basic"

    # compiler-level attempts
    max_attempts: int = 2

    # llm.generate_json(...) attempts
    json_attempts: int = 2

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return compile_protocol_prompt.compile_protocol_node_info()

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
        deps = CompileProtocolDeps.from_loaded(previous_state_dependencies)

        ld = deps.load_dataset
        ds_summary: DatasetSummaryModel | None = ld.payload.summary
        assert ds_summary is not None, (
            "CompileProtocolNode requires dataset summary from LoadDatasetState"
        )

        protocol_discussion = deps.protocol_discussion.payload.discussion.strip()
        if len(protocol_discussion) < 10:
            raise ValueError(
                "CompileProtocolNode requires non-empty protocol discussion "
                "from ProtocolDiscussionState"
            )

        dataset_summary_json_str = ds_summary.model_dump_json()

        last_protocol_json = ""
        last_exclusion_json = ""
        last_errors: List[str] = []
        last_issues: List[Dict[str, Any]] = []

        for attempt in range(1, max(1, self.max_attempts) + 1):
            try:
                protocol_prompt = _build_protocol_prompt(
                    attempt=attempt,
                    protocol_text=protocol_discussion,
                    dataset_summary_json_str=dataset_summary_json_str,
                    previous_protocol_json=last_protocol_json,
                    previous_exclusion_json=last_exclusion_json,
                    validation_errors=last_errors,
                )

                protocol_model = _get_protocol_model(
                    llm=self.llm,
                    model_name=self.protocol_model_name,
                    prompt=protocol_prompt,
                    history=messages_history,
                    json_attempts=max(1, self.json_attempts),
                )

                last_protocol_json = json.dumps(
                    protocol_model.model_dump(mode="json"),
                    ensure_ascii=False,
                )

                exclusion_prompt = _build_exclusion_prompt(
                    attempt=attempt,
                    protocol_text=protocol_discussion,
                    dataset_summary_json_str=dataset_summary_json_str,
                    compiled_protocol_json=last_protocol_json,
                    previous_exclusion_json=last_exclusion_json,
                    validation_errors=last_errors,
                )

                exclusion_model = _get_exclusion_model(
                    llm=self.llm,
                    model_name=self.exclusion_model_name,
                    prompt=exclusion_prompt,
                    history=messages_history,
                    json_attempts=max(1, self.json_attempts),
                )

                last_exclusion_json = json.dumps(
                    exclusion_model.model_dump(mode="json"),
                    ensure_ascii=False,
                )

                issues: List[Dict[str, Any]] = []
                issues.extend(
                    _semantic_validate_protocol_values_against_dataset_summary(
                        causal_spec=protocol_model,
                        dataset_summary=ds_summary,
                    )
                )
                issues.extend(
                    _semantic_validate_exclusion_values_against_dataset_summary(
                        exclusion=exclusion_model,
                        dataset_summary=ds_summary,
                    )
                )

                hard_errors = _hard_errors_only(issues)
                if hard_errors:
                    last_issues = issues
                    last_errors = [_issue_to_str(x) for x in hard_errors]
                    continue

                return CompileProtocolState(
                    payload=CompileProtocolPayloadModel(
                        causal_specs=protocol_model,
                        exclusion=exclusion_model,
                        compile_error=None,
                        compile_issues=issues or None,
                        user_message="Protocol and exclusion rules compiled successfully.",
                    )
                )

            except ValidationError as ve:
                issues = _pydantic_error_to_issues(ve)
                last_issues = issues
                last_errors = [_issue_to_str(x) for x in issues]
                continue

            except Exception as e:
                msg = f"Attempt {attempt} failed: {e!r}"
                log.exception(msg)
                last_errors = [msg]
                last_issues = [
                    {
                        "path": "",
                        "message": msg,
                        "type": "exception",
                        "input": None,
                        "severity": "ERROR",
                    }
                ]
                continue

        err_text = _format_errors(
            errors=last_errors,
            protocol_json=last_protocol_json,
            exclusion_json=last_exclusion_json,
        )

        return CompileProtocolState(
            payload=CompileProtocolPayloadModel(
                causal_specs=None,
                exclusion=None,
                compile_error=err_text,
                compile_issues=last_issues or None,
                user_message="Failed to compile a valid protocol. Lets discuss the specs again.",
            )
        )


# =============================================================================
# structured llm calls
# =============================================================================

def _get_protocol_model(
    *,
    llm: LLMService,
    model_name: AvailableModelsKey,
    prompt: str,
    history: Optional[Sequence[ChatMessage]],
    json_attempts: int,
) -> CausalSpec:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    return llm.generate_json(
        schema=CausalSpec,
        system_prompt="Return JSON only. No extra text.",
        user_prompt=prompt,
        config=cfg,
        history=history,
        max_attempts=max(1, json_attempts),
    )


def _get_exclusion_model(
    *,
    llm: LLMService,
    model_name: AvailableModelsKey,
    prompt: str,
    history: Optional[Sequence[ChatMessage]],
    json_attempts: int,
) -> ExclusionRulesModel:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    return llm.generate_json(
        schema=ExclusionRulesModel,
        system_prompt="Return JSON only. No extra text.",
        user_prompt=prompt,
        config=cfg,
        history=history,
        max_attempts=max(1, json_attempts),
    )


# =============================================================================
# prompt builders
# =============================================================================

def _build_protocol_prompt(
    *,
    attempt: int,
    protocol_text: str,
    dataset_summary_json_str: str,
    previous_protocol_json: str,
    previous_exclusion_json: str,
    validation_errors: List[str],
) -> str:
    if attempt == 1:
        return (
            compile_protocol_prompt.compile_protocol_prompt()
            .replace("{{PROTOCOL_TEXT}}", protocol_text)
            .replace("{{DATASET_SUMMARY_JSON}}", dataset_summary_json_str)
        )

    return (
        compile_protocol_prompt.compile_protocol_repair_prompt()
        .replace("{{PROTOCOL_TEXT}}", protocol_text)
        .replace("{{DATASET_SUMMARY_JSON}}", dataset_summary_json_str)
        .replace("{{PREVIOUS_CAUSAL_SPEC_JSON}}", previous_protocol_json or "{}")
        .replace("{{PREVIOUS_EXCLUSION_JSON}}", previous_exclusion_json or "{}")
        .replace(
            "{{VALIDATION_ERRORS}}",
            json.dumps(validation_errors or ["Unknown compiler error"], ensure_ascii=False),
        )
    )


def _build_exclusion_prompt(
    *,
    attempt: int,
    protocol_text: str,
    dataset_summary_json_str: str,
    compiled_protocol_json: str,
    previous_exclusion_json: str,
    validation_errors: List[str],
) -> str:
    if attempt == 1:
        return (
            compile_protocol_prompt.compile_exclusion_prompt()
            .replace("{{PROTOCOL_TEXT}}", protocol_text)
            .replace("{{CAUSAL_SPEC_JSON}}", compiled_protocol_json or "{}")
            .replace("{{DATASET_SUMMARY_JSON}}", dataset_summary_json_str)
        )

    return (
        compile_protocol_prompt.compile_exclusion_repair_prompt()
        .replace("{{PROTOCOL_TEXT}}", protocol_text)
        .replace("{{CAUSAL_SPEC_JSON}}", compiled_protocol_json or "{}")
        .replace("{{DATASET_SUMMARY_JSON}}", dataset_summary_json_str)
        .replace("{{PREVIOUS_EXCLUSION_JSON}}", previous_exclusion_json or "{}")
        .replace(
            "{{VALIDATION_ERRORS}}",
            json.dumps(validation_errors or ["Unknown compiler error"], ensure_ascii=False),
        )
    )


# =============================================================================
# semantic validation
# =============================================================================

def _semantic_validate_protocol_values_against_dataset_summary(
    *,
    causal_spec: CausalSpec,
    dataset_summary: DatasetSummaryModel,
) -> List[Dict[str, Any]]:
    by_name = _build_profile_index(dataset_summary)
    issues: List[Dict[str, Any]] = []

    def add_issue(
        *,
        path: str,
        message: str,
        typ: str,
        val: Any,
        severity: str = "ERROR",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        out: Dict[str, Any] = {
            "path": path,
            "message": message,
            "type": typ,
            "input": val,
            "severity": severity,
        }
        if evidence:
            out["evidence"] = evidence
        issues.append(out)

    def prof(col: str) -> Optional[Any]:
        return by_name.get(col.strip())

    ts = causal_spec.treatment_spec
    ys = causal_spec.outcome_spec

    # simple protocol sanity
    if str(ts.column).strip() == str(ys.column).strip():
        add_issue(
            path="causal_spec",
            message=(
                f"Treatment column and outcome column must differ, got "
                f"{ts.column!r} for both treatment and outcome."
            ),
            typ="treatment_outcome_same_column",
            val={"treatment": ts.column, "outcome": ys.column},
            severity="ERROR",
        )

    # treatment validation
    if isinstance(ts, BinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        tcol = str(ts.column)
        p = prof(tcol)
        if p is None:
            add_issue(
                path="causal_spec.treatment_spec.column",
                message=f"Treatment column not found in dataset summary: {tcol!r}",
                typ="column_not_found",
                val=tcol,
                severity="ERROR",
            )
        else:
            k = _kind_of(p)

            if _norm_value(ts.treated) == _norm_value(ts.control):
                add_issue(
                    path="causal_spec.treatment_spec",
                    message=(
                        f"Binary treatment treated and control must differ, got "
                        f"{ts.treated!r} == {ts.control!r}"
                    ),
                    typ="treated_equals_control",
                    val={"treated": ts.treated, "control": ts.control},
                    severity="ERROR",
                )

            if k == "BOOLEAN":
                if not _is_bool_literal(ts.treated):
                    add_issue(
                        path="treatment_spec.treated",
                        message=(
                            f"treated value not boolean-like for boolean column "
                            f"{tcol!r}: {ts.treated!r}"
                        ),
                        typ="invalid_boolean_literal",
                        val=ts.treated,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": _dtype_of(p)},
                    )
                if not _is_bool_literal(ts.control):
                    add_issue(
                        path="treatment_spec.control",
                        message=(
                            f"control value not boolean-like for boolean column "
                            f"{tcol!r}: {ts.control!r}"
                        ),
                        typ="invalid_boolean_literal",
                        val=ts.control,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": _dtype_of(p)},
                    )

            elif k == "NUMERIC":
                if _parse_float_like(ts.treated) is None:
                    add_issue(
                        path="treatment_spec.treated",
                        message=(
                            f"treated value not parseable as float for numeric column "
                            f"{tcol!r}: {ts.treated!r}"
                        ),
                        typ="invalid_numeric_literal",
                        val=ts.treated,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": _dtype_of(p)},
                    )
                if _parse_float_like(ts.control) is None:
                    add_issue(
                        path="treatment_spec.control",
                        message=(
                            f"control value not parseable as float for numeric column "
                            f"{tcol!r}: {ts.control!r}"
                        ),
                        typ="invalid_numeric_literal",
                        val=ts.control,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": _dtype_of(p)},
                    )

            elif k == "CATEGORICAL":
                _check_value_membership_if_possible(
                    column=tcol,
                    path="treatment_spec.treated",
                    value=ts.treated,
                    label="Treatment",
                    profile=p,
                    add_issue=add_issue,
                )
                _check_value_membership_if_possible(
                    column=tcol,
                    path="treatment_spec.control",
                    value=ts.control,
                    label="Treatment",
                    profile=p,
                    add_issue=add_issue,
                )

            else:
                add_issue(
                    path="treatment_spec.column",
                    message=(
                        f"Binary treatment expects BOOLEAN/CATEGORICAL/NUMERIC column, "
                        f"got {k!r} for {tcol!r}"
                    ),
                    typ="column_kind_mismatch",
                    val={"column": tcol, "inferred_kind": k},
                    severity="ERROR",
                    evidence={"dtype": _dtype_of(p)},
                )
    else:
        raise ValueError(f"Unknown treatment_spec type: {type(ts).__name__}")

    # outcome validation
    if isinstance(ys, BinaryOutcomeSpecModel):
        ycol = str(ys.column)
        p = prof(ycol)
        if p is None:
            add_issue(
                path="outcome_spec.column",
                message=f"Outcome column not found in dataset summary: {ycol!r}",
                typ="column_not_found",
                val=ycol,
                severity="ERROR",
            )
        else:
            k = _kind_of(p)

            if _norm_value(ys.event) == _norm_value(ys.non_event):
                add_issue(
                    path="causal_spec.outcome_spec",
                    message=(
                        f"Binary outcome event and non_event must differ, got "
                        f"{ys.event!r} == {ys.non_event!r}"
                    ),
                    typ="event_equals_non_event",
                    val={"event": ys.event, "non_event": ys.non_event},
                    severity="ERROR",
                )

            if k == "BOOLEAN":
                if not _is_bool_literal(ys.event):
                    add_issue(
                        path="causal_spec.outcome_spec.event",
                        message=(
                            f"event value not boolean-like for boolean column "
                            f"{ycol!r}: {ys.event!r}"
                        ),
                        typ="invalid_boolean_literal",
                        val=ys.event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": _dtype_of(p)},
                    )
                if not _is_bool_literal(ys.non_event):
                    add_issue(
                        path="causal_spec.outcome_spec.non_event",
                        message=(
                            f"non_event value not boolean-like for boolean column "
                            f"{ycol!r}: {ys.non_event!r}"
                        ),
                        typ="invalid_boolean_literal",
                        val=ys.non_event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": _dtype_of(p)},
                    )

            elif k == "NUMERIC":
                if _parse_float_like(ys.event) is None:
                    add_issue(
                        path="causal_spec.outcome_spec.event",
                        message=(
                            f"event value not parseable as float for numeric column "
                            f"{ycol!r}: {ys.event!r}"
                        ),
                        typ="invalid_numeric_literal",
                        val=ys.event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": _dtype_of(p)},
                    )
                if _parse_float_like(ys.non_event) is None:
                    add_issue(
                        path="causal_spec.outcome_spec.non_event",
                        message=(
                            f"non_event value not parseable as float for numeric column "
                            f"{ycol!r}: {ys.non_event!r}"
                        ),
                        typ="invalid_numeric_literal",
                        val=ys.non_event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": _dtype_of(p)},
                    )

            elif k == "CATEGORICAL":
                _check_value_membership_if_possible(
                    column=ycol,
                    path="outcome_spec.event",
                    value=ys.event,
                    label="Outcome",
                    profile=p,
                    add_issue=add_issue,
                )
                _check_value_membership_if_possible(
                    column=ycol,
                    path="outcome_spec.non_event",
                    value=ys.non_event,
                    label="Outcome",
                    profile=p,
                    add_issue=add_issue,
                )

            else:
                add_issue(
                    path="outcome_spec.column",
                    message=(
                        f"Binary outcome expects BOOLEAN/CATEGORICAL/NUMERIC column, "
                        f"got {k!r} for {ycol!r}"
                    ),
                    typ="column_kind_mismatch",
                    val={"column": ycol, "inferred_kind": k},
                    severity="ERROR",
                    evidence={"dtype": _dtype_of(p)},
                )

    elif isinstance(ys, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        ycol = str(ys.column)
        p = prof(ycol)
        if p is None:
            add_issue(
                path="outcome_spec.column",
                message=f"Outcome column not found in dataset summary: {ycol!r}",
                typ="column_not_found",
                val=ycol,
                severity="ERROR",
            )
        else:
            if _kind_of(p) != "NUMERIC":
                add_issue(
                    path="outcome_spec.column",
                    message=(
                        f"Continuous outcome requires NUMERIC column, got "
                        f"{_kind_of(p)!r} for {ycol!r}"
                    ),
                    typ="column_kind_mismatch",
                    val={"column": ycol, "inferred_kind": _kind_of(p)},
                    severity="ERROR",
                    evidence={"dtype": _dtype_of(p)},
                )

        if ys.clip_min is not None and ys.clip_max is not None and ys.clip_min > ys.clip_max:
            add_issue(
                path="outcome_spec",
                message=f"clip_min must be <= clip_max, got {ys.clip_min} > {ys.clip_max}",
                typ="invalid_clip_bounds",
                val={"clip_min": ys.clip_min, "clip_max": ys.clip_max},
                severity="ERROR",
            )

    else:
        raise ValueError(f"Unknown outcome_spec type: {type(ys).__name__}")

    return issues


def _semantic_validate_exclusion_values_against_dataset_summary(
    *,
    exclusion: ExclusionRulesModel,
    dataset_summary: DatasetSummaryModel,
) -> List[Dict[str, Any]]:
    by_name = _build_profile_index(dataset_summary)
    issues: List[Dict[str, Any]] = []

    def add_issue(
        *,
        path: str,
        message: str,
        typ: str,
        val: Any,
        severity: str = "ERROR",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        out: Dict[str, Any] = {
            "path": path,
            "message": message,
            "type": typ,
            "input": val,
            "severity": severity,
        }
        if evidence:
            out["evidence"] = evidence
        issues.append(out)

    for i, ex in enumerate(exclusion.exclusion_rules):
        col = str(ex.column)
        p = by_name.get(col)

        if p is None:
            add_issue(
                path=f"exclusion_rules.{i}.column",
                message=f"Exclusion column not found in dataset summary: {col!r}",
                typ="column_not_found",
                val=col,
                severity="ERROR",
            )
            continue

        k = _kind_of(p)
        op = ex.op
        vals = list(ex.values)

        if op in (">", ">=", "<", "<="):
            if len(vals) != 1:
                add_issue(
                    path=f"exclusion_rules.{i}.values",
                    message=f"Operator {op!r} requires exactly 1 value, got {vals!r}",
                    typ="invalid_threshold_arity",
                    val=vals,
                    severity="ERROR",
                )
                continue

            v0 = vals[0]
            if v0 is None:
                add_issue(
                    path=f"exclusion_rules.{i}.values.0",
                    message=f"Operator {op!r} cannot use None/NA for column {col!r}",
                    typ="invalid_null_threshold",
                    val=v0,
                    severity="ERROR",
                    evidence={"column": col, "inferred_kind": k, "dtype": _dtype_of(p)},
                )
                continue

            if k == "NUMERIC":
                if _parse_float_like(v0) is None:
                    add_issue(
                        path=f"exclusion_rules.{i}.values.0",
                        message=(
                            f"Threshold value not parseable as float for numeric column "
                            f"{col!r}: {v0!r}"
                        ),
                        typ="invalid_numeric_threshold",
                        val=v0,
                        severity="ERROR",
                        evidence={"column": col, "inferred_kind": k, "dtype": _dtype_of(p)},
                    )
            elif k == "DATETIME":
                if _parse_iso_datetime_like(v0) is None:
                    add_issue(
                        path=f"exclusion_rules.{i}.values.0",
                        message=(
                            f"Threshold value not parseable as ISO datetime for datetime column "
                            f"{col!r}: {v0!r}"
                        ),
                        typ="invalid_datetime_threshold",
                        val=v0,
                        severity="ERROR",
                        evidence={"column": col, "inferred_kind": k, "dtype": _dtype_of(p)},
                    )
            else:
                add_issue(
                    path=f"exclusion_rules.{i}.op",
                    message=(
                        f"Operator {op!r} requires NUMERIC or DATETIME column; "
                        f"got {k!r} for {col!r}"
                    ),
                    typ="op_incompatible_with_column_kind",
                    val=op,
                    severity="ERROR",
                    evidence={"column": col, "inferred_kind": k, "dtype": _dtype_of(p)},
                )
            continue

        if op in ("==", "in", "not_in"):
            if k in ("CATEGORICAL", "BOOLEAN"):
                for j, v in enumerate(vals):
                    if v is None:
                        continue
                    _check_value_membership_if_possible(
                        column=col,
                        path=f"exclusion_rules.{i}.values.{j}",
                        value=v,
                        label="Exclusion",
                        profile=p,
                        add_issue=add_issue,
                    )
            elif k == "NUMERIC":
                for j, v in enumerate(vals):
                    if v is None:
                        continue
                    if _parse_float_like(v) is None:
                        add_issue(
                            path=f"exclusion_rules.{i}.values.{j}",
                            message=(
                                f"Value not parseable as float for numeric column "
                                f"{col!r}: {v!r}"
                            ),
                            typ="invalid_numeric_value",
                            val=v,
                            severity="ERROR",
                            evidence={"column": col, "dtype": _dtype_of(p)},
                        )
            elif k == "DATETIME":
                for j, v in enumerate(vals):
                    if v is None:
                        continue
                    if _parse_iso_datetime_like(v) is None:
                        add_issue(
                            path=f"exclusion_rules.{i}.values.{j}",
                            message=(
                                f"Value not parseable as ISO datetime for datetime column "
                                f"{col!r}: {v!r}"
                            ),
                            typ="invalid_datetime_value",
                            val=v,
                            severity="ERROR",
                            evidence={"column": col, "dtype": _dtype_of(p)},
                        )

    return issues


# =============================================================================
# dataset profile helpers
# =============================================================================

def _build_profile_index(dataset_summary: DatasetSummaryModel) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in dataset_summary.profiles:
        n = getattr(p, "name", None)
        if isinstance(n, str) and n.strip():
            out[n.strip()] = p
    return out


def _kind_of(p: Any) -> str:
    k = getattr(p, "inferred_kind", "OTHER")
    return str(k)


def _dtype_of(p: Any) -> Optional[str]:
    dt = getattr(p, "dtype", None)
    return str(dt) if isinstance(dt, str) else None


def _norm_value(v: Any) -> str:
    if isinstance(v, str):
        return v.strip().casefold()
    return str(v).strip().casefold()


def _categorical_domain(p: Any) -> Tuple[Set[str], bool, Dict[str, Any]]:
    distinct = getattr(p, "distinct_count", None)
    distinct_i = distinct if isinstance(distinct, int) else None

    summary = getattr(p, "summary", None)
    top = getattr(summary, "top_categories", None) if summary is not None else None
    other = getattr(summary, "other_count", None) if summary is not None else None
    other_i = other if isinstance(other, int) else None

    obs: Set[str] = set()
    top_count = 0

    if isinstance(top, list):
        top_count = len(top) # pyright: ignore[reportUnknownArgumentType]
        for item in top: # pyright: ignore[reportUnknownVariableType]
            v = getattr(item, "value", None) # pyright: ignore[reportUnknownArgumentType]
            if v is not None:
                obs.add(_norm_value(v))

    complete = False
    if other_i is not None and distinct_i is not None:
        complete = (other_i == 0) and (distinct_i == top_count)

    evidence: Dict[str, Any] = {
        "distinct_count": distinct_i,
        "top_count": top_count,
        "other_count": other_i,
    }
    return obs, complete, evidence


def _boolean_domain(p: Any) -> Tuple[Set[str], Dict[str, Any]]:
    summary = getattr(p, "summary", None)
    counts = getattr(summary, "counts", None) if summary is not None else None

    counts_keys: List[str] = []
    if isinstance(counts, dict):
        counts_keys = [_norm_value(k) for k in counts.keys()] # pyright: ignore[reportUnknownVariableType]

    obs: Set[str] = set(counts_keys)
    obs |= {"true", "false", "1", "0", "yes", "no", "t", "f", "y", "n"}

    return obs, {"counts_keys": sorted(list(set(counts_keys)))}


def _is_bool_literal(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, int) and v in (0, 1):
        return True
    if isinstance(v, str):
        return _norm_value(v) in {"true", "false", "1", "0", "yes", "no", "t", "f", "y", "n"}
    return False


def _parse_float_like(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _parse_iso_datetime_like(v: Any) -> Optional[datetime]:
    if not isinstance(v, str):
        return None

    s = v.strip()
    if not s:
        return None

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _check_value_membership_if_possible(
    *,
    column: str,
    path: str,
    value: Any,
    label: str,
    profile: Any,
    add_issue: Any,
) -> None:
    if value is None:
        return

    kind = _kind_of(profile)

    if kind == "CATEGORICAL":
        obs, complete, ev = _categorical_domain(profile)
        vn = _norm_value(value)
        if vn in obs:
            return

        if complete:
            add_issue(
                path=path,
                message=f"{label} value not present in column domain for {column!r}: {value!r}",
                typ="value_not_in_column_domain",
                val=value,
                severity="ERROR",
                evidence={"column": column, **ev},
            )
        else:
            add_issue(
                path=path,
                message=(
                    f"{label} value not found in profiled top categories for {column!r}: "
                    f"{value!r} (profile may be truncated; value could still exist)"
                ),
                typ="value_not_in_profile_sample",
                val=value,
                severity="WARNING",
                evidence={"column": column, **ev},
            )
        return

    if kind == "BOOLEAN":
        obs, ev = _boolean_domain(profile)
        vn = _norm_value(value)
        if vn not in obs:
            add_issue(
                path=path,
                message=f"{label} value incompatible with boolean column {column!r}: {value!r}",
                typ="value_incompatible_with_boolean",
                val=value,
                severity="ERROR",
                evidence={"column": column, **ev},
            )


# =============================================================================
# error helpers
# =============================================================================

def _hard_errors_only(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for x in issues:
        sev = str(x.get("severity", "ERROR")).upper()
        if sev == "ERROR":
            out.append(x)
    return out


def _pydantic_error_to_issues(err: ValidationError) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for e in err.errors():
        loc = e.get("loc", ())
        path = ".".join(str(x) for x in loc)
        issues.append(
            {
                "path": path,
                "message": str(e.get("msg", "Invalid value")),
                "type": str(e.get("type", "validation_error")),
                "input": e.get("input"),
                "severity": "ERROR",
            }
        )
    return issues


def _issue_to_str(issue: Mapping[str, Any]) -> str:
    path = str(issue.get("path", "")).strip()
    msg = str(issue.get("message", "Invalid value")).strip()
    return f"{path}: {msg}" if path else msg


def _format_errors(
    *,
    errors: List[str],
    protocol_json: str,
    exclusion_json: str,
) -> str:
    e = "\n".join([f"- {x}" for x in (errors or ["Unknown error"])])

    protocol_snippet = (protocol_json or "").strip()
    exclusion_snippet = (exclusion_json or "").strip()

    if len(protocol_snippet) > 1000:
        protocol_snippet = protocol_snippet[:1000] + "…"
    if len(exclusion_snippet) > 1000:
        exclusion_snippet = exclusion_snippet[:1000] + "…"

    return (
        f"Validation/compile errors:\n{e}\n\n"
        f"Last protocol JSON snippet:\n{protocol_snippet}\n\n"
        f"Last exclusion JSON snippet:\n{exclusion_snippet}"
    )