from __future__ import annotations

from datetime import datetime
import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import UUID

from pydantic import ValidationError

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol import compile_protocol_prompt
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_deps import CompileProtocolDeps
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolPayloadModel, CompileProtocolState
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import BinaryOutcomeSpecModel, BinaryTreatmentSpecModel, CategoricalOutcomeSpecModel, CategoricalTreatmentSpecModel, ContinuousOutcomeSpecModel, ContinuousTreatmentSpecModel, DurationOutcomeSpecModel, ProtocolSpec
from python.implementation.workflows.nodes.load_dataset.load_dataset_utils import ColumnProfile, DatasetSummary
from python.implementation.workflows.utils.utils import json_sanitize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileProtocolNode(Node):
    NAME: ClassVar[str] = CompileProtocolState.NAME

    llm: LLMService
    model_name: str

    # Compile-level attempts: attempt 1 (base prompt) -> attempt 2 (repair prompt)
    max_attempts: int = 2

    # LLM internal attempts per prompt (handled inside llm.generate_json)
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
        tool_factory: Optional[ToolFactory],
        previous_state_dependencies: Mapping[str, Any],
        user_message: Optional[str],
        router_message: Optional[str],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        deps = CompileProtocolDeps.from_loaded(previous_state_dependencies)
        ld = deps.load_dataset
        ds_summary: DatasetSummary | None = ld.payload.summary
        assert ds_summary is not None, "CompileProtocolNode requires dataset summary from LoadDatasetState"  
        protocol_discussion = deps.protocol_discussion.payload.discussion
        if len(protocol_discussion.strip()) < 10:
            raise ValueError("CompileProtocolNode requires non-empty protocol discussion from ProtocolDiscussionState")
        
        last_json: str = ""
        last_errors: List[str] = []
        last_issues: List[Dict[str, Any]] = []

        for attempt in range(1, max(1, self.max_attempts) + 1):
            prompt = _build_prompt(
                attempt=attempt,
                protocol_text=protocol_discussion,
                dataset_summary=ds_summary,
                previous_json=last_json,
                validation_errors=last_errors,
                router_message=router_message,
            )
            
            attempt_history = messages_history if attempt == 1 else None

            try:
                protocol_model = _llm_protocol_model(
                    llm=self.llm,
                    model_name=self.model_name,
                    prompt=prompt,
                    history=attempt_history,
                    json_attempts=max(1, self.json_attempts),
                )

                sem_issues = _semantic_validate_values_against_dataset_summary(protocol_model, ds_summary)
                if sem_issues:
                    last_issues = sem_issues
                    last_errors = [_issue_to_str(x) for x in sem_issues]
                    continue
                
                llm_validate_response = _validate_through_llm(
                    llm=self.llm,
                    model_name=self.model_name,
                    protocol=protocol_model,
                    protocol_discussion=protocol_discussion,
                    dataset_summary=ds_summary,
                )
                
                
                if llm_validate_response and llm_validate_response.strip().lower() != "valid":
                    last_errors = [f"LLM validation failed: {llm_validate_response.strip()}"]
                    last_issues = [{"path": "", "message": llm_validate_response.strip(), "type": "llm_validation_failed", "input": None}]
                    continue

                # Store protocol as JSON dict for deterministic state serialization
                return CompileProtocolState( 
                    payload=CompileProtocolPayloadModel(
                        protocol=protocol_model,
                        compile_error=None,
                        compile_issues=None,
                        user_message="Protocol compiled successfully.",
                    )
                )
                
            except ValidationError as ve:
                # Safety: if the parser returns something but Pydantic validation fails
                issues = _pydantic_error_to_issues(ve)
                last_issues = issues
                last_errors = [_issue_to_str(x) for x in issues]
                continue

            except Exception as e:
                msg = f"Attempt {attempt} failed: {e!r}"
                log.exception(msg)
                last_errors = [msg]
                last_issues = [{"path": "", "message": msg, "type": "exception", "input": None}]
                continue

        err_text = _format_errors(last_errors, last_json)
        return CompileProtocolState(
            payload=CompileProtocolPayloadModel(
                protocol=None,
                compile_error=err_text,
                compile_issues=last_issues or None,
                user_message="Failed to compile a valid protocol. Lets discuss the specs again.",
            )
        )


# =============================================================================
# LLM call
# =============================================================================

def _llm_protocol_model(
    *,
    llm: LLMService,
    model_name: str,
    prompt: str,
    history: Optional[Sequence[ChatMessage]],
    json_attempts: int,
) -> ProtocolSpec:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    return llm.generate_json(
        schema=ProtocolSpec,
        system_prompt="Return JSON only. No extra text.",
        user_prompt=prompt,
        config=cfg,
        history=history,
        max_attempts=max(1, json_attempts),
    )


def _validate_through_llm(
    *,
    llm: LLMService,
    model_name: str,
    protocol: ProtocolSpec, 
    protocol_discussion: str, 
    dataset_summary: DatasetSummary):
    user_prompt = compile_protocol_prompt.protocol_validate_through_llm_prompt().replace("{{PROTOCOL_JSON}}",
    json.dumps(protocol.model_dump(mode="json"),
    ensure_ascii=False)).replace("{{PROTOCOL_DISCUSSION}}",
    protocol_discussion).replace("{{DATASET_SUMMARY_JSON}}", 
    json.dumps(json_sanitize(dict(dataset_summary)), ensure_ascii=False))
    
    llm_config = LLMConfig(model=model_name, temperature=0.0)
    return llm.generate(
        config=llm_config,
        system_prompt="Validate the protocol against the discussion and dataset summary.",
        user_prompt=user_prompt,
        history=None,
    ).content
    


# =============================================================================
# prompt builder
# =============================================================================

def _build_prompt(
    *,
    attempt: int,
    protocol_text: str,
    dataset_summary: Mapping[str, Any],
    previous_json: str,
    validation_errors: List[str],
    router_message: Optional[str],
) -> str:
    ds_json = json.dumps(json_sanitize(dict(dataset_summary)), ensure_ascii=False)

    appendix = ""
    if router_message:
        appendix = "\n\nROUTER_MESSAGE:\n" + router_message.strip()

    if attempt == 1:
        return (
            compile_protocol_prompt.compile_protocol_prompt()
            .replace("{{PROTOCOL_TEXT}}", protocol_text)
            .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
            + appendix
        )

    return (
        compile_protocol_prompt.compile_protocol_repair_prompt()
        .replace("{{PROTOCOL_TEXT}}", protocol_text)
        .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
        .replace("{{PREVIOUS_JSON}}", previous_json or "{}")
        .replace(
            "{{VALIDATION_ERRORS}}",
            json.dumps(validation_errors or ["Unknown compiler error"], ensure_ascii=False),
        )
        + appendix
    )


# =============================================================================
# semantic validation (columns)
# =============================================================================

def _semantic_validate_values_against_dataset_summary(
    protocol: ProtocolSpec,
    dataset_summary: DatasetSummary,
) -> List[Dict[str, Any]]:
    """
    Validate protocol *value literals* against DatasetSummary profiles.

    Produces structured issues:
      {path, message, type, input, severity, evidence?}

    Notes:
    - For CATEGORICAL columns, DatasetSummary only contains top_categories + other_count.
      So membership checks are:
        * ERROR if the categorical domain appears fully captured (other_count==0 and distinct_count == len(top_categories))
        * WARNING otherwise (value may exist but not in top sample)
    - For BOOLEAN columns, uses summary.counts keys + common boolean tokens.
    - For NUMERIC/DATETIME columns, enforces operator/value arity and parseability where reasonable.
    """

    profiles = dataset_summary.get("profiles")
    by_name: Dict[str, ColumnProfile] = {}
    for p in profiles:
        n = p.get("name")
        by_name[n.strip()] =  p

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

    def prof(col: str) -> Optional[ColumnProfile]:
        return by_name.get(col)

    def kind_of(p: ColumnProfile) -> str:
        k = p.get("inferred_kind")
        return k

    def norm(s: str) -> str:
        return s.strip().casefold()

    # -------------------------
    # Domain extraction helpers
    # -------------------------
    def categorical_domain(p: ColumnProfile) -> Tuple[Set[str], bool, Dict[str, Any]]:
        """
        Returns (observed_norm_set, is_domain_complete, evidence)
        """
        distinct = p.get("distinct_count")
        distinct_i = distinct if isinstance(distinct, int) else None

        summary = p.get("summary")
        top = None
        other_count_raw = None
        top = summary.get("top_categories")
        other_count_raw = summary.get("other_count")
        other_count = other_count_raw if isinstance(other_count_raw, int) else None
        obs: Set[str] = set()
        if top is not None:
            for item in top:
                if isinstance(item, dict):
                   v = item.get("value") # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                   if v is not None:
                        obs.add(norm(str(v))) # pyright: ignore[reportUnknownArgumentType]

        # Domain is "complete" only if we can strongly infer it.
        # With your profiler: if other_count==0 and distinct_count == len(top_categories), then fully captured.
        complete = False
        if other_count is not None and distinct_i is not None:
            complete = (other_count == 0) and (distinct_i == len(obs))

        ev = { # pyright: ignore[reportUnknownVariableType]
            "distinct_count": distinct_i,
            "top_count": len(obs),
            "other_count": other_count,
        }
        
        return obs, complete, ev # pyright: ignore[reportUnknownVariableType]

    def boolean_domain(p: ColumnProfile) -> Tuple[Set[str], Dict[str, Any]]:
        summary = p.get("summary")
        counts_keys: List[str] = []
        counts = summary.get("counts")
        if isinstance(counts, dict):
            counts_keys = [norm(str(k)) for k in counts.keys() if isinstance(k, str)] # pyright: ignore[reportUnknownVariableType]
        obs = set(counts_keys)
        obs |= {"true", "false", "1", "0", "yes", "no", "t", "f", "y", "n"}
        return obs, {"counts_keys": sorted(list(set(counts_keys)))}

    def is_bool_literal(v: str) -> bool:
        return norm(v) in {"true", "false", "1", "0", "yes", "no", "t", "f", "y", "n"}

    def parse_float(v: str) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    def parse_iso_datetime(v: str) -> Optional[datetime]:
        # Best-effort; your summary stores isoformat-ish strings.
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return None

    def check_value_membership_if_possible(*, col: str, path: str, value: str, label: str) -> None:
        p = prof(col)
        if p is None:
            return

        k = kind_of(p)

        if k == "CATEGORICAL":
            obs, complete, ev = categorical_domain(p)
            vn = norm(value)
            if vn in obs:
                return
            if complete:
                add_issue(
                    path=path,
                    message=f"{label} value not present in column domain for {col!r}: {value!r}",
                    typ="value_not_in_column_domain",
                    val=value,
                    severity="ERROR",
                    evidence={"column": col, **ev},
                )
            else:
                add_issue(
                    path=path,
                    message=f"{label} value not found in profiled top categories for {col!r}: {value!r} "
                            f"(profile may be truncated; value could still exist)",
                    typ="value_not_in_profile_sample",
                    val=value,
                    severity="WARNING",
                    evidence={"column": col, **ev},
                )
            return

        if k == "BOOLEAN":
            obs, ev = boolean_domain(p)
            vn = norm(value)
            if vn not in obs:
                add_issue(
                    path=path,
                    message=f"{label} value incompatible with boolean column {col!r}: {value!r}",
                    typ="value_incompatible_with_boolean",
                    val=value,
                    severity="ERROR",
                    evidence={"column": col, **ev},
                )
            return

        # NUMERIC/DATETIME/OTHER: membership not checkable via summary (no category domain).
        return

    # -------------------------
    # Exclusions values validation
    # -------------------------
    for i, ex in enumerate(protocol.exclusions):
        col = ex.column
        p = prof(col)
        if p is None:
            continue  # missing column is handled by your column-existence validator

        k = kind_of(p)
        op = ex.op
        vals = list(ex.values)
        
        if op in (">", ">=", "<", "<="):
            if len(vals) != 1:
                add_issue(
                    path=f"exclusions.{i}.values",
                    message=f"Operator {op!r} requires exactly 1 value, got {vals!r}",
                    typ="invalid_threshold_arity",
                    val=vals,
                    severity="ERROR",
                )
                continue

            v0 = vals[0]
            if k == "NUMERIC":
                if parse_float(v0) is None:
                    add_issue(
                        path=f"exclusions.{i}.values.0",
                        message=f"Threshold value not parseable as float for numeric column {col!r}: {v0!r}",
                        typ="invalid_numeric_threshold",
                        val=v0,
                        severity="ERROR",
                        evidence={"column": col, "inferred_kind": k, "dtype": p.get("dtype")},
                    )
            elif k == "DATETIME":
                if parse_iso_datetime(v0) is None:
                    add_issue(
                        path=f"exclusions.{i}.values.0",
                        message=f"Threshold value not parseable as ISO datetime for datetime column {col!r}: {v0!r}",
                        typ="invalid_datetime_threshold",
                        val=v0,
                        severity="ERROR",
                        evidence={"column": col, "inferred_kind": k, "dtype": p.get("dtype")},
                    )
            else:
                add_issue(
                    path=f"exclusions.{i}.op",
                    message=f"Operator {op!r} requires NUMERIC or DATETIME column; got {k!r} for {col!r}",
                    typ="op_incompatible_with_column_kind",
                    val=op,
                    severity="ERROR",
                    evidence={"column": col, "inferred_kind": k, "dtype": p.get("dtype")},
                )
            continue

        # Membership-like ops
        if op in ("==","in", "not_in"):
            # For categorical/boolean: check membership when possible.
            if k in ("CATEGORICAL", "BOOLEAN"):
                for j, v in enumerate(vals):
                    check_value_membership_if_possible(
                        col=col,
                        path=f"exclusions.{i}.values.{j}",
                        value=v,
                        label="Exclusion",
                    )
            elif k == "NUMERIC":
                # Ensure parseable numbers (no membership check)
                for j, v in enumerate(vals):
                    if parse_float(v) is None:
                        add_issue(
                            path=f"exclusions.{i}.values.{j}",
                            message=f"Value not parseable as float for numeric column {col!r}: {v!r}",
                            typ="invalid_numeric_value",
                            val=v,
                            severity="ERROR",
                            evidence={"column": col, "dtype": p.get("dtype")},
                        )
            elif k == "DATETIME":
                for j, v in enumerate(vals):
                    if parse_iso_datetime(v) is None:
                        add_issue(
                            path=f"exclusions.{i}.values.{j}",
                            message=f"Value not parseable as ISO datetime for datetime column {col!r}: {v!r}",
                            typ="invalid_datetime_value",
                            val=v,
                            severity="ERROR",
                            evidence={"column": col, "dtype": p.get("dtype")},
                        )

    # -------------------------
    # Treatment values validation
    # -------------------------
    ts = protocol.treatment_spec

    if isinstance(ts, BinaryTreatmentSpecModel):
        tcol = ts.column
        p = prof(tcol)
        if p is not None:
            k = kind_of(p)

            # treated/control must differ
            if norm(ts.treated) == norm(ts.control):
                add_issue(
                    path="treatment_spec",
                    message=f"Binary treatment treated and control must differ, got {ts.treated!r} == {ts.control!r}",
                    typ="treated_equals_control",
                    val={"treated": ts.treated, "control": ts.control},
                    severity="ERROR",
                )

            # Compatibility + membership checks
            if k == "BOOLEAN":
                if not is_bool_literal(ts.treated):
                    add_issue(
                        path="treatment_spec.treated",
                        message=f"treated value not boolean-like for boolean column {tcol!r}: {ts.treated!r}",
                        typ="invalid_boolean_literal",
                        val=ts.treated,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": p.get("dtype")},
                    )
                if not is_bool_literal(ts.control):
                    add_issue(
                        path="treatment_spec.control",
                        message=f"control value not boolean-like for boolean column {tcol!r}: {ts.control!r}",
                        typ="invalid_boolean_literal",
                        val=ts.control,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": p.get("dtype")},
                    )
            elif k == "NUMERIC":
                if parse_float(ts.treated) is None:
                    add_issue(
                        path="treatment_spec.treated",
                        message=f"treated value not parseable as float for numeric column {tcol!r}: {ts.treated!r}",
                        typ="invalid_numeric_literal",
                        val=ts.treated,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": p.get("dtype")},
                    )
                if parse_float(ts.control) is None:
                    add_issue(
                        path="treatment_spec.control",
                        message=f"control value not parseable as float for numeric column {tcol!r}: {ts.control!r}",
                        typ="invalid_numeric_literal",
                        val=ts.control,
                        severity="ERROR",
                        evidence={"column": tcol, "dtype": p.get("dtype")},
                    )
            elif k == "CATEGORICAL":
                check_value_membership_if_possible(col=tcol, path="treatment_spec.treated", value=ts.treated, label="Treatment")
                check_value_membership_if_possible(col=tcol, path="treatment_spec.control", value=ts.control, label="Treatment")
            else:
                add_issue(
                    path="treatment_spec.column",
                    message=f"Binary treatment expects BOOLEAN/CATEGORICAL/NUMERIC(0/1) column, got {k!r} for {tcol!r}",
                    typ="column_kind_mismatch",
                    val={"column": tcol, "inferred_kind": k},
                    severity="ERROR",
                    evidence={"dtype": p.get("dtype")},
                )

    elif isinstance(ts, CategoricalTreatmentSpecModel):
        tcol = ts.column
        p = prof(tcol)
        if p is not None:
            k = kind_of(p)
            if k != "CATEGORICAL":
                add_issue(
                    path="treatment_spec.column",
                    message=f"Categorical treatment requires CATEGORICAL column, got {k!r} for {tcol!r}",
                    typ="column_kind_mismatch",
                    val={"column": tcol, "inferred_kind": k},
                    severity="ERROR",
                    evidence={"dtype": p.get("dtype")},
                )
        # membership checks against dataset (if possible)
        for j, lvl in enumerate(ts.levels):
            check_value_membership_if_possible(col=tcol, path=f"treatment_spec.levels.{j}", value=lvl, label="Treatment level")

    elif isinstance(ts, ContinuousTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        tcol = ts.column
        p = prof(tcol)
        if p is not None and kind_of(p) != "NUMERIC":
            add_issue(
                path="treatment_spec.column",
                message=f"Continuous treatment requires NUMERIC column, got {kind_of(p)!r} for {tcol!r}",
                typ="column_kind_mismatch",
                val={"column": tcol, "inferred_kind": kind_of(p)},
                severity="ERROR",
                evidence={"dtype": p.get("dtype")},
            )

        if ts.clip_min is not None and ts.clip_max is not None and ts.clip_min > ts.clip_max:
            add_issue(
                path="treatment_spec",
                message=f"clip_min must be <= clip_max, got {ts.clip_min} > {ts.clip_max}",
                typ="invalid_clip_bounds",
                val={"clip_min": ts.clip_min, "clip_max": ts.clip_max},
                severity="ERROR",
            )

    # -------------------------
    # Outcome values validation
    # -------------------------
    ys = protocol.outcome_spec

    if isinstance(ys, BinaryOutcomeSpecModel):
        ycol = ys.column
        p = prof(ycol)
        if p is not None:
            k = kind_of(p)

            if norm(ys.event) == norm(ys.non_event):
                add_issue(
                    path="outcome_spec",
                    message=f"Binary outcome event and non_event must differ, got {ys.event!r} == {ys.non_event!r}",
                    typ="event_equals_non_event",
                    val={"event": ys.event, "non_event": ys.non_event},
                    severity="ERROR",
                )

            if k == "BOOLEAN":
                if not is_bool_literal(ys.event):
                    add_issue(
                        path="outcome_spec.event",
                        message=f"event value not boolean-like for boolean column {ycol!r}: {ys.event!r}",
                        typ="invalid_boolean_literal",
                        val=ys.event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": p.get("dtype")},
                    )
                if not is_bool_literal(ys.non_event):
                    add_issue(
                        path="outcome_spec.non_event",
                        message=f"non_event value not boolean-like for boolean column {ycol!r}: {ys.non_event!r}",
                        typ="invalid_boolean_literal",
                        val=ys.non_event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": p.get("dtype")},
                    )
            elif k == "NUMERIC":
                if parse_float(ys.event) is None:
                    add_issue(
                        path="outcome_spec.event",
                        message=f"event value not parseable as float for numeric column {ycol!r}: {ys.event!r}",
                        typ="invalid_numeric_literal",
                        val=ys.event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": p.get("dtype")},
                    )
                if parse_float(ys.non_event) is None:
                    add_issue(
                        path="outcome_spec.non_event",
                        message=f"non_event value not parseable as float for numeric column {ycol!r}: {ys.non_event!r}",
                        typ="invalid_numeric_literal",
                        val=ys.non_event,
                        severity="ERROR",
                        evidence={"column": ycol, "dtype": p.get("dtype")},
                    )
            elif k == "CATEGORICAL":
                check_value_membership_if_possible(col=ycol, path="outcome_spec.event", value=ys.event, label="Outcome")
                check_value_membership_if_possible(col=ycol, path="outcome_spec.non_event", value=ys.non_event, label="Outcome")
            else:
                add_issue(
                    path="outcome_spec.column",
                    message=f"Binary outcome expects BOOLEAN/CATEGORICAL/NUMERIC column, got {k!r} for {ycol!r}",
                    typ="column_kind_mismatch",
                    val={"column": ycol, "inferred_kind": k},
                    severity="ERROR",
                    evidence={"dtype": p.get("dtype")},
                )

    elif isinstance(ys, CategoricalOutcomeSpecModel):
        ycol = ys.column
        p = prof(ycol)
        if p is not None and kind_of(p) != "CATEGORICAL":
            add_issue(
                path="outcome_spec.column",
                message=f"Categorical outcome requires CATEGORICAL column, got {kind_of(p)!r} for {ycol!r}",
                typ="column_kind_mismatch",
                val={"column": ycol, "inferred_kind": kind_of(p)},
                severity="ERROR",
                evidence={"dtype": p.get("dtype")},
            )



        for j, lvl in enumerate(ys.levels):
            check_value_membership_if_possible(col=ycol, path=f"outcome_spec.levels.{j}", value=lvl, label="Outcome level")

    elif isinstance(ys, ContinuousOutcomeSpecModel):
        ycol = ys.column
        p = prof(ycol)
        if p is not None and kind_of(p) != "NUMERIC":
            add_issue(
                path="outcome_spec.column",
                message=f"Continuous outcome requires NUMERIC column, got {kind_of(p)!r} for {ycol!r}",
                typ="column_kind_mismatch",
                val={"column": ycol, "inferred_kind": kind_of(p)},
                severity="ERROR",
                evidence={"dtype": p.get("dtype")},
            )

        if ys.clip_min is not None and ys.clip_max is not None and ys.clip_min > ys.clip_max:
            add_issue(
                path="outcome_spec",
                message=f"clip_min must be <= clip_max, got {ys.clip_min} > {ys.clip_max}",
                typ="invalid_clip_bounds",
                val={"clip_min": ys.clip_min, "clip_max": ys.clip_max},
                severity="ERROR",
            )

    elif isinstance(ys, DurationOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        dcol = ys.duration_column
        ecol = ys.event_column

        dp = prof(dcol)
        if dp is not None and kind_of(dp) != "NUMERIC":
            add_issue(
                path="outcome_spec.duration_column",
                message=f"Duration column must be NUMERIC, got {kind_of(dp)!r} for {dcol!r}",
                typ="column_kind_mismatch",
                val={"column": dcol, "inferred_kind": kind_of(dp)},
                severity="ERROR",
                evidence={"dtype": dp.get("dtype")},
            )

        ep = prof(ecol)
        if ep is not None:
            k = kind_of(ep)

            if norm(ys.event_value) == norm(ys.censor_value):
                add_issue(
                    path="outcome_spec",
                    message=f"event_value and censor_value must differ, got {ys.event_value!r} == {ys.censor_value!r}",
                    typ="event_equals_censor",
                    val={"event_value": ys.event_value, "censor_value": ys.censor_value},
                    severity="ERROR",
                )

            if k == "BOOLEAN":
                if not is_bool_literal(ys.event_value):
                    add_issue(
                        path="outcome_spec.event_value",
                        message=f"event_value not boolean-like for boolean column {ecol!r}: {ys.event_value!r}",
                        typ="invalid_boolean_literal",
                        val=ys.event_value,
                        severity="ERROR",
                        evidence={"column": ecol, "dtype": ep.get("dtype")},
                    )
                if not is_bool_literal(ys.censor_value):
                    add_issue(
                        path="outcome_spec.censor_value",
                        message=f"censor_value not boolean-like for boolean column {ecol!r}: {ys.censor_value!r}",
                        typ="invalid_boolean_literal",
                        val=ys.censor_value,
                        severity="ERROR",
                        evidence={"column": ecol, "dtype": ep.get("dtype")},
                    )
            elif k == "NUMERIC":
                if parse_float(ys.event_value) is None:
                    add_issue(
                        path="outcome_spec.event_value",
                        message=f"event_value not parseable as float for numeric column {ecol!r}: {ys.event_value!r}",
                        typ="invalid_numeric_literal",
                        val=ys.event_value,
                        severity="ERROR",
                        evidence={"column": ecol, "dtype": ep.get("dtype")},
                    )
                if parse_float(ys.censor_value) is None:
                    add_issue(
                        path="outcome_spec.censor_value",
                        message=f"censor_value not parseable as float for numeric column {ecol!r}: {ys.censor_value!r}",
                        typ="invalid_numeric_literal",
                        val=ys.censor_value,
                        severity="ERROR",
                        evidence={"column": ecol, "dtype": ep.get("dtype")},
                    )
            elif k == "CATEGORICAL":
                check_value_membership_if_possible(col=ecol, path="outcome_spec.event_value", value=ys.event_value, label="Outcome event_value")
                check_value_membership_if_possible(col=ecol, path="outcome_spec.censor_value", value=ys.censor_value, label="Outcome censor_value")
            else:
                add_issue(
                    path="outcome_spec.event_column",
                    message=f"event_column kind should be BOOLEAN/CATEGORICAL/NUMERIC, got {k!r} for {ecol!r}",
                    typ="column_kind_mismatch",
                    val={"column": ecol, "inferred_kind": k},
                    severity="ERROR",
                    evidence={"dtype": ep.get("dtype")},
                )
                
    return issues

# =============================================================================
# error formatting
# =============================================================================

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
            }
        )
    return issues


def _issue_to_str(issue: Mapping[str, Any]) -> str:
    path = str(issue.get("path", "")).strip()
    msg = str(issue.get("message", "Invalid value")).strip()
    return f"{path}: {msg}" if path else msg


def _format_errors(errors: List[str], raw_json: str) -> str:
    e = "\n".join([f"- {x}" for x in (errors or ["Unknown error"])])
    snippet = (raw_json or "").strip()
    if len(snippet) > 1000:
        snippet = snippet[:1000] + "…"
    return f"Validation/compile errors:\n{e}\n\nLast JSON snippet:\n{snippet}"
