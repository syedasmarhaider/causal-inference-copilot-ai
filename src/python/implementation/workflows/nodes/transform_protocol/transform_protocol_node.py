from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_deps import TransformProtocolDeps
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_encoding import (
    EncodingSpec,
    FeatureMapModel,
    apply_encoding,
    get_encoding_models_with_description,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_prompts import (
    build_encoding_plan_system_prompt,
    build_encoding_plan_user_prompt_template,
    build_transformed_protocol_system_prompt,
    build_transformed_protocol_user_prompt_template,
    get_transform_protocol_node_info,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import (
    EncodingType,
    TransformedProtocolSpec,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import TransformProtocolPayloadModel, TransformProtocolState
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_validation import validate_binary_and_one_hot_invariants, validate_constant_or_near_constant_controls, validate_dimensionality_caps, validate_encoding_postconditions, validate_id_like_features_in_controls, validate_input_columns_exist_and_are_unambiguous, validate_model_inputs_are_numeric_dtypes, validate_treatment_outcome_domains_by_kind
from python.implementation.workflows.tools.data.data_profiling_tool import (
    CategoricalColumnProfileModel,
    DatasetProfilingStateTool,
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel, ValidationSeverity
from python.implementation.workflows.utils.utils import json_sanitize




def _repair_messages_from_issues(
    *,
    attempt: int,
    stage: str,
    issues: List[ValidationIssueModel],
) -> List[ChatMessage]:
    """
    Minimal “propagate error” into next attempt.
    Kept small: only FAIL issues + up to N.
    """
    fails = [x for x in issues if x.severity == "FAIL"]
    sample = (fails or issues)[:10]
    payload = { # pyright: ignore[reportUnknownVariableType]
        "attempt": attempt,
        "stage": stage,
        "n_issues": len(issues),
        "issues_sample": [x.model_dump(mode="json") for x in sample],
        "instruction": "On the next attempt, fix the JSON output to satisfy schema and constraints. Output JSON only.",
    }
    return [ChatMessage(role="system", content=f"REPAIR_CONTEXT={payload}")]


@dataclass(frozen=True)
class TransformProtocolNode(Node):
    NAME: ClassVar[str] = TransformProtocolState.NAME

    llm: LLMService
    data_repo: DataRepo
    model_name: str

    # global retry count for the WHOLE pipeline
    max_attempts: int = 2

    # knobs for deterministic steps
    profiling_max_categories: int = 50
    profiling_sample_distinct: int = 50
    fail_fast_apply: bool = False
    fail_fast_validate: bool = False
    validation_policy: Optional["TransformValidationPolicy"] = None  # optional

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_transform_protocol_node_info()

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
        # ------------------------------------------------------------------
        # Dependencies
        # ------------------------------------------------------------------
        data_profiling_tool = cast(DatasetProfilingStateTool, tool_factory.get_tool("DATA_PROFILING_TOOL"))                
        deps = TransformProtocolDeps.from_loaded(previous_state_dependencies)
        clean_state= deps.clean_protocol
        compile_state = deps.compile_protocol
        validate_clean_protcol = deps.validate_cleaned_protocol
        clean_dataset_validation_issues = validate_clean_protcol.payload.issues
        clean_dataset_id = clean_state.payload.clean_dataset_id
        assert clean_dataset_id is not None, "CleanProtocolState.payload.clean_dataset_id is required for TransformProtocolNode"
        protocol = compile_state.payload.protocol
        assert protocol is not None, "CompileProtocolState.payload.protocol is required for TransformProtocolNode"


        # ------------------------------------------------------------------
        # Load dataset once (no need to reload every attempt)
        # ------------------------------------------------------------------
        try:
            df_clean: pd.DataFrame = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_dataset_id,
            )
        except Exception as e:  # noqa: BLE001
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    error=f"Failed to read clean dataset: {e}",
                    user_message="Failed to read the cleaned dataset. Please check if the previous steps completed successfully and try again.",
                )
            )

        # ------------------------------------------------------------------
        # Profile once (no need to reprofile every attempt)
        # ------------------------------------------------------------------
        try:
            clean_dataset_summary: DatasetSummaryModel = data_profiling_tool.extract_dataset_summary(
                df_clean,
                max_categories=self.profiling_max_categories,
                sample_distinct=self.profiling_sample_distinct,
                compute_quantiles=True,
                strict=True,
            )
        except Exception as e:  # noqa: BLE001
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    error=f"Dataset profiling failed: {e}",
                    user_message="Failed to profile the cleaned dataset. Please check if the previous steps completed successfully and try again.",
                )
            )
            
        retry_hist: List[ChatMessage] = []  # accumulates repair context across attempts
        all_issues: List[ValidationIssueModel] = []
        last_error: Optional[str] = None

        # ==================================================================
        # WHOLE-PIPELINE retry loop
        # ==================================================================
        for attempt in range(1, max(1, int(self.max_attempts)) + 1):
            attempt_issues: List[ValidationIssueModel] = []
            last_error = None

            # ---------- Stage 1: encoding plan ----------
            plan, plan_issues = llm_generate_encoding_plan_from_protocol_and_summary(
                llm=self.llm,
                protocol=protocol,
                dataset_summary=clean_dataset_summary,
                supported_encodings=None,
                llm_config=LLMConfig(temperature=0.4),
                history=retry_hist,
                max_attempts=2,  
            )
            attempt_issues.extend(plan_issues)
            retry_as_there_a_issue = any(x.severity == "FAIL" for x in plan_issues)
            if retry_as_there_a_issue:
                last_error = f"Attempt {attempt}: encoding plan generation had issues."
                retry_hist.extend(_repair_messages_from_issues(attempt=attempt, stage="ENCODING_PLAN", issues=plan_issues))
                all_issues.extend(attempt_issues)
                continue
                
                

            if plan is None:
                last_error = f"Attempt {attempt}: encoding plan generation failed."
                retry_hist.extend(_repair_messages_from_issues(attempt=attempt, stage="ENCODING_PLAN", issues=attempt_issues))
                all_issues.extend(attempt_issues)
                continue

            # ---------- Stage 2: apply encoding ----------
            df_after, feature_map, apply_issues = apply_encoding_plan(
                df=df_clean,
                plan=plan,
                dataset_summary=clean_dataset_summary,
                fail_fast=self.fail_fast_apply,
            )
            attempt_issues.extend(apply_issues)

            if any(_is_fail(x) for x in apply_issues):
                last_error = f"Attempt {attempt}: encoding application failed."
                retry_hist.extend(_repair_messages_from_issues(attempt=attempt, stage="APPLY_ENCODING", issues=apply_issues))
                all_issues.extend(attempt_issues)
                continue

            # ---------- Stage 3: transformed protocol spec ----------
            spec, spec_issues = llm_generate_transformed_protocol_spec(
                llm=self.llm,
                protocol=protocol,
                df_after=df_after,
                feature_map=feature_map,
                history=retry_hist,
                llm_config=LLMConfig(temperature=0.2),
                max_attempts=1,  # node controls attempts
            )
            attempt_issues.extend(spec_issues)

            if spec is None:
                last_error = f"Attempt {attempt}: transformed spec generation failed."
                retry_hist.extend(_repair_messages_from_issues(attempt=attempt, stage="TRANSFORMED_SPEC", issues=spec_issues or attempt_issues))
                all_issues.extend(attempt_issues)
                continue

            # ---------- Stage 4: validations ----------
            val_issues = run_transform_validations(
                df_after=df_after,
                spec=spec,
                policy=self.validation_policy,
                fail_fast=self.fail_fast_validate,
            )
            attempt_issues.extend(val_issues)

            if any(_is_fail(x) for x in val_issues):
                last_error = f"Attempt {attempt}: validation failed."
                # This is the most useful feedback for the LLM on the next attempt
                retry_hist.extend(_repair_messages_from_issues(attempt=attempt, stage="VALIDATION", issues=val_issues))
                all_issues.extend(attempt_issues)
                continue

            # ---------- Stage 5: persist + success ----------
            try:
                transformed_dataset_id = uuid4()
                self.data_repo.save_csv_data(
                    df=df_after,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=transformed_dataset_id,
                )
                
            except Exception as e:  # noqa: BLE001
                last_error = f"Attempt {attempt}: failed to persist transformed dataset: {e}"
                persist_issue = _issue(
                    severity="FAIL",
                    message=str(last_error),
                    evidence={"attempt": attempt},
                )
                attempt_issues.append(persist_issue)
                all_issues.extend(attempt_issues)
                continue

            # SUCCESS
            payload = TransformProtocolPayloadModel(
                error=None,
                transformed_dataset_id=transformed_dataset_id,
                transformed_spec=spec,
                transformation_issues=attempt_issues,
                cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                cleaned_dataset_id=clean_dataset_id,
                cleaned_dataset_summary=clean_dataset_summary,
                user_message="Transform pipeline succeeded.",
            )
            return TransformProtocolState(payload)

        # ==================================================================
        # Exhausted attempts
        # ==================================================================
        payload = TransformProtocolPayloadModel(
            error=last_error or "Transform pipeline failed after retries.",
            transformed_dataset_id=None,
            transformed_spec=None,
            transformation_issues=all_issues,
            user_message="Transform pipeline failed after retries. Please check the previous steps and try again.",
        )
        return TransformProtocolState(payload)

# =============================================================================
# LLM output schema (strict)
# =============================================================================

class ColumnEncodingDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    spec: EncodingSpec
    rationale: Optional[str] = None

    @field_validator("column", mode="before")
    @classmethod
    def _strip_nonempty(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise TypeError("column must be str")
        s = v.strip()
        if not s:
            raise ValueError("column must be non-empty")
        return s


class EncodingPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    decisions: List[ColumnEncodingDecisionModel] = Field(..., min_length=1)


# =============================================================================
# Helper: infer Y/T/W/X raw columns from ProtocolSpec
# =============================================================================

def _infer_roles_from_protocol(protocol: ProtocolSpec) -> Dict[str, List[str]]:
    t_col = protocol.treatment_spec.column
    if protocol.outcome_spec.kind == "duration":
        y_cols = [protocol.outcome_spec.duration_column, protocol.outcome_spec.event_column]
    else:
        y_cols = [protocol.outcome_spec.column]

    w_cols = list(protocol.covariates or [])
    x_cols = list(protocol.effect_modifiers or [])

    return {"Y": y_cols, "T": [t_col], "W": w_cols, "X": x_cols}


def _issue(
    *,
    severity: ValidationSeverity,
    message: str,
    evidence: Optional[Dict[str, Any]] = None,
    fix_hint: Optional[str] = None,
) -> ValidationIssueModel:
    return ValidationIssueModel(
        severity=severity,
        message=message,
        evidence=evidence or {},
        fix_hint=fix_hint,
    )


def _extract_columns(dataset_summary: DatasetSummaryModel) -> List[str]:
    cols: List[str] = []
    for p in dataset_summary.profiles:
        n = getattr(p, "name", None)
        if isinstance(n, str):
            s = n.strip()
            if s:
                cols.append(s)
    return cols



# TODO: fix later
def _encoding_catalog_with_idx_semantics() -> str:
    base = get_encoding_models_with_description()
    return (
        f"{base}\n"
        "\n"
        "IMPORTANT index semantics for *_idx encodings:\n"
        "- For binary_map_idx / ordinal_map_idx, indices refer to the order of "
        "dataset_summary.profiles[*].summary.top_categories.\n"
        "- Index 0 -> top_categories[0].value, index 1 -> top_categories[1].value, etc.\n"
        "- This is a TOP-K list only (not necessarily the full domain).\n"
    )


def _build_user_prompt(
    *,
    columns: List[str],
    roles: Dict[str, List[str]],
    protocol_json_obj: Dict[str, Any],
    dataset_summary_json_obj: Dict[str, Any],
    encoding_catalog: str,
) -> str:
    tmpl = build_encoding_plan_user_prompt_template()
    return tmpl.format(
        encoding_catalog_text=encoding_catalog,
        columns_json=json.dumps(columns, ensure_ascii=False),
        protocol_json=json.dumps(protocol_json_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        roles_json=json.dumps(roles, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        summary_json=json.dumps(dataset_summary_json_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def llm_generate_encoding_plan_from_protocol_and_summary(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    dataset_summary: DatasetSummaryModel,
    supported_encodings: Optional[Sequence[EncodingType]] = None,
    llm_config: Optional[LLMConfig] = None,
    history: Optional[Sequence[ChatMessage]] = None,
    max_attempts: int = 2,
) -> Tuple[Optional[EncodingPlanModel], List[ValidationIssueModel]]:
    if not dataset_summary.profiles:
        return None, [
            _issue(
                severity="FAIL",
                message="Encoding plan: dataset_summary.profiles missing or empty.",
                evidence={"n_profiles": 0},
                fix_hint="Provide DatasetSummaryModel with a non-empty profiles list.",
            )
        ]

    columns = _extract_columns(dataset_summary)
    if not columns:
        return None, [
            _issue(
                severity="FAIL",
                message="Encoding plan: no column names found in dataset_summary.profiles.",
                fix_hint="Ensure each profile has a non-empty name.",
            )
        ]

    roles = _infer_roles_from_protocol(protocol)

    encoding_catalog = _encoding_catalog_with_idx_semantics()

    system_prompt = build_encoding_plan_system_prompt()

    protocol_obj: Dict[str, Any] = protocol.model_dump(mode="json")
    summary_obj: Dict[str, Any] = json_sanitize(dataset_summary.model_dump(mode="python"))

    user_prompt = _build_user_prompt(
        columns=columns,
        roles=roles,
        protocol_json_obj=protocol_obj,
        dataset_summary_json_obj=summary_obj,
        encoding_catalog=encoding_catalog,
    )

    cfg = llm_config or LLMConfig(
        temperature=0.4,
    )

    try:
        plan = llm.generate_json(
            schema=EncodingPlanModel,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=cfg,
            history=history,
            max_attempts=max_attempts,
        )
    except Exception as e:  # noqa: BLE001
        return None, [
            _issue(
                severity="FAIL",
                message=f"Encoding plan: LLM JSON generation failed: {e}",
                evidence={"n_columns": len(columns)},
                fix_hint="check schema mismatch while generating the plan",
            )
        ]

    provided = set(columns)
    seen: set[str] = set()
    unknown_cols: List[str] = []
    dup_cols: List[str] = []

    for d in plan.decisions:
        if d.column not in provided:
            unknown_cols.append(d.column)
        if d.column in seen:
            dup_cols.append(d.column)
        seen.add(d.column)

    issues: List[ValidationIssueModel] = []
    if unknown_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Encoding plan: LLM referenced columns not present in dataset_summary.",
                evidence={"unknown_columns": sorted(set(unknown_cols))[:50]},
                fix_hint="LLM must only pick from the provided column list.",
            )
        )
    if dup_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Encoding plan: duplicate decisions for the same column.",
                evidence={"duplicate_columns": sorted(set(dup_cols))[:50]},
                fix_hint="LLM must output at most one decision per column.",
            )
        )

    if issues:
        return None, issues

    return plan, []


# =============================================================================
# LLM output schema + validation for transformed protocol spec
# =============================================================================
def _categories_in_order_from_summary(
    *,
    dataset_summary: DatasetSummaryModel,
    column: str,
) -> Optional[List[str]]:
    for p in dataset_summary.profiles:
        if getattr(p, "name", None) == column and isinstance(p, CategoricalColumnProfileModel):
            top = p.summary.top_categories
            return [c.value for c in top]
    return None


def _warn_if_idx_on_truncated_topk(
    *,
    dataset_summary: DatasetSummaryModel,
    column: str,
    encoding: str,
) -> Optional[ValidationIssueModel]:
    """
    ONLY CHANGE (w.r.t #4): warn when *_idx is used but categorical profile is truncated.
    This makes the risk explicit without changing behavior.
    """
    for p in dataset_summary.profiles:
        if getattr(p, "name", None) == column and isinstance(p, CategoricalColumnProfileModel):
            other = int(p.summary.other_count)
            top_k = len(p.summary.top_categories)
            if other > 0:
                return _issue(
                    severity="WARN",
                    message=f"{encoding} uses indices into top_categories only; '{column}' has truncated categorical profile (other_count>0).",
                    evidence={"column": column, "encoding": encoding, "top_k": top_k, "other_count": other},
                    fix_hint="Prefer label-based mapping or increase profiling max_categories/store a full categories list.",
                )
            return None
    return None


def _merge_feature_maps(base: FeatureMapModel, add: FeatureMapModel) -> FeatureMapModel:
    produced = dict(base.produced_columns)
    dropped = list(base.dropped)

    for k, v in add.produced_columns.items():
        produced[k] = list(v)

    for c in add.dropped:
        if c not in dropped:
            dropped.append(c)

    return FeatureMapModel(produced_columns=produced, dropped=dropped)


def apply_encoding_plan(
    *,
    df: pd.DataFrame,
    plan: "EncodingPlanModel",
    dataset_summary: DatasetSummaryModel,
    fail_fast: bool = False,
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    out = df.copy()
    fmap = FeatureMapModel()
    issues: List[ValidationIssueModel] = []

    for d in plan.decisions:
        col = d.column
        spec: EncodingSpec = d.spec
        enc = getattr(spec, "encoding", type(spec).__name__)

        if col not in out.columns:
            miss = ValidationIssueModel(
                severity="FAIL",
                message=f"Encoding plan refers to missing column '{col}'.",
                evidence={"column": col, "encoding": enc},
                fix_hint="Ensure plan columns come from dataset summary and are applied before dropping/renaming.",
            )
            issues.append(miss)
            if fail_fast:
                return out, fmap, issues
            continue

        # ONLY CHANGE (w.r.t #4): warn when *_idx indices are top-K and domain is truncated
        if enc in ("binary_map_idx", "ordinal_map_idx"):
            w = _warn_if_idx_on_truncated_topk(dataset_summary=dataset_summary, column=col, encoding=enc)
            if w is not None:
                issues.append(w)

        cats: Optional[Sequence[str]] = _categories_in_order_from_summary(
            dataset_summary=dataset_summary,
            column=col,
        )

        out2, fmap2, iss2 = apply_encoding(
            df=out,
            column=col,
            spec=spec,
            categories_in_order=cats,
        )

        out = out2
        fmap = _merge_feature_maps(fmap, fmap2)
        issues.extend(iss2)

        if fail_fast and any(_is_fail(x) for x in iss2):
            return out, fmap, issues

    return out, fmap, issues


def llm_generate_transformed_protocol_spec(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    df_after: pd.DataFrame,
    feature_map: FeatureMapModel,
    llm_config: Optional[LLMConfig] = None,
    history: Optional[Sequence[ChatMessage]] = None,
    max_attempts: int = 2,
) -> Tuple[Optional[TransformedProtocolSpec], List[ValidationIssueModel]]:
    df_cols = [str(c) for c in list(df_after.columns)]
    if not df_cols:
        return None, [
            _issue(
                severity="FAIL",
                message="TransformedProtocolSpec: df_after has no columns.",
                fix_hint="Apply encoding plan first and ensure df_after is non-empty.",
            )
        ]

    system_prompt = build_transformed_protocol_system_prompt()
    tmpl = build_transformed_protocol_user_prompt_template()

    protocol_json = json.dumps(protocol.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    df_cols_json = json.dumps(df_cols, ensure_ascii=False)
    fmap_json = json.dumps(
        json_sanitize(feature_map.model_dump(mode="python")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    user_prompt = tmpl.format(
        protocol_json=protocol_json,
        df_after_columns_json=df_cols_json,
        feature_map_json=fmap_json,
    )

    cfg = llm_config or LLMConfig(temperature=0.2)

    try:
        spec = llm.generate_json(
            schema=TransformedProtocolSpec,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=cfg,
            history=history,
            max_attempts=max_attempts,
        )
    except Exception as e:  # noqa: BLE001
        return None, [
            _issue(
                severity="FAIL",
                message=f"TransformedProtocolSpec: LLM JSON generation failed: {e}",
                evidence={"n_df_after_cols": len(df_cols)},
                fix_hint="Check schema mismatch; reduce prompt size; ensure JSON-only instruction.",
            )
        ]

    return spec, []



@dataclass(frozen=True)
class TransformValidationPolicy:
    # --- Validation #2
    allow_bool_inputs: bool = True

    # --- Validation #3
    binary_allowed_values: tuple[int, int] = (0, 1)
    min_variance: float = 1e-12
    duration_min_value: float = 0.0

    # --- Validation #4
    value_tol: float = 1e-9
    check_one_hot_group_row_sums: bool = True
    allow_zero_sum_rows: bool = True

    # --- Validation #5
    uniqueness_ratio_threshold: float = 0.98
    max_allowed_id_like: int = 0  # 0 => FAIL if any

    # --- Validation #6
    controls_min_variance: float = 1e-12
    max_constant_allowed: Optional[int] = None
    skip_binary_one_hot_in_constant_check: bool = True

    # --- Validation #7
    max_total_features: Optional[int] = 5000
    max_w_features: Optional[int] = None
    max_x_features: Optional[int] = None
    max_features_per_source_raw: Optional[int] = None

    # --- Validation #8
    minmax_tol: float = 1e-6
    zscore_warn_abs: float = 10.0
    zscore_fail_abs: Optional[float] = None


def _is_fail(issue: ValidationIssueModel) -> bool:
    return issue.severity == "FAIL"


def run_transform_validations(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    policy: Optional[TransformValidationPolicy] = None,
    fail_fast: bool = False,
) -> List[ValidationIssueModel]:
    """
    Runs validation suite in deterministic order.

    Order matters:
      #1 schema referential integrity
      #2 dtype numeric
      #3 Y/T domain checks
      #4 binary/one-hot invariants
      #5 id-like controls
      #6 constant-like controls
      #7 dimensionality caps
      #8 encoding post-conditions

    If fail_fast=True, returns immediately after first FAIL appears.
    """
    pol = policy or TransformValidationPolicy()
    issues: List[ValidationIssueModel] = []

    def _extend(new_issues: List[ValidationIssueModel]) -> None:
        nonlocal issues
        if not new_issues:
            return
        issues.extend(new_issues)
        if fail_fast and any(_is_fail(x) for x in new_issues):
            raise _StopValidation()

    class _StopValidation(Exception):
        pass

    try:
        _extend(
            validate_input_columns_exist_and_are_unambiguous(
                df_after=df_after,
                spec=spec,
            )
        )

        _extend(
            validate_model_inputs_are_numeric_dtypes(
                df_after=df_after,
                spec=spec,
                allow_bool=pol.allow_bool_inputs,
            )
        )

        _extend(
            validate_treatment_outcome_domains_by_kind(
                df_after=df_after,
                spec=spec,
                binary_allowed_values=pol.binary_allowed_values,
                min_variance=pol.min_variance,
                duration_min_value=pol.duration_min_value,
            )
        )

        _extend(
            validate_binary_and_one_hot_invariants(
                df_after=df_after,
                spec=spec,
                value_tol=pol.value_tol,
                check_one_hot_group_row_sums=pol.check_one_hot_group_row_sums,
                allow_zero_sum_rows=pol.allow_zero_sum_rows,
            )
        )

        _extend(
            validate_id_like_features_in_controls(
                df_after=df_after,
                spec=spec,
                uniqueness_ratio_threshold=pol.uniqueness_ratio_threshold,
                max_allowed_id_like=pol.max_allowed_id_like,
            )
        )

        _extend(
            validate_constant_or_near_constant_controls(
                df_after=df_after,
                spec=spec,
                min_variance=pol.controls_min_variance,
                max_constant_allowed=pol.max_constant_allowed,
                skip_binary_one_hot=pol.skip_binary_one_hot_in_constant_check,
            )
        )

        _extend(
            validate_dimensionality_caps(
                df_after=df_after,
                spec=spec,
                max_total_features=pol.max_total_features,
                max_w_features=pol.max_w_features,
                max_x_features=pol.max_x_features,
                max_features_per_source_raw=pol.max_features_per_source_raw,
            )
        )

        _extend(
            validate_encoding_postconditions(
                df_after=df_after,
                spec=spec,
                minmax_tol=pol.minmax_tol,
                zscore_warn_abs=pol.zscore_warn_abs,
                zscore_fail_abs=pol.zscore_fail_abs,
            )
        )

    except _StopValidation:
        pass

    return issues