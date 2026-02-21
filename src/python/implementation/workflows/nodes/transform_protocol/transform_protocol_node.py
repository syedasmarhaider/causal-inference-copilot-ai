from __future__ import annotations

import json
from typing import Any, Dict, List,  Optional, Sequence, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_encoding import EncodingSpec, FeatureMapModel, apply_encoding, get_encoding_models_with_description
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_prompts import build_encoding_plan_system_prompt, build_encoding_plan_user_prompt_template, build_transformed_protocol_system_prompt, build_transformed_protocol_user_prompt_template
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import (
    EncodingType,
    TransformedProtocolSpec,
)
from python.implementation.workflows.tools.data.data_profiling_tool import CategoricalColumnProfileModel, DatasetSummaryModel
from python.implementation.workflows.utils.validation import ValidationIssueModel, ValidationSeverity
from python.implementation.workflows.utils.utils import json_sanitize


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
    history: Optional[List[Dict[str, Any]]] = None,
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
    encoding_catalog = get_encoding_models_with_description()
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
            history=None,
            max_attempts=max_attempts
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



def _is_fail(issue: ValidationIssueModel) -> bool:
    return issue.severity == "FAIL"


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
    """
    Applies EncodingPlanModel.decisions sequentially.

    Returns:
      (transformed_df, feature_map, issues)

    Behavior:
    - Never mutates input df.
    - Applies in plan order (deterministic).
    - For *_idx encodings, passes categories_in_order from dataset_summary.
    - Aggregates issues across columns.
    """
    out = df.copy()
    fmap = FeatureMapModel()
    issues: List[ValidationIssueModel] = []

    for d in plan.decisions:
        col = d.column
        spec: EncodingSpec = d.spec

        if col not in out.columns:
            miss = ValidationIssueModel(
                severity="FAIL",
                message=f"Encoding plan refers to missing column '{col}'.",
                evidence={"column": col, "encoding": getattr(spec, "encoding", type(spec).__name__)},
                fix_hint="Ensure plan columns come from dataset summary and are applied before dropping/renaming.",
            )
            issues.append(miss)
            if fail_fast:
                return out, fmap, issues
            continue

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
    max_attempts: int = 2,
) -> Tuple[Optional[TransformedProtocolSpec], List[ValidationIssueModel]]:
    """
    LLM step ONLY:
    - produce schema-valid TransformedProtocolSpec
    - do NOT enforce column existence/overlap invariants here (that’s the next validation gate)

    We still provide df_after_columns + feature_map to reduce hallucination.
    """
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
    fmap_json = json.dumps(json_sanitize(feature_map.model_dump(mode="python")), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

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
            history=None,
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