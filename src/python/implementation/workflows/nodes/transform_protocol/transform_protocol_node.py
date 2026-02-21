from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_encoding import EncodingSpec, _issue
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import EncodingType
from python.implementation.workflows.utils.dataset_utils import DatasetSummary
from python.implementation.workflows.utils.validation import ValidationIssueModel
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec


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
    # Treatment
    t_col = protocol.treatment_spec.column  # binary/continuous/categorical all have .column in your model

    # Outcome
    if protocol.outcome_spec.kind == "duration":
        # duration outcome uses 2 raw columns
        y_cols = [protocol.outcome_spec.duration_column, protocol.outcome_spec.event_column]
    else:
        y_cols = [protocol.outcome_spec.column]

    # W / X
    w_cols = list(protocol.covariates or [])
    x_cols = list(protocol.effect_modifiers or [])

    return {"Y": y_cols, "T": [t_col], "W": w_cols, "X": x_cols}


# =============================================================================
# ONE public function: generate encoding plan (no application)
# =============================================================================
def llm_generate_encoding_plan_from_protocol_and_summary(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    dataset_summary: DatasetSummary,
    supported_encodings: Optional[Sequence[EncodingType]] = None,
    llm_config: Optional[LLMConfig] = None,
    max_attempts: int = 2,
) -> Tuple[Optional[EncodingPlanModel], List[ValidationIssueModel]]:
    issues: List[ValidationIssueModel] = []
    profiles = dataset_summary.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        return None, [
            _issue(
                severity="FAIL",
                message="Encoding plan: dataset_summary.profiles missing or empty.",
                evidence={"keys": sorted(list(dataset_summary.keys()))},
                fix_hint="Provide DatasetSummary with a non-empty 'profiles' list.",
            )
        ]

    # Column list comes from summary (deterministic order)
    columns: List[str] = []
    for p in profiles:
        if isinstance(p, dict) and isinstance(p.get("name"), str):
            name = p["name"].strip()
            if name:
                columns.append(name)

    if not columns:
        return None, [
            _issue(
                severity="FAIL",
                message="Encoding plan: no column names found in dataset_summary.profiles.",
            )
        ]

    # ---- Infer Y/T/W/X from protocol ----
    roles = _infer_roles_from_protocol(protocol)

    # ---- Supported encodings (static whitelist) ----
    allowed = list(supported_encodings) if supported_encodings is not None else get_supported_encodings_model().encodings
    encoding_catalog = "\n".join(f"- {enc}: {DESCRIPTIONS[enc]}" for enc in allowed)

    # ---- Prompts (externalized to prompt folder) ----
    system_prompt = build_encoding_plan_system_prompt()

    user_prompt = build_encoding_plan_user_prompt(
        columns=columns,
        protocol=protocol.model_dump(mode="json"),
        roles=roles,
        dataset_summary=dataset_summary,
        supported_encodings=allowed,
        encoding_catalog_text=encoding_catalog,
    )

    cfg = llm_config or LLMConfig()

    # ---- Call LLM with strict JSON schema ----
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
                fix_hint="Reduce prompt size; ensure prompt asks for JSON only; check schema mismatch.",
            )
        ]

    # ---- Cheap deterministic sanity checks (must be static) ----
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


def _issue(
    *,
    severity: Literal["WARN", "FAIL"],
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