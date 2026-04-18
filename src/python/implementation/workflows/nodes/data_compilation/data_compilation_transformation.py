from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    batch_transform_prompt,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    DatasetSummaryModel,
    DatetimeColumnProfileModel,
    NumericColumnProfileModel,
    OtherColumnProfileModel,
)


@dataclass(frozen=True)
class TransformationResult:
    transformation_plan: TransformPlan | None
    required_dataset_changes: DatasetRepairPlan | None


class DatasetRepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    role: Literal["covariate", "effect_modifier"]
    problem: Literal[
        "missing_values",
        "numeric_coded_category",
        "unsupported_dtype",
        "ungrounded_mapping",
        "ungrounded_order",
        "other",
    ]
    action: Literal[
        "impute_missing",
        "normalize_dtype",
        "normalize_categorical_representation",
        "recode_values",
    ]
    reason: str = Field(..., min_length=1)
    repair_instruction: str = Field(..., min_length=1)
    user_explanation: str | None = None

    @model_validator(mode="after")
    def _validate_action_problem_pair(self) -> "DatasetRepairAction":
        if self.problem == "missing_values" and self.action != "impute_missing":
            raise ValueError(
                "missing_values blockers must use action='impute_missing'"
            )
        if self.action == "impute_missing" and self.problem != "missing_values":
            raise ValueError(
                "action='impute_missing' is only allowed for problem='missing_values'"
            )
        return self


class DatasetRepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[DatasetRepairAction] = Field(..., min_length=1)


class _DraftColumnBase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    role: Literal["covariate", "effect_modifier"]


class _PlanDraftColumn(_DraftColumnBase):
    decision: Literal["plan"]
    preset: Literal[
        "drop",
        "passthrough",
        "cat_onehot",
        "num_standard",
        "num_minmax",
        "num_log1p",
        "datetime_epoch_seconds",
        "map_binary",
        "map_ordinal",
    ]
    mapping: dict[str, float] | None = None
    order: list[str] | None = None

    @model_validator(mode="after")
    def _validate_plan_column(self) -> _PlanDraftColumn:
        if self.preset == "map_binary" and not self.mapping:
            raise ValueError("map_binary requires a grounded mapping")
        if self.preset == "map_ordinal" and not self.order:
            raise ValueError("map_ordinal requires a grounded order")
        if self.preset != "map_binary" and self.mapping is not None:
            raise ValueError("mapping is only allowed for map_binary")
        if self.preset != "map_ordinal" and self.order is not None:
            raise ValueError("order is only allowed for map_ordinal")
        return self


class _DatasetChangeDraftColumn(_DraftColumnBase):
    decision: Literal["dataset_change"]
    problem: Literal[
        "missing_values",
        "numeric_coded_category",
        "unsupported_dtype",
        "ungrounded_mapping",
        "ungrounded_order",
        "other",
    ]
    action: Literal[
        "impute_missing",
        "normalize_dtype",
        "normalize_categorical_representation",
        "recode_values",
    ]
    reason: str = Field(..., min_length=1)
    repair_instruction: str = Field(..., min_length=1)
    user_explanation: str | None = None

    @model_validator(mode="after")
    def _validate_dataset_change(self) -> "_DatasetChangeDraftColumn":
        if self.problem == "missing_values" and self.action != "impute_missing":
            raise ValueError(
                "missing_values blockers must use action='impute_missing'"
            )
        if self.action == "impute_missing" and self.problem != "missing_values":
            raise ValueError(
                "action='impute_missing' is only allowed for problem='missing_values'"
            )
        return self


_DraftColumn = Annotated[
    _PlanDraftColumn | _DatasetChangeDraftColumn,
    Field(discriminator="decision"),
]


class _BatchTransformDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[_DraftColumn] = Field(..., min_length=1)


def transform(
    *,
    transformation_instructions: str,
    causal_spec: CausalSpec,
    data_summary: DatasetSummaryModel,
    llm: LLMService,
) -> TransformationResult:
    expected_role_by_column = _protocol_scope_role_by_column(causal_spec)
    if not expected_role_by_column:
        return TransformationResult(
            transformation_plan=None,
            required_dataset_changes=None,
        )

    scoped_summary = _eligible_dataset_summary(
        data_summary,
        include_columns=list(expected_role_by_column.keys()),
    )
    encoding_plan_tool = EncodingPlanTool()

    return _generate_batch_result(
        llm=llm,
        transformation_instructions=transformation_instructions,
        causal_spec=causal_spec,
        scoped_summary=scoped_summary,
        expected_role_by_column=expected_role_by_column,
        encoding_plan_tool=encoding_plan_tool,
    )


def _generate_batch_result(
    *,
    llm: LLMService,
    transformation_instructions: str,
    causal_spec: CausalSpec,
    scoped_summary: DatasetSummaryModel,
    expected_role_by_column: dict[str, Literal["covariate", "effect_modifier"]],
    encoding_plan_tool: EncodingPlanTool,
) -> TransformationResult:
    repair_request: str | None = None

    for _ in range(2):
        try:
            draft = llm.generate_json(
                schema=_BatchTransformDraft,
                system_prompt=batch_transform_prompt(),
                user_prompt=json.dumps(
                    _batch_prompt_payload(
                        transformation_instructions=transformation_instructions,
                        causal_spec=causal_spec,
                        scoped_summary=scoped_summary,
                        expected_role_by_column=expected_role_by_column,
                        repair_request=repair_request,
                    ),
                    ensure_ascii=False,
                ),
                config=LLMConfig(
                    model="basic",
                    temperature=0.2,
                    max_tokens=8000,
                ),
                history=None,
                max_attempts=2,
            )
            return _result_from_draft(
                draft_columns=draft.columns,
                scoped_summary=scoped_summary,
                expected_role_by_column=expected_role_by_column,
                encoding_plan_tool=encoding_plan_tool,
            )
        except Exception as exc:
            repair_request = _exception_chain_text(exc)

    raise ValueError(
        "batch transformation draft failed after retry: "
        f"{repair_request or 'unknown batch generation error'}"
    )


def _result_from_draft(
    *,
    draft_columns: Sequence[_DraftColumn],
    scoped_summary: DatasetSummaryModel,
    expected_role_by_column: dict[str, Literal["covariate", "effect_modifier"]],
    encoding_plan_tool: EncodingPlanTool,
) -> TransformationResult:
    column_drafts_by_name = _validate_and_index_draft_columns(
        draft_columns=draft_columns,
        expected_role_by_column=expected_role_by_column,
    )

    blockers = [
        draft
        for draft in column_drafts_by_name.values()
        if isinstance(draft, _DatasetChangeDraftColumn)
    ]
    if blockers:
        return TransformationResult(
            transformation_plan=None,
            required_dataset_changes=_build_dataset_repair_plan(blockers),
        )

    payload = _materialize_transform_plan_payload(
        planned_columns=[
            column_drafts_by_name[column]
            for column in expected_role_by_column
            if isinstance(column_drafts_by_name[column], _PlanDraftColumn)
        ],  # type: ignore[arg-type]
        scoped_summary=scoped_summary,
    )
    model_dict, issues = encoding_plan_tool.validate_encoding_payload_structured(
        payload=payload,
        data_summary=scoped_summary,
        covariate_columns=[
            column
            for column, role in expected_role_by_column.items()
            if role == "covariate"
        ],
        effect_modifier_columns=[
            column
            for column, role in expected_role_by_column.items()
            if role == "effect_modifier"
        ],
    )
    if issues:
        raise ValueError(_format_structured_issues(issues))
    if model_dict is None:
        raise ValueError("encoding validation returned no plan and no issues")

    validated_plan = encoding_plan_tool.validate_encoding_payload(
        payload=payload,
        data_summary=scoped_summary,
        covariate_columns=[
            column
            for column, role in expected_role_by_column.items()
            if role == "covariate"
        ],
        effect_modifier_columns=[
            column
            for column, role in expected_role_by_column.items()
            if role == "effect_modifier"
        ],
    )
    return TransformationResult(
        transformation_plan=validated_plan,
        required_dataset_changes=None,
    )


def _validate_and_index_draft_columns(
    *,
    draft_columns: Sequence[_DraftColumn],
    expected_role_by_column: dict[str, Literal["covariate", "effect_modifier"]],
) -> dict[str, _DraftColumn]:
    indexed: dict[str, _DraftColumn] = {}
    duplicates: list[str] = []

    for draft in draft_columns:
        column = str(draft.column).strip()
        role = str(draft.role).strip()
        expected_role = expected_role_by_column.get(column)
        if expected_role is None:
            raise ValueError(f"draft contains non-eligible column: {column}")
        if role != expected_role:
            raise ValueError(
                f"draft assigned wrong role for column '{column}': expected "
                f"{expected_role}, got {role}"
            )
        if column in indexed:
            duplicates.append(column)
            continue
        indexed[column] = draft

    if duplicates:
        raise ValueError(f"draft contains duplicate column entries: {sorted(duplicates)}")

    missing_columns = sorted(set(expected_role_by_column) - set(indexed))
    if missing_columns:
        raise ValueError(f"draft is missing eligible columns: {missing_columns}")

    return indexed


def _materialize_transform_plan_payload(
    *,
    planned_columns: Sequence[_PlanDraftColumn],
    scoped_summary: DatasetSummaryModel,
) -> dict[str, Any]:
    profiles_by_name = {
        str(profile.name).strip(): profile for profile in scoped_summary.profiles
    }
    payload_columns: list[dict[str, Any]] = []

    for draft_column in planned_columns:
        column = str(draft_column.column).strip()
        profile = profiles_by_name.get(column)
        if profile is None:
            raise ValueError(f"draft references unknown dataset summary column: {column}")
        payload_columns.append(
            {
                "column": column,
                "role": str(draft_column.role),
                "encoding": _materialize_encoding_payload_from_draft_column(
                    draft_column=draft_column,
                    profile=profile,
                ),
            }
        )

    return {"columns": payload_columns}


def _materialize_encoding_payload_from_draft_column(
    *,
    draft_column: _PlanDraftColumn,
    profile: (
        NumericColumnProfileModel
        | DatetimeColumnProfileModel
        | BooleanColumnProfileModel
        | CategoricalColumnProfileModel
        | OtherColumnProfileModel
    ),
) -> dict[str, Any]:
    _ = profile
    match draft_column.preset:
        case "drop":
            return {"preset": "drop"}
        case "passthrough":
            return {"preset": "passthrough"}
        case "cat_onehot":
            return {
                "preset": "cat_onehot",
                "drop_first": False,
                "handle_unknown": "ignore",
                "missing": "impute_token",
                "missing_token": "__MISSING__",
            }
        case "num_standard":
            return {
                "preset": "num_standard",
                "impute": "median",
                "add_missing_indicator": True,
            }
        case "num_minmax":
            return {
                "preset": "num_minmax",
                "impute": "median",
                "add_missing_indicator": True,
                "eps": 1e-12,
            }
        case "num_log1p":
            return {
                "preset": "num_log1p",
                "impute": "median",
                "add_missing_indicator": True,
                "allow_negative": False,
                "then_scale": "none",
            }
        case "datetime_epoch_seconds":
            return {
                "preset": "datetime_epoch_seconds",
                "errors": "coerce",
                "unit": "s",
                "add_missing_indicator": True,
            }
        case "map_binary":
            if not draft_column.mapping:
                raise ValueError(
                    f"map_binary requires a grounded mapping for column '{draft_column.column}'"
                )
            return {
                "preset": "map_binary",
                "mapping": draft_column.mapping,
                "allow_unknown": True,
                "unknown_value": -1.0,
                "missing": "as_unknown",
            }
        case "map_ordinal":
            if not draft_column.order:
                raise ValueError(
                    f"map_ordinal requires a grounded order for column '{draft_column.column}'"
                )
            return {
                "preset": "map_ordinal",
                "order": draft_column.order,
                "start": 0,
                "allow_unknown": True,
                "unknown_value": -1,
                "missing": "as_unknown",
            }
        case _:
            raise ValueError(
                f"unsupported preset '{draft_column.preset}' for column '{draft_column.column}'"
            )


def _protocol_scope_role_by_column(
    causal_spec: CausalSpec,
) -> dict[str, Literal["covariate", "effect_modifier"]]:
    role_by_column: dict[str, Literal["covariate", "effect_modifier"]] = {}
    for column in causal_spec.covariates:
        normalized = str(column).strip()
        if normalized:
            role_by_column[normalized] = "covariate"
    for column in causal_spec.effect_modifiers:
        normalized = str(column).strip()
        if normalized:
            role_by_column[normalized] = "effect_modifier"
    return role_by_column


def _eligible_dataset_summary(
    summary: DatasetSummaryModel,
    *,
    include_columns: Sequence[str],
) -> DatasetSummaryModel:
    profiles_by_name = {
        str(profile.name).strip(): profile
        for profile in summary.profiles
        if str(profile.name).strip()
    }
    scoped_profiles = []

    for column in include_columns:
        normalized = str(column).strip()
        if not normalized:
            continue
        profile = profiles_by_name.get(normalized)
        if profile is None:
            raise ValueError(
                f"dataset summary is missing eligible transformation column: {normalized}"
            )
        scoped_profiles.append(profile)

    return DatasetSummaryModel(
        n_rows=summary.n_rows,
        profiles=scoped_profiles,
    )


def _batch_prompt_payload(
    *,
    transformation_instructions: str,
    causal_spec: CausalSpec,
    scoped_summary: DatasetSummaryModel,
    expected_role_by_column: dict[str, Literal["covariate", "effect_modifier"]],
    repair_request: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transformation_instructions": _normalize_text(transformation_instructions),
        "compiled_causal_specification": causal_spec.model_dump(mode="json"),
        "scoped_dataset_summary": _dataset_summary_prompt_payload(scoped_summary),
        "eligible_columns": list(expected_role_by_column.keys()),
        "expected_role_by_column": expected_role_by_column,
        "required_plan_column_count": len(expected_role_by_column),
    }
    if repair_request:
        payload["repair_request"] = repair_request
    return payload


def _dataset_summary_prompt_payload(summary: DatasetSummaryModel) -> dict[str, Any]:
    return {
        "n_rows": summary.n_rows,
        "columns": [_column_prompt_payload(profile) for profile in summary.profiles],
    }


def _column_prompt_payload(
    profile: (
        NumericColumnProfileModel
        | DatetimeColumnProfileModel
        | BooleanColumnProfileModel
        | CategoricalColumnProfileModel
        | OtherColumnProfileModel
    ),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": str(profile.name).strip(),
        "kind": str(profile.inferred_kind),
        "dtype": profile.dtype,
        "missing_rate": profile.missing_rate,
        "distinct_count": profile.distinct_count,
    }

    if isinstance(profile, NumericColumnProfileModel):
        payload["range"] = {
            "min": profile.summary.min,
            "max": profile.summary.max,
        }
        return payload

    if isinstance(profile, DatetimeColumnProfileModel):
        payload["range"] = {
            "min": profile.summary.min,
            "max": profile.summary.max,
        }
        return payload

    if isinstance(profile, BooleanColumnProfileModel):
        payload["known_values"] = list(profile.summary.counts.keys())
        return payload

    if isinstance(profile, CategoricalColumnProfileModel):
        payload["top_values"] = [
            item.value for item in profile.summary.top_categories
        ]
        return payload

    if isinstance(profile, OtherColumnProfileModel):
        payload["sample_values"] = list(profile.summary.distinct_values_sample)
        return payload

    return payload


def _build_dataset_repair_plan(
    blockers: Sequence[_DatasetChangeDraftColumn],
) -> DatasetRepairPlan:
    return DatasetRepairPlan(
        actions=[
            DatasetRepairAction(
                column=str(blocker.column).strip(),
                role=blocker.role,
                problem=blocker.problem,
                action=blocker.action,
                reason=str(blocker.reason).strip(),
                repair_instruction=str(blocker.repair_instruction).strip(),
                user_explanation=(
                    str(blocker.user_explanation).strip()
                    if blocker.user_explanation is not None
                    else None
                ),
            )
            for blocker in blockers
        ]
    )


def _format_structured_issues(issues: Sequence[dict[str, Any]]) -> str:
    rendered = []
    for issue in issues:
        path = str(issue.get("path", "unknown"))
        message = str(issue.get("message", "Invalid transform payload"))
        rendered.append(f"{path}: {message}")
    return "; ".join(rendered)


def _normalize_text(raw: str | None) -> str:
    if raw is None:
        return ""
    return raw.strip()


def _exception_chain_text(exc: Exception) -> str:
    chain: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        message = str(current).strip() or current.__class__.__name__
        if message not in chain:
            chain.append(message)
        current = current.__cause__ or current.__context__
    return " | ".join(chain[:5])

__all__ = [
    "DatasetRepairAction",
    "DatasetRepairPlan",
    "TransformationResult",
    "transform",
]
