from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

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

ColumnRole: TypeAlias = Literal["covariate", "effect_modifier"]
PreferredType: TypeAlias = Literal[
    "NUMERIC",
    "CATEGORICAL",
    "BOOLEAN",
    "DATETIME",
    "OTHER",
]
DraftPreset: TypeAlias = Literal[
    "drop",
    "passthrough",
    "cat_onehot",
    "num_standard",
    "num_minmax",
    "num_log1p",
    "datetime_epoch_seconds",
]
ColumnProfile: TypeAlias = (
    NumericColumnProfileModel
    | DatetimeColumnProfileModel
    | BooleanColumnProfileModel
    | CategoricalColumnProfileModel
    | OtherColumnProfileModel
)

_ALLOWED_PRESETS_BY_KIND: dict[PreferredType, set[DraftPreset]] = {
    "NUMERIC": {"passthrough", "num_standard", "num_minmax", "num_log1p"},
    "CATEGORICAL": {"cat_onehot"},
    "BOOLEAN": {"passthrough", "cat_onehot"},
    "DATETIME": {"datetime_epoch_seconds"},
    "OTHER": {"drop"},
}


@dataclass(frozen=True)
class TransformationResult:
    transformation_plan: TransformPlan | None
    transformation_suggestions: ColumnTransformationSuggestionList | None


class ColumnTransformationSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    role: ColumnRole
    preferred_type: PreferredType
    preferred_type_reason: str = Field(..., min_length=1)


class ColumnTransformationSuggestionList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestions: list[ColumnTransformationSuggestion] = Field(..., min_length=1)


class _DraftColumn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    role: ColumnRole
    preset: DraftPreset
    preferred_type: PreferredType
    preferred_type_reason: str = Field(..., min_length=1)


class _BatchTransformDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[_DraftColumn] = Field(..., min_length=1)


def transform(
    *,
    transformation_instructions: str,
    causal_spec: CausalSpec,
    data_summary: DatasetSummaryModel,
    llm: LLMService,
    encoding_plan_tool: EncodingPlanTool,
) -> TransformationResult:
    expected_role_by_column = _protocol_scope_role_by_column(causal_spec)
    if not expected_role_by_column:
        return TransformationResult(
            transformation_plan=None,
            transformation_suggestions=None,
        )

    scoped_summary = _eligible_dataset_summary(
        data_summary,
        include_columns=list(expected_role_by_column.keys()),
    )
    
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
    expected_role_by_column: dict[str, ColumnRole],
    encoding_plan_tool: EncodingPlanTool,
) -> TransformationResult:
    retry_note: str | None = None

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
                        retry_note=retry_note,
                    ),
                    ensure_ascii=False,
                ),
                config=LLMConfig(
                    model="basic",
                    temperature=0.6,
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
            retry_note = _exception_chain_text(exc)

    raise ValueError(
        "batch transformation draft failed after retry: "
        f"{retry_note or 'unknown batch generation error'}"
    )


def _result_from_draft(
    *,
    draft_columns: Sequence[_DraftColumn],
    scoped_summary: DatasetSummaryModel,
    expected_role_by_column: dict[str, ColumnRole],
    encoding_plan_tool: EncodingPlanTool,
) -> TransformationResult:
    column_drafts_by_name = _validate_and_index_draft_columns(
        draft_columns=draft_columns,
        expected_role_by_column=expected_role_by_column,
    )
    profiles_by_name = _profiles_by_name(scoped_summary)
    _validate_draft_presets_by_profile(
        draft_columns=column_drafts_by_name,
        profiles_by_name=profiles_by_name,
    )

    payload = _materialize_transform_plan_payload(
        planned_columns=[
            column_drafts_by_name[column] for column in expected_role_by_column
        ],
        scoped_summary=scoped_summary,
    )
    covariate_columns = [
        column for column, role in expected_role_by_column.items() if role == "covariate"
    ]
    effect_modifier_columns = [
        column
        for column, role in expected_role_by_column.items()
        if role == "effect_modifier"
    ]
    model_dict, issues = encoding_plan_tool.validate_encoding_payload_structured(
        payload=payload,
        data_summary=scoped_summary,
        covariate_columns=covariate_columns,
        effect_modifier_columns=effect_modifier_columns,
    )
    if issues:
        raise ValueError(_format_structured_issues(issues))
    if model_dict is None:
        raise ValueError("encoding validation returned no plan and no issues")

    validated_plan = encoding_plan_tool.validate_encoding_payload(
        payload=payload,
        data_summary=scoped_summary,
        covariate_columns=covariate_columns,
        effect_modifier_columns=effect_modifier_columns,
    )
    return TransformationResult(
        transformation_plan=validated_plan,
        transformation_suggestions=_build_transformation_suggestions(
            planned_columns=[
                column_drafts_by_name[column] for column in expected_role_by_column
            ]
        ),
    )


def _validate_and_index_draft_columns(
    *,
    draft_columns: Sequence[_DraftColumn],
    expected_role_by_column: dict[str, ColumnRole],
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


def _profiles_by_name(summary: DatasetSummaryModel) -> dict[str, ColumnProfile]:
    return {str(profile.name).strip(): profile for profile in summary.profiles}


def _validate_draft_presets_by_profile(
    *,
    draft_columns: dict[str, _DraftColumn],
    profiles_by_name: dict[str, ColumnProfile],
) -> None:
    for column, draft in draft_columns.items():
        profile = profiles_by_name.get(column)
        if profile is None:
            raise ValueError(f"draft references unknown dataset summary column: {column}")
        current_kind = _profile_kind(profile)
        allowed_presets = _ALLOWED_PRESETS_BY_KIND[current_kind]
        if draft.preset not in allowed_presets:
            raise ValueError(
                f"column '{column}' has current kind '{current_kind}' so preset "
                f"'{draft.preset}' is not allowed; allowed presets: {sorted(allowed_presets)}"
            )


def _materialize_transform_plan_payload(
    *,
    planned_columns: Sequence[_DraftColumn],
    scoped_summary: DatasetSummaryModel,
) -> dict[str, Any]:
    profiles_by_name = _profiles_by_name(scoped_summary)
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
    draft_column: _DraftColumn,
    profile: ColumnProfile,
) -> dict[str, Any]:
    current_kind = _profile_kind(profile)
    allowed_presets = _ALLOWED_PRESETS_BY_KIND[current_kind]
    if draft_column.preset not in allowed_presets:
        raise ValueError(
            f"column '{draft_column.column}' has current kind '{current_kind}' so preset "
            f"'{draft_column.preset}' is not allowed"
        )

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
        case _:
            raise ValueError(
                f"unsupported preset '{draft_column.preset}' for column '{draft_column.column}'"
            )


def _build_transformation_suggestions(
    *,
    planned_columns: Sequence[_DraftColumn],
) -> ColumnTransformationSuggestionList:
    return ColumnTransformationSuggestionList(
        suggestions=[
            ColumnTransformationSuggestion(
                column=str(draft.column).strip(),
                role=draft.role,
                preferred_type=draft.preferred_type,
                preferred_type_reason=str(draft.preferred_type_reason).strip(),
            )
            for draft in planned_columns
        ]
    )


def _profile_kind(profile: ColumnProfile) -> PreferredType:
    return str(profile.inferred_kind)  # type: ignore[return-value]


def _protocol_scope_role_by_column(
    causal_spec: CausalSpec,
) -> dict[str, ColumnRole]:
    role_by_column: dict[str, ColumnRole] = {}
    identifier_column = str(causal_spec.id_col).strip()
    for column in causal_spec.covariates:
        normalized = str(column).strip()
        if normalized and normalized != identifier_column:
            role_by_column[normalized] = "covariate"
    for column in causal_spec.effect_modifiers:
        normalized = str(column).strip()
        if normalized and normalized != identifier_column:
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
    expected_role_by_column: dict[str, ColumnRole],
    retry_note: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transformation_instructions": _normalize_text(transformation_instructions),
        "compiled_causal_specification": causal_spec.model_dump(mode="json"),
        "scoped_dataset_summary": _dataset_summary_prompt_payload(scoped_summary),
        "eligible_columns": list(expected_role_by_column.keys()),
        "expected_role_by_column": expected_role_by_column,
        "required_plan_column_count": len(expected_role_by_column),
        "allowed_presets_by_kind": {
            kind: sorted(presets) for kind, presets in _ALLOWED_PRESETS_BY_KIND.items()
        },
    }
    if retry_note:
        payload["retry_note"] = retry_note
    return payload


def _dataset_summary_prompt_payload(summary: DatasetSummaryModel) -> dict[str, Any]:
    return {
        "n_rows": summary.n_rows,
        "columns": [_column_prompt_payload(profile) for profile in summary.profiles],
    }


def _column_prompt_payload(profile: ColumnProfile) -> dict[str, Any]:
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
        payload["top_values"] = [item.value for item in profile.summary.top_categories]
        return payload

    if isinstance(profile, OtherColumnProfileModel):
        payload["sample_values"] = list(profile.summary.distinct_values_sample)
        return payload

    return payload


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
    "ColumnTransformationSuggestion",
    "ColumnTransformationSuggestionList",
    "TransformationResult",
    "transform",
]
