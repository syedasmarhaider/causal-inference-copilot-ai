from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from python.domain.models.models import NonEmptyStr
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    DatasetSummaryModel,
    DatetimeColumnProfileModel,
    NumericColumnProfileModel,
    OtherColumnProfileModel,
)

# ----------------------------
# Core types
# ----------------------------
ExperimentType = Literal["RCT", "OBSERVATIONAL"]


class BinaryTreatmentSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["binary"]
    column: NonEmptyStr
    treated: NonEmptyStr
    control: NonEmptyStr


TreatmentSpecModel = Annotated[
    BinaryTreatmentSpecModel,
    Field(discriminator="kind"),
]


class BinaryOutcomeSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["binary"]
    column: NonEmptyStr
    event: NonEmptyStr
    non_event: NonEmptyStr


class ContinuousOutcomeSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["continuous"]
    column: NonEmptyStr
    unit: NonEmptyStr | None = None
    clip_min: float | None = None
    clip_max: float | None = None


OutcomeSpecModel = Annotated[
    BinaryOutcomeSpecModel | ContinuousOutcomeSpecModel,
    Field(discriminator="kind"),
]


class CausalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    SUMMARY_FIELD_NAMES: ClassVar[tuple[str, ...] | None] = None
    SUMMARY_FIELD_KINDS: ClassVar[dict[str, str] | None] = None
    SUMMARY_KNOWN_VALUES: ClassVar[dict[str, set[str] | None] | None] = None

    treatment_spec: TreatmentSpecModel
    outcome_spec: OutcomeSpecModel
    covariates: list[NonEmptyStr]
    effect_modifiers: list[NonEmptyStr]
    experiment_type: ExperimentType

    @model_validator(mode="after")
    def _validate_against_summary(self) -> CausalSpec:
        summary_field_names = type(self).SUMMARY_FIELD_NAMES
        if summary_field_names is None:
            return self

        _validate_causal_spec_against_summary(
            spec=self,
            summary_field_names=summary_field_names,
            summary_field_kinds=type(self).SUMMARY_FIELD_KINDS or {},
            summary_known_values=type(self).SUMMARY_KNOWN_VALUES or {},
        )
        return self

    @classmethod
    def for_dataset_summary(cls, dataset_summary: DatasetSummaryModel) -> type[CausalSpec]:
        field_names = _extract_summary_field_names(dataset_summary)
        if not field_names:
            raise ValueError("dataset_summary must contain at least one non-empty column name")

        return type(
            f"{cls.__name__}ForFields_{len(field_names)}",
            (cls,),
            {
                "__module__": cls.__module__,
                "SUMMARY_FIELD_NAMES": field_names,
                "SUMMARY_FIELD_KINDS": _extract_summary_field_kinds(dataset_summary),
                "SUMMARY_KNOWN_VALUES": _extract_summary_known_values(dataset_summary),
            },
        )


# ----------------------------
# Validation helpers
# ----------------------------
def _fmt_loc(loc: Any) -> str:
    if isinstance(loc, (tuple, list)):
        return ".".join(
            str(x) for x in loc
        )  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return str(loc)


def validate_protocol_payload_structured(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    try:
        model = CausalSpec.model_validate(payload)
    except ValidationError as e:
        return None, _structured_validation_issues(e)

    return model.model_dump(mode="json"), []


def validate_protocol_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    model_dict, issues = validate_protocol_payload_structured(payload)
    if model_dict is None:
        return None, [f"{i.get('path')}: {i.get('message')}" for i in issues]
    return model_dict, []


def validate_backdoor_payload_structured(
    payload: Mapping[str, Any],
    *,
    dataset_summary: DatasetSummaryModel,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    try:
        model = CausalSpec.for_dataset_summary(dataset_summary).model_validate(payload)
    except ValidationError as e:
        return None, _structured_validation_issues(e)

    return model.model_dump(mode="json"), []


def validate_backdoor_payload(
    payload: Mapping[str, Any],
    *,
    dataset_summary: DatasetSummaryModel,
) -> tuple[dict[str, Any] | None, list[str]]:
    model_dict, issues = validate_backdoor_payload_structured(
        payload,
        dataset_summary=dataset_summary,
    )
    if model_dict is None:
        return None, [f"{i.get('path')}: {i.get('message')}" for i in issues]
    return model_dict, []


def _structured_validation_issues(exc: ValidationError) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for err in exc.errors():
        issues.append(
            {
                "path": _fmt_loc(err.get("loc")),
                "message": str(err.get("msg", "Invalid value")),
                "type": str(err.get("type", "unknown")),
                "input": err.get("input"),
            }
        )
    return issues


def _validate_causal_spec_against_summary(
    *,
    spec: CausalSpec,
    summary_field_names: tuple[str, ...],
    summary_field_kinds: dict[str, str],
    summary_known_values: dict[str, set[str] | None],
) -> None:
    treatment_col = str(spec.treatment_spec.column).strip()
    outcome_col = str(spec.outcome_spec.column).strip()
    covariates = [str(value).strip() for value in spec.covariates]
    effect_modifiers = [str(value).strip() for value in spec.effect_modifiers]
    referenced_columns = [treatment_col, outcome_col, *covariates, *effect_modifiers]

    missing = [column for column in referenced_columns if column not in set(summary_field_names)]
    if missing:
        raise ValueError(
            f"causal spec references unknown dataset_summary columns: {sorted(set(missing))}"
        )

    if treatment_col == outcome_col:
        raise ValueError("treatment and outcome must be different columns")

    duplicate_covariates = _find_duplicates(covariates)
    if duplicate_covariates:
        raise ValueError(f"covariates contain duplicates: {duplicate_covariates}")

    duplicate_effect_modifiers = _find_duplicates(effect_modifiers)
    if duplicate_effect_modifiers:
        raise ValueError(f"effect_modifiers contain duplicates: {duplicate_effect_modifiers}")

    overlap = sorted(set(covariates).intersection(effect_modifiers))
    if overlap:
        raise ValueError(f"covariates and effect_modifiers overlap: {overlap}")

    protected_overlap = sorted(
        {
            column
            for column in covariates + effect_modifiers
            if column in {treatment_col, outcome_col}
        }
    )
    if protected_overlap:
        raise ValueError(
            "covariates and effect_modifiers must not include treatment or outcome columns: "
            f"{protected_overlap}"
        )

    if spec.treatment_spec.treated == spec.treatment_spec.control:
        raise ValueError("treated and control must be different values")

    if isinstance(spec.outcome_spec, BinaryOutcomeSpecModel):
        if spec.outcome_spec.event == spec.outcome_spec.non_event:
            raise ValueError("event and non_event must be different values")

    treatment_kind = summary_field_kinds.get(treatment_col)
    if treatment_kind == "DATETIME":
        raise ValueError("binary treatment column cannot be DATETIME in dataset_summary")

    outcome_kind = summary_field_kinds.get(outcome_col)
    if isinstance(spec.outcome_spec, ContinuousOutcomeSpecModel):
        if outcome_kind != "NUMERIC":
            raise ValueError(
                f"continuous outcome requires NUMERIC dataset_summary column, got {outcome_kind}"
            )
        if (
            spec.outcome_spec.clip_min is not None
            and spec.outcome_spec.clip_max is not None
            and spec.outcome_spec.clip_min > spec.outcome_spec.clip_max
        ):
            raise ValueError("clip_min must be <= clip_max")
    elif outcome_kind == "DATETIME":
        raise ValueError("binary outcome column cannot be DATETIME in dataset_summary")

    treatment_values = summary_known_values.get(treatment_col)
    if treatment_values is not None:
        _validate_discrete_literals(
            column=treatment_col,
            expected_values={str(spec.treatment_spec.treated), str(spec.treatment_spec.control)},
            known_values=treatment_values,
            label="treatment",
        )

    if isinstance(spec.outcome_spec, BinaryOutcomeSpecModel):
        outcome_values = summary_known_values.get(outcome_col)
        if outcome_values is not None:
            _validate_discrete_literals(
                column=outcome_col,
                expected_values={str(spec.outcome_spec.event), str(spec.outcome_spec.non_event)},
                known_values=outcome_values,
                label="outcome",
            )


def _validate_discrete_literals(
    *,
    column: str,
    expected_values: set[str],
    known_values: set[str],
    label: str,
) -> None:
    missing = sorted(value for value in expected_values if value not in known_values)
    if missing:
        raise ValueError(
            f"{label} literals are not supported by dataset_summary for column '{column}': {missing}"
        )


def _find_duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen and normalized not in duplicates:
            duplicates.append(normalized)
            continue
        seen.add(normalized)
    return duplicates


def _extract_summary_field_names(dataset_summary: DatasetSummaryModel) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(profile.name).strip()
            for profile in dataset_summary.profiles
            if str(profile.name).strip()
        )
    )


def _extract_summary_field_kinds(dataset_summary: DatasetSummaryModel) -> dict[str, str]:
    return {
        str(profile.name).strip(): str(profile.inferred_kind)
        for profile in dataset_summary.profiles
        if str(profile.name).strip()
    }


def _extract_summary_known_values(
    dataset_summary: DatasetSummaryModel,
) -> dict[str, set[str] | None]:
    return {
        str(profile.name).strip(): _known_values_from_profile(profile)
        for profile in dataset_summary.profiles
        if str(profile.name).strip()
    }


def _known_values_from_profile(
    profile: (
        NumericColumnProfileModel
        | DatetimeColumnProfileModel
        | BooleanColumnProfileModel
        | CategoricalColumnProfileModel
        | OtherColumnProfileModel
    ),
) -> set[str] | None:
    if isinstance(profile, BooleanColumnProfileModel):
        return {str(value) for value in profile.summary.counts.keys()}

    if isinstance(profile, CategoricalColumnProfileModel):
        top_values = [str(item.value) for item in profile.summary.top_categories]
        if profile.distinct_count is not None and profile.distinct_count <= len(top_values):
            return set(top_values)
        return None

    if isinstance(profile, OtherColumnProfileModel):
        sampled_values = [str(value) for value in profile.summary.distinct_values_sample]
        if profile.distinct_count is not None and profile.distinct_count <= len(sampled_values):
            return set(sampled_values)
        return None

    return None


__all__ = [
    "BinaryOutcomeSpecModel",
    "BinaryTreatmentSpecModel",
    "CausalSpec",
    "ContinuousOutcomeSpecModel",
    "OutcomeSpecModel",
    "TreatmentSpecModel",
    "validate_backdoor_payload",
    "validate_backdoor_payload_structured",
    "validate_protocol_payload",
    "validate_protocol_payload_structured",
]
