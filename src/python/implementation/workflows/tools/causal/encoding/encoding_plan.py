from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from python.implementation.workflows.utils.validation import NonEmptyStr
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    DatasetSummaryModel,
    DatetimeColumnProfileModel,
    NumericColumnProfileModel,
    OtherColumnProfileModel,
)

PosInt = Annotated[int, Field(ge=1)]
EncodingRole = Literal["covariate", "effect_modifier"]

EncodingPreset = Literal[
    # structural
    "drop",
    "passthrough",
    # categorical
    "cat_onehot",
    # numeric
    "num_standard",
    "num_minmax",
    "num_log1p",
    # datetime
    "datetime_epoch_seconds",
    # explicit mapping
    "map_binary",
    "map_ordinal",
]
     


# ----------------------------
# Params (small + optional)
# ----------------------------
class _BaseParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DropParams(_BaseParams):
    preset: Literal["drop"]


class PassthroughParams(_BaseParams):
    preset: Literal["passthrough"]


class CatOneHotParams(_BaseParams):
    preset: Literal["cat_onehot"]
    drop_first: bool = False
    handle_unknown: Literal["ignore", "error"] = "ignore"
    # truly optional (None means "no cap")
    max_categories: PosInt | None = None
    missing: Literal["impute_token", "dummy_na", "error"] = "impute_token"
    missing_token: NonEmptyStr = "__MISSING__"

    @model_validator(mode="after")
    def _validate_cat_onehot(self) -> CatOneHotParams:
        if self.missing in ("dummy_na", "error"):
            # token is ignored in these modes, but keep it non-empty anyway
            return self
        # missing == "impute_token"
        if not self.missing_token:
            raise ValueError("cat_onehot: missing_token must be non-empty when missing='impute_token'.")
        return self


class NumParams(_BaseParams):
    impute: Literal["median", "mean"] = "median"
    add_missing_indicator: bool = True


class NumStandardParams(NumParams):
    preset: Literal["num_standard"]


class NumMinMaxParams(NumParams):
    preset: Literal["num_minmax"]
    eps: float = 1e-12
    
    @model_validator(mode="after")
    def _validate_minmax(self) -> NumMinMaxParams:
        if not (self.eps > 0.0):
            raise ValueError("num_minmax: eps must be > 0.")
        return self


class NumLog1pParams(NumParams):
    preset: Literal["num_log1p"]
    allow_negative: bool = False
    then_scale: Literal["none", "standard", "minmax"] = "none"


class DateTimeEpochParams(_BaseParams):
    preset: Literal["datetime_epoch_seconds"]
    errors: Literal["coerce", "raise"] = "coerce"
    unit: Literal["s", "ms", "us", "ns"] = "s"
    add_missing_indicator: bool = True


class MapBinaryParams(_BaseParams):
    preset: Literal["map_binary"]
    mapping: dict[NonEmptyStr, float] = Field(..., min_length=1)

    allow_unknown: bool = True
    unknown_value: float | None = None

    missing: Literal["as_unknown", "impute_token", "error"] = "as_unknown"
    missing_token: NonEmptyStr | None = None  # required if missing="impute_token"

    @model_validator(mode="after")
    def _validate_map_binary(self) -> MapBinaryParams:
        if self.missing == "impute_token" and self.missing_token is None:
            raise ValueError("map_binary: missing_token required when missing='impute_token'.")
        # Strong safety: avoid NaNs escaping unless explicitly configured
        if self.allow_unknown and self.unknown_value is None:
            raise ValueError("map_binary: unknown_value required when allow_unknown=True (avoid NaNs).")
        if self.missing == "as_unknown" and self.unknown_value is None:
            raise ValueError("map_binary: unknown_value required when missing='as_unknown' (avoid NaNs).")
        return self


class MapOrdinalParams(_BaseParams):
    preset: Literal["map_ordinal"]
    order: list[NonEmptyStr] = Field(..., min_length=1)
    start: int = 0

    allow_unknown: bool = True
    unknown_value: int | None = None

    missing: Literal["as_unknown", "impute_token", "error"] = "as_unknown"
    missing_token: NonEmptyStr | None = None
    token_position: Literal["prepend", "append"] | None = None  # required if missing="impute_token"

    @model_validator(mode="after")
    def _validate_map_ordinal(self) -> MapOrdinalParams:
        if len(self.order) != len(set(self.order)):
            raise ValueError("map_ordinal: 'order' must not contain duplicates.")
        if self.missing == "impute_token" and (
            self.missing_token is None or self.token_position is None
        ):
            raise ValueError(
                "map_ordinal: missing_token and token_position required when missing='impute_token'."
            )
        # Strong safety: avoid NaNs escaping unless explicitly configured
        if self.allow_unknown and self.unknown_value is None:
            raise ValueError("map_ordinal: unknown_value required when allow_unknown=True (avoid NaNs).")
        if self.missing == "as_unknown" and self.unknown_value is None:
            raise ValueError("map_ordinal: unknown_value required when missing='as_unknown' (avoid NaNs).")
        return self


EncodingPresetSpec = Annotated[
    DropParams | PassthroughParams | CatOneHotParams | NumStandardParams | NumMinMaxParams | NumLog1pParams | DateTimeEpochParams | MapBinaryParams | MapOrdinalParams,
    Field(discriminator="preset"),
]


# ----------------------------
# Plan models
# ----------------------------
class ColumnEncodingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    column: NonEmptyStr
    role: EncodingRole
    encoding: EncodingPresetSpec


class TransformPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    SUMMARY_FIELD_NAMES: ClassVar[tuple[str, ...] | None] = None
    SUMMARY_FIELD_KINDS: ClassVar[dict[str, str] | None] = None
    SUMMARY_KNOWN_VALUES: ClassVar[dict[str, set[str] | None] | None] = None
    ELIGIBLE_COLUMNS: ClassVar[tuple[str, ...] | None] = None
    EXPECTED_ROLE_BY_COLUMN: ClassVar[dict[str, EncodingRole] | None] = None

    columns: list[ColumnEncodingPlan] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_plan(self) -> TransformPlan:
        cols = [c.column for c in self.columns]
        if len(cols) != len(set(cols)):
            dup = sorted({c for c in cols if cols.count(c) > 1})
            raise ValueError(f"TransformPlan: duplicate column entries are not allowed: {dup}")

        has_covariate = any(c.role == "covariate" for c in self.columns)
        has_effect_modifier = any(c.role == "effect_modifier" for c in self.columns)
        if not has_covariate and not has_effect_modifier:
            raise ValueError("TransformPlan: must contain at least one covariate or effect_modifier column.")

        summary_field_names = type(self).SUMMARY_FIELD_NAMES
        if summary_field_names is not None:
            _validate_transform_plan_against_constraints(
                plan=self,
                summary_field_names=summary_field_names,
                summary_field_kinds=type(self).SUMMARY_FIELD_KINDS or {},
                summary_known_values=type(self).SUMMARY_KNOWN_VALUES or {},
                eligible_columns=type(self).ELIGIBLE_COLUMNS,
                expected_role_by_column=type(self).EXPECTED_ROLE_BY_COLUMN,
            )

        return self

    @classmethod
    def for_dataset_summary(
        cls,
        dataset_summary: DatasetSummaryModel,
        *,
        covariate_columns: Sequence[str] | None = None,
        effect_modifier_columns: Sequence[str] | None = None,
    ) -> type[TransformPlan]:
        field_names = _extract_summary_field_names(dataset_summary)
        if not field_names:
            raise ValueError("dataset_summary must contain at least one non-empty column name")

        expected_role_by_column = _build_expected_role_by_column(
            covariate_columns=covariate_columns,
            effect_modifier_columns=effect_modifier_columns,
        )
        if expected_role_by_column is None:
            raise ValueError("At least one covariate or effect_modifier column is required")

        unknown_constraint_columns = sorted(set(expected_role_by_column) - set(field_names))
        if unknown_constraint_columns:
            raise ValueError(
                "covariate/effect_modifier columns are not present in dataset_summary: "
                f"{unknown_constraint_columns}"
            )

        return type(
            f"{cls.__name__}ForFields_{len(field_names)}",
            (cls,),
            {
                "__module__": cls.__module__,
                "SUMMARY_FIELD_NAMES": field_names,
                "SUMMARY_FIELD_KINDS": _extract_summary_field_kinds(dataset_summary),
                "SUMMARY_KNOWN_VALUES": _extract_summary_known_values(dataset_summary),
                "ELIGIBLE_COLUMNS": tuple(expected_role_by_column.keys()),
                "EXPECTED_ROLE_BY_COLUMN": expected_role_by_column,
            },
        )


def validate_transform_payload_structured(
    payload: Mapping[str, Any],
    *,
    dataset_summary: DatasetSummaryModel,
    covariate_columns: Sequence[str] | None = None,
    effect_modifier_columns: Sequence[str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    schema = TransformPlan.for_dataset_summary(
        dataset_summary,
        covariate_columns=covariate_columns,
        effect_modifier_columns=effect_modifier_columns,
    )
    try:
        model = schema.model_validate(payload)
    except ValidationError as exc:
        return None, _structured_validation_issues(exc)

    return model.model_dump(mode="json"), []


def validate_transform_payload(
    payload: Mapping[str, Any],
    *,
    dataset_summary: DatasetSummaryModel,
    covariate_columns: Sequence[str] | None = None,
    effect_modifier_columns: Sequence[str] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    model_dict, issues = validate_transform_payload_structured(
        payload,
        dataset_summary=dataset_summary,
        covariate_columns=covariate_columns,
        effect_modifier_columns=effect_modifier_columns,
    )
    if model_dict is None:
        return None, [f"{i.get('path')}: {i.get('message')}" for i in issues]
    return model_dict, []


def _validate_transform_plan_against_constraints(
    *,
    plan: TransformPlan,
    summary_field_names: tuple[str, ...],
    summary_field_kinds: dict[str, str],
    summary_known_values: dict[str, set[str] | None],
    eligible_columns: tuple[str, ...] | None,
    expected_role_by_column: dict[str, EncodingRole] | None,
) -> None:
    plan_columns = [str(column_plan.column).strip() for column_plan in plan.columns]
    role_by_column = {
        str(column_plan.column).strip(): str(column_plan.role)
        for column_plan in plan.columns
    }

    summary_field_name_set = set(summary_field_names)
    unknown_columns = sorted(set(column for column in plan_columns if column not in summary_field_name_set))
    if unknown_columns:
        raise ValueError(
            f"encoding plan references unknown dataset_summary columns: {unknown_columns}"
        )

    if eligible_columns is not None:
        eligible_set = set(eligible_columns)
        extra_columns = sorted(set(plan_columns) - eligible_set)
        if extra_columns:
            raise ValueError(f"encoding plan contains non-eligible columns: {extra_columns}")

        missing_columns = sorted(eligible_set - set(plan_columns))
        if missing_columns:
            raise ValueError(f"encoding plan is missing eligible columns: {missing_columns}")

    if expected_role_by_column:
        wrong_roles: list[dict[str, str | None]] = []
        for column, expected_role in expected_role_by_column.items():
            actual_role = role_by_column.get(column)
            if actual_role != expected_role:
                wrong_roles.append(
                    {
                        "column": column,
                        "expected_role": expected_role,
                        "actual_role": actual_role,
                    }
                )
        if wrong_roles:
            raise ValueError(f"encoding plan assigned wrong roles: {wrong_roles}")

    incompatible_presets: list[dict[str, str]] = []
    for column_plan in plan.columns:
        column = str(column_plan.column).strip()
        inferred_kind = summary_field_kinds.get(column)
        preset = str(column_plan.encoding.preset)
        if inferred_kind is None:
            continue
        if not _is_encoding_preset_compatible_with_kind(
            inferred_kind=inferred_kind,
            preset=preset,
        ):
            incompatible_presets.append(
                {
                    "column": column,
                    "inferred_kind": inferred_kind,
                    "preset": preset,
                }
            )

        known_values = summary_known_values.get(column)
        if known_values is None:
            continue
        _validate_mapping_values_against_known_values(
            column=column,
            encoding=column_plan.encoding,
            known_values=known_values,
        )

    if incompatible_presets:
        raise ValueError(
            "encoding plan has column type and preset incompatibilities: "
            f"{incompatible_presets}"
        )


def _validate_mapping_values_against_known_values(
    *,
    column: str,
    encoding: EncodingPresetSpec,
    known_values: set[str],
) -> None:
    if isinstance(encoding, MapBinaryParams):
        mapping_values = {str(value) for value in encoding.mapping.keys()}
        if encoding.missing == "impute_token" and encoding.missing_token is not None:
            mapping_values.discard(str(encoding.missing_token))
        unsupported = sorted(value for value in mapping_values if value not in known_values)
        if unsupported:
            raise ValueError(
                "map_binary mapping contains values not supported by dataset_summary "
                f"for column '{column}': {unsupported}"
            )

    if isinstance(encoding, MapOrdinalParams):
        order_values = {str(value) for value in encoding.order}
        if encoding.missing == "impute_token" and encoding.missing_token is not None:
            order_values.discard(str(encoding.missing_token))
        unsupported = sorted(value for value in order_values if value not in known_values)
        if unsupported:
            raise ValueError(
                "map_ordinal order contains values not supported by dataset_summary "
                f"for column '{column}': {unsupported}"
            )


def _is_encoding_preset_compatible_with_kind(
    *,
    inferred_kind: str,
    preset: str,
) -> bool:
    if preset in {"drop", "passthrough"}:
        return True
    if inferred_kind == "NUMERIC":
        return preset in {"num_standard", "num_minmax", "num_log1p"}
    if inferred_kind == "CATEGORICAL":
        return preset in {"cat_onehot", "map_binary", "map_ordinal"}
    if inferred_kind == "BOOLEAN":
        return preset in {
            "cat_onehot",
            "map_binary",
            "map_ordinal",
            "num_standard",
            "num_minmax",
            "num_log1p",
        }
    if inferred_kind == "DATETIME":
        return preset == "datetime_epoch_seconds"
    return False


def _fmt_loc(loc: Any) -> str:
    if isinstance(loc, (tuple, list)):
        return ".".join(str(item) for item in loc)
    return str(loc)


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


def _normalize_column_sequence(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    return tuple(
        dict.fromkeys(
            str(column).strip()
            for column in columns
            if str(column).strip()
        )
    )


def _build_expected_role_by_column(
    *,
    covariate_columns: Sequence[str] | None,
    effect_modifier_columns: Sequence[str] | None,
) -> dict[str, EncodingRole] | None:
    covariate_columns = _normalize_column_sequence(covariate_columns) or ()
    effect_modifier_columns = _normalize_column_sequence(effect_modifier_columns) or ()
    overlap = sorted(set(covariate_columns).intersection(effect_modifier_columns))
    if overlap:
        raise ValueError(
            "covariate_columns and effect_modifier_columns overlap: "
            f"{overlap}"
        )

    expected_role_by_column: dict[str, EncodingRole] = {
        column: "covariate" for column in covariate_columns
    }
    expected_role_by_column.update(
        {column: "effect_modifier" for column in effect_modifier_columns}
    )
    return expected_role_by_column or None


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
    profile: NumericColumnProfileModel
    | DatetimeColumnProfileModel
    | BooleanColumnProfileModel
    | CategoricalColumnProfileModel
    | OtherColumnProfileModel,
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
    "CatOneHotParams",
    "ColumnEncodingPlan",
    "DateTimeEpochParams",
    "DropParams",
    "EncodingPreset",
    "EncodingPresetSpec",
    "EncodingRole",
    "MapBinaryParams",
    "MapOrdinalParams",
    "NumLog1pParams",
    "NumMinMaxParams",
    "NumStandardParams",
    "PassthroughParams",
    "TransformPlan",
    "validate_transform_payload",
    "validate_transform_payload_structured",
]
