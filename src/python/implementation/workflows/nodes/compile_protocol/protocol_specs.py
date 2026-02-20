from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from python.implementation.workflows.utils.validation import NonEmptyStr

# ----------------------------
# Core types
# ----------------------------
TimeZeroType = Literal["COLUMN", "CONCEPTUAL"]
WindowUnit = Literal["minutes", "hours", "days", "weeks", "months", "years"]
FilterOp = Literal["==", "in", "not_in", ">=", "<=", ">", "<"]


# ----------------------------
# Models
# ----------------------------
class ExclusionRuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    column: NonEmptyStr
    op: FilterOp
    values: List[NonEmptyStr]  # allow empty list for is_null/not_null; items must be non-empty if present


class BinaryTreatmentSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["binary"]
    column: NonEmptyStr
    treated: NonEmptyStr
    control: NonEmptyStr

class ContinuousTreatmentSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["continuous"]
    column: NonEmptyStr
    unit: Optional[NonEmptyStr] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


class CategoricalTreatmentSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["categorical"]
    column: NonEmptyStr
    levels: List[NonEmptyStr] = Field(..., min_length=2)


TreatmentSpecModel = Annotated[
    Union[BinaryTreatmentSpecModel, ContinuousTreatmentSpecModel, CategoricalTreatmentSpecModel],
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
    unit: Optional[NonEmptyStr] = None
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


class CategoricalOutcomeSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["categorical"]
    column: NonEmptyStr
    levels: List[NonEmptyStr] = Field(..., min_length=2)


class DurationOutcomeSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["duration"]
    duration_column: NonEmptyStr
    event_column: NonEmptyStr
    event_value: NonEmptyStr
    censor_value: NonEmptyStr


OutcomeSpecModel = Annotated[
    Union[BinaryOutcomeSpecModel, ContinuousOutcomeSpecModel, CategoricalOutcomeSpecModel, DurationOutcomeSpecModel],
    Field(discriminator="kind"),
]


class ProtocolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    
    exclusions: List[ExclusionRuleModel]

    time_zero_type: TimeZeroType
    time_zero: NonEmptyStr
    time_zero_definition: NonEmptyStr

    treatment_spec: TreatmentSpecModel
    treatment_window_start: NonEmptyStr
    treatment_window_end: NonEmptyStr
    treatment_window_unit: WindowUnit

    outcome_spec: OutcomeSpecModel
    outcome_window: NonEmptyStr
    outcome_window_unit: WindowUnit

    covariates: List[NonEmptyStr]
    effect_modifiers: List[NonEmptyStr]
    experiment_type: NonEmptyStr

# ----------------------------
# Validation helpers
# ----------------------------
def _fmt_loc(loc: Any) -> str:
    if isinstance(loc, (tuple, list)):
        return ".".join(str(x) for x in loc) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return str(loc)


def validate_protocol_payload_structured(
    payload: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        model = ProtocolSpec.model_validate(payload)
    except ValidationError as e:
        issues: List[Dict[str, Any]] = []
        for err in e.errors():
            issues.append(
                {
                    "path": _fmt_loc(err.get("loc")),
                    "message": str(err.get("msg", "Invalid value")),
                    "type": str(err.get("type", "unknown")),
                    "input": err.get("input"),
                }
            )
        return None, issues

    return model.model_dump(mode="json"), []


def validate_protocol_payload(
    payload: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    model_dict, issues = validate_protocol_payload_structured(payload)
    if model_dict is None:
        msgs = [f"{i.get('path')}: {i.get('message')}" for i in issues]
        return None, msgs
    return model_dict, []
