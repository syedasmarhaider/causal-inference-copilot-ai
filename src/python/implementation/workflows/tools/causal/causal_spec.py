from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from python.domain.models.models import NonEmptyStr

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
    Union[BinaryTreatmentSpecModel],
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


OutcomeSpecModel = Annotated[
    Union[BinaryOutcomeSpecModel, ContinuousOutcomeSpecModel],
    Field(discriminator="kind"),
]


# TODO: change name later to causal specs
class CausalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    treatment_spec: TreatmentSpecModel
    outcome_spec: OutcomeSpecModel
    covariates: List[NonEmptyStr]
    effect_modifiers: List[NonEmptyStr]
    experiment_type: ExperimentType

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
        model = CausalSpec.model_validate(payload)
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
