from __future__ import annotations

import logging
from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.models.models import NonEmptyStr
from python.implementation.workflows.utils.utils import ScalarValue

class BinaryTreatmentSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["binary"]
    column: NonEmptyStr
    treated_values: List[ScalarValue] = Field(..., min_length=1)
    control_values: List[ScalarValue] = Field(..., min_length=1)


class CategoricalTreatmentSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["categorical"]
    column: NonEmptyStr
    levels: List[ScalarValue] = Field(..., min_length=2)
    baseline: Optional[ScalarValue] = None  # optional default


TreatmentSpecModel = Annotated[
    Union[BinaryTreatmentSpecModel, CategoricalTreatmentSpecModel],
    Field(discriminator="kind"),
]


class BinaryOutcomeSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["binary"]
    column: NonEmptyStr
    event_values: List[ScalarValue] = Field(..., min_length=1)
    non_event_values: List[ScalarValue] = Field(..., min_length=1)


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


class CausalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    Y: OutcomeSpecModel
    T: TreatmentSpecModel

    W: List[NonEmptyStr] = Field(default_factory=list) 
    X: List[NonEmptyStr] = Field(default_factory=list)
    Z: List[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_roles(self) -> "CausalSpec":
        y = self.Y.column
        t = self.T.column
        w = set(self.W)
        x = set(self.X)
        z = set(self.Z)

        # hard invariants
        if y == t:
            raise ValueError("Outcome column and treatment column must be different.")
        if y in w or y in x or y in z:
            raise ValueError("Outcome column must not appear in W/X/Z.")
        if t in w or t in x or t in z:
            raise ValueError("Treatment column must not appear in W/X/Z.")
        if w & x:
            logging.warning("Columns %s appear in both W and X. This is not recommended but we will keep them in W.", w & x)
        if (w | x | z) and (len(w | x | z) != len(list(w | x | z))):
            # defensive; sets already unique. keep if you later change representation.
            pass
        return self