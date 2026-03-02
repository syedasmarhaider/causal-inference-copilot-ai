from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing_extensions import Annotated

class NumericSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None  # {"0.05": 1.2, ...}


class DatetimeSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[str] = None  # isoformat-ish
    max: Optional[str] = None


class BooleanSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: Dict[str, int]  # keys are stringified values


class CategoryCountModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    count: int


class CategoricalSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_categories: List[CategoryCountModel]
    other_count: int


class OtherSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distinct_values_sample: List[str]


class ColumnProfileCommonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    dtype: Optional[str] = None
    n_rows: int
    n_missing: int
    missing_rate: float
    distinct_count: Optional[int] = None
    note: Optional[str] = None  # only used in non-strict mode fallbacks


class NumericColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["NUMERIC"]
    summary: NumericSummaryModel


class DatetimeColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["DATETIME"]
    summary: DatetimeSummaryModel


class BooleanColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["BOOLEAN"]
    summary: BooleanSummaryModel


class CategoricalColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["CATEGORICAL"]
    summary: CategoricalSummaryModel


class OtherColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["OTHER"]
    summary: OtherSummaryModel


ColumnProfileModel = Union[
    NumericColumnProfileModel,
    DatetimeColumnProfileModel,
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    OtherColumnProfileModel,
]


DiscriminatedColumnProfile = Annotated[ColumnProfileModel, Field(discriminator="inferred_kind")]

class DatasetSummaryModel(BaseModel):
    """
    Deterministic order: profiles follow df.columns order.
    """
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    profiles: List[ColumnProfileModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]