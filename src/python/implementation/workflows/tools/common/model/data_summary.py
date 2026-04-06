from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class NumericSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: dict[str, float] | None = None  # {"0.05": 1.2, ...}


class DatetimeSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: str | None = None  # isoformat-ish
    max: str | None = None


class BooleanSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int]  # keys are stringified values


class CategoryCountModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    count: int


class CategoricalSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_categories: list[CategoryCountModel]
    other_count: int


class OtherSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distinct_values_sample: list[str]


class ColumnProfileCommonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    dtype: str | None = None
    n_rows: int
    n_missing: int
    missing_rate: float
    distinct_count: int | None = None
    note: str | None = None  # only used in non-strict mode fallbacks


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


ColumnProfileModel = (
    NumericColumnProfileModel
    | DatetimeColumnProfileModel
    | BooleanColumnProfileModel
    | CategoricalColumnProfileModel
    | OtherColumnProfileModel
)


DiscriminatedColumnProfile = Annotated[ColumnProfileModel, Field(discriminator="inferred_kind")]


class DatasetSummaryModel(BaseModel):
    """
    Deterministic order: profiles follow df.columns order.
    """

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    profiles: list[ColumnProfileModel] = Field(
        default_factory=list
    )  # pyright: ignore[reportUnknownVariableType]
