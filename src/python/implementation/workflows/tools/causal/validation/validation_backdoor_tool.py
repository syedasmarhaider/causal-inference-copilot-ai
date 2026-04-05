from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pandas as pd
import pandas.api.types as ptypes
from pydantic import BaseModel, ConfigDict, Field

from python.domain.workflows.tool import Tool
from python.implementation.workflows.tools.causal.specs.causal_spec import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CausalSpec,
    ContinuousOutcomeSpecModel,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    CatOneHotParams,
    DateTimeEpochParams,
    MapBinaryParams,
    MapOrdinalParams,
    NumLog1pParams,
    TransformPlan,
)
from python.implementation.workflows.tools.causal.encoding.encoding_util import compile_plan_to_transformers
from python.domain.models.validation import ValidationIssueModel, ValidationStatus


class ValidationBackdoorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[ValidationIssueModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    metrics: dict[str, Any] = Field(default_factory=dict)

    @property
    def status(self) -> ValidationStatus:
        if any(issue.severity == "FAIL" for issue in self.issues):
            return "FAIL"
        if self.issues:
            return "WARN"
        return "PASS"


@dataclass(frozen=True)
class ValidationBackdoorTool(Tool):
    NAME: ClassVar[str] = "VALIDATION_BACKDOOR"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Tool for validating the suitability of a dataset and causal specification for causal effect estimation. "
            "Checks for common issues such as missing values, class imbalance, and consistency between the dataset and causal spec. "
            "Returns a report detailing any detected issues along with relevant metrics about the dataset and spec."
        )

    def validate(
        self,
        *,
        causal_spec: CausalSpec,
        dataframe: pd.DataFrame,
        transform_plan: TransformPlan | None,
    ) -> ValidationBackdoorReport:
        treatment_col = str(causal_spec.treatment_spec.column)
        outcome_col = str(causal_spec.outcome_spec.column)
        covariates = _dedup_keep_order(list(causal_spec.covariates))
        effect_modifiers = _dedup_keep_order(list(causal_spec.effect_modifiers))
        eligible_cols = _dedup_keep_order(covariates + effect_modifiers)

        metrics: dict[str, Any] = {
            "experiment_type": causal_spec.experiment_type,
            "n_rows": int(len(dataframe)),
            "n_columns": int(len(dataframe.columns)),
            "columns": [str(column) for column in dataframe.columns.tolist()],
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "covariates": covariates,
            "effect_modifiers": effect_modifiers,
            "eligible_plan_columns": eligible_cols,
            "plan_columns": [] if transform_plan is None else [column_plan.column for column_plan in transform_plan.columns],
        }

        issues: list[ValidationIssueModel] = []
        issues.extend(
            _guard_validation_step(
                step_name="dataframe structure",
                validator=lambda: _validate_dataframe_structure(
                    dataframe=dataframe,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    eligible_cols=eligible_cols,
                ),
            )
        )
        issues.extend(
            _guard_validation_step(
                step_name="causal spec",
                validator=lambda: _validate_causal_spec(
                    causal_spec=causal_spec,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    covariates=covariates,
                    effect_modifiers=effect_modifiers,
                ),
            )
        )

        if treatment_col in dataframe.columns:
            issues.extend(
                _guard_validation_step(
                    step_name="treatment column",
                    validator=lambda: _validate_treatment_column(
                        dataframe=dataframe,
                        treatment_spec=causal_spec.treatment_spec,
                        experiment_type=causal_spec.experiment_type,
                    ),
                )
            )

        if outcome_col in dataframe.columns:
            issues.extend(
                _guard_validation_step(
                    step_name="outcome column",
                    validator=lambda: _validate_outcome_column(
                        dataframe=dataframe,
                        treatment_spec=causal_spec.treatment_spec,
                        outcome_spec=causal_spec.outcome_spec,
                    ),
                )
            )

        issues.extend(
            _guard_validation_step(
                step_name="transform plan",
                validator=lambda: _validate_transform_plan(
                    dataframe=dataframe,
                    treatment_spec=causal_spec.treatment_spec,
                    transform_plan=transform_plan,
                    covariates=covariates,
                    effect_modifiers=effect_modifiers,
                    eligible_cols=eligible_cols,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                ),
            )
        )

        return ValidationBackdoorReport(issues=issues, metrics=metrics)


def validate_backdoor(
    *,
    causal_spec: CausalSpec,
    dataframe: pd.DataFrame,
    transform_plan: TransformPlan | None,
) -> ValidationBackdoorReport:
    return ValidationBackdoorTool().validate(
        causal_spec=causal_spec,
        dataframe=dataframe,
        transform_plan=transform_plan,
    )


def _guard_validation_step(
    *,
    step_name: str,
    validator: Callable[[], list[ValidationIssueModel]],
) -> list[ValidationIssueModel]:
    try:
        return validator()
    except Exception as exc:
        return [
            _issue(
                severity="FAIL",
                message=f"{step_name.capitalize()} validation failed unexpectedly.",
                evidence={"step": step_name, "error": repr(exc)},
                fix_hint="Inspect the validator logic or sanitize the incoming inputs before retrying.",
            )
        ]


def _validate_dataframe_structure(
    *,
    dataframe: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    eligible_cols: list[str],
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []

    if dataframe.empty:
        issues.append(
            _issue(
                severity="FAIL",
                message="Dataframe has no rows.",
                evidence={"n_rows": 0},
                fix_hint="Load a non-empty dataset before causal validation.",
            )
        )

    if len(dataframe) < 20:
        issues.append(
            _issue(
                severity="FAIL",
                message="Dataset has very few rows for causal estimation.",
                evidence={"n_rows": int(len(dataframe))},
                fix_hint="Expect unstable estimates unless more observations are available.",
            )
        )

    columns = [str(column) for column in dataframe.columns.tolist()]
    duplicate_columns = _find_duplicates(columns)
    if duplicate_columns:
        issues.append(
            _issue(
                severity="FAIL",
                message="Dataframe contains duplicate column names.",
                evidence={"duplicate_columns": duplicate_columns},
                fix_hint="Deduplicate column names before validation.",
            )
        )

    required_cols = _dedup_keep_order([treatment_col, outcome_col, *eligible_cols])
    missing_cols = [column for column in required_cols if column not in dataframe.columns]
    if missing_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Dataframe is missing columns referenced by the causal spec.",
                evidence={"missing_columns": missing_cols},
                fix_hint="Ensure the working dataset still contains every referenced treatment, outcome, covariate, and effect modifier column.",
            )
        )

    return issues


def _validate_causal_spec(
    *,
    causal_spec: CausalSpec,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    effect_modifiers: list[str],
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []

    if treatment_col == outcome_col:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment and outcome columns must be different.",
                evidence={"column": treatment_col},
                fix_hint="Choose distinct treatment and outcome columns in the causal spec.",
            )
        )

    cov_duplicates = _find_duplicates(list(causal_spec.covariates))
    if cov_duplicates:
        issues.append(
            _issue(
                severity="FAIL",
                message="Causal spec contains duplicate covariates.",
                evidence={"duplicate_covariates": cov_duplicates},
                fix_hint="Keep each covariate only once.",
            )
        )

    effect_duplicates = _find_duplicates(list(causal_spec.effect_modifiers))
    if effect_duplicates:
        issues.append(
            _issue(
                severity="FAIL",
                message="Causal spec contains duplicate effect modifiers.",
                evidence={"duplicate_effect_modifiers": effect_duplicates},
                fix_hint="Keep each effect modifier only once.",
            )
        )

    overlap = sorted(set(covariates).intersection(effect_modifiers))
    if overlap:
        issues.append(
            _issue(
                severity="FAIL",
                message="Covariates and effect modifiers overlap.",
                evidence={"overlap_columns": overlap},
                fix_hint="Assign each feature to exactly one role.",
            )
        )

    forbidden_overlap = sorted(
        {
            column
            for column in covariates + effect_modifiers
            if column in {treatment_col, outcome_col}
        }
    )
    if forbidden_overlap:
        issues.append(
            _issue(
                severity="FAIL",
                message="Covariates and effect modifiers must not include treatment or outcome columns.",
                evidence={"overlap_columns": forbidden_overlap},
                fix_hint="Remove treatment and outcome columns from adjustment features.",
            )
        )

    if causal_spec.experiment_type == "OBSERVATIONAL" and not covariates:
        issues.append(
            _issue(
                severity="FAIL",
                message="Observational studies require covariate for adjustment.",
                evidence={"experiment_type": causal_spec.experiment_type},
                fix_hint="Add observed confounders to causal_spec.covariates.",
            )
        )
        
    elif causal_spec.experiment_type == "RCT" and not covariates:
        issues.append(
            _issue(
                severity="WARN",
                message="RCT has no covariates; this is acceptable but limits precision gains.",
                evidence={"experiment_type": causal_spec.experiment_type},
                fix_hint="Add prognostic covariates only if you want adjusted estimates.",
            )
        )

    return issues


def _validate_treatment_column(
    *,
    dataframe: pd.DataFrame,
    treatment_spec: BinaryTreatmentSpecModel,
    experiment_type: Literal["RCT", "OBSERVATIONAL"],
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []
    treatment_col = str(treatment_spec.column)
    series = dataframe[treatment_col]

    missing_rate = float(series.isna().mean()) if len(series) else 0.0
    if missing_rate > 0.0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column contains missing values.",
                evidence={"treatment_col": treatment_col, "missing_rate": missing_rate},
                fix_hint="exclude rows with missing treatment values before estimation. Imputation can bias causal estimates and is not recommended for treatments.",
            )
        )
        return issues

    treated_key = _normalize_discrete_value(treatment_spec.treated)
    control_key = _normalize_discrete_value(treatment_spec.control)
    if treated_key == control_key:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment treated/control literals collapse to the same value.",
                evidence={"treated": treatment_spec.treated, "control": treatment_spec.control},
                fix_hint="Use distinct treated and control literals.",
            )
        )
        return issues

    observed = _normalized_value_counts(series)
    allowed_keys = {treated_key, control_key}
    unexpected = sorted(_discrete_key_text(key) for key in observed if key not in allowed_keys)
    if unexpected:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column contains values outside the declared treated/control literals.",
                evidence={"treatment_col": treatment_col, "unexpected_values": unexpected},
                fix_hint="Map the treatment column so only the declared treated and control values remain.",
            )
        )

    treated_count = int(observed.get(treated_key, 0))
    control_count = int(observed.get(control_key, 0))
    if treated_count == 0 or control_count == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Both treatment arms must be present in the dataframe.",
                evidence={
                    "treated_count": treated_count,
                    "control_count": control_count,
                    "treatment_col": treatment_col,
                },
                fix_hint="Check filtering and treatment mapping so both arms remain.",
            )
        )
        return issues

    min_arm_count = min(treated_count, control_count)
    min_arm_share = float(min_arm_count / max(treated_count + control_count, 1))

    if min_arm_count < 15:
        issues.append(
            _issue(
                severity="FAIL",
                message="One treatment arm has a low row count.",
                evidence={
                    "treated_count": treated_count,
                    "control_count": control_count,
                    "min_arm_count": min_arm_count,
                },
                fix_hint="Broaden the cohort or relax filters so both arms have enough support.",
            )
        )

    if min_arm_share < 0.20:
        issues.append(
            _issue(
                severity="FAIL",
                message=(
                    "Treatment-arm imbalance detected."
                    if experiment_type == "RCT"
                    else "Treatment-arm imbalance suggests a positivity risk."
                ),
                evidence={
                    "experiment_type": experiment_type,
                    "treated_count": treated_count,
                    "control_count": control_count,
                    "min_arm_share": min_arm_share,
                },
                fix_hint="Consider broader inclusion criteria or a different treatment definition.",
            )
        )

    return issues


def _validate_outcome_column(
    *,
    dataframe: pd.DataFrame,
    treatment_spec: BinaryTreatmentSpecModel,
    outcome_spec: BinaryOutcomeSpecModel | ContinuousOutcomeSpecModel,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []
    outcome_col = str(outcome_spec.column)
    treatment_col = str(treatment_spec.column)
    series = dataframe[outcome_col]

    missing_rate = float(series.isna().mean()) if len(series) else 0.0
    if missing_rate > 0.0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome column has missingness.",
                evidence={"outcome_col": outcome_col, "missing_rate": missing_rate},
                fix_hint="Redefine the cohort before estimation or impute outcome value in data set state.",
            )
        )

    non_missing = series.dropna()
    if non_missing.empty:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome column has no non-missing values.",
                evidence={"outcome_col": outcome_col},
                fix_hint="Load a dataset with observed outcome values.",
            )
        )
        return issues

    if isinstance(outcome_spec, BinaryOutcomeSpecModel):
        issues.extend(
            _validate_binary_outcome_column(
                dataframe=dataframe,
                treatment_col=treatment_col,
                treatment_spec=treatment_spec,
                outcome_col=outcome_col,
                outcome_spec=outcome_spec,
            )
        )
        return issues

    numeric = pd.to_numeric(non_missing, errors="coerce")
    failed_cast = int(numeric.isna().sum())
    if failed_cast > 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Continuous outcome column contains non-numeric values.",
                evidence={"outcome_col": outcome_col, "failed_numeric_cast_rows": failed_cast},
                fix_hint="Coerce the continuous outcome to numeric before causal estimation.",
            )
        )
        return issues

    unique_count = int(numeric.nunique(dropna=True))
    if unique_count < 2:
        issues.append(
            _issue(
                severity="FAIL",
                message="Continuous outcome has fewer than two unique numeric values.",
                evidence={"outcome_col": outcome_col, "unique_values": unique_count},
                fix_hint="Outcome variation is required for causal estimation.",
            )
        )
    elif unique_count < 5:
        issues.append(
            _issue(
                severity="WARN",
                message="Continuous outcome has very low numeric variation.",
                evidence={"outcome_col": outcome_col, "unique_values": unique_count},
                fix_hint="Expect unstable estimates if the outcome has little variation.",
            )
        )

    return issues


def _validate_binary_outcome_column(
    *,
    dataframe: pd.DataFrame,
    treatment_col: str,
    treatment_spec: BinaryTreatmentSpecModel,
    outcome_col: str,
    outcome_spec: BinaryOutcomeSpecModel,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []
    event_key = _normalize_discrete_value(outcome_spec.event)
    non_event_key = _normalize_discrete_value(outcome_spec.non_event)
    if event_key == non_event_key:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome event/non-event literals collapse to the same value.",
                evidence={"event": outcome_spec.event, "non_event": outcome_spec.non_event},
                fix_hint="Use distinct literals for the binary outcome.",
            )
        )
        return issues

    observed = _normalized_value_counts(dataframe[outcome_col].dropna())
    allowed_outcomes = {event_key, non_event_key}
    unexpected = sorted(_discrete_key_text(key) for key in observed if key not in allowed_outcomes)
    if unexpected:
        issues.append(
            _issue(
                severity="FAIL",
                message="Binary outcome column contains values outside event/non-event literals.",
                evidence={"outcome_col": outcome_col, "unexpected_values": unexpected},
                fix_hint="Map the outcome column to the declared event and non-event values only.",
            )
        )

    event_count = int(observed.get(event_key, 0))
    non_event_count = int(observed.get(non_event_key, 0))
    if event_count == 0 or non_event_count == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Binary outcome must contain both event and non-event observations.",
                evidence={"event_count": event_count, "non_event_count": non_event_count},
                fix_hint="Check outcome mapping or cohort filters so both classes are present.",
            )
        )
        return issues

    if event_count < 30:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome event count is low.",
                evidence={"outcome_col": outcome_col, "event_count": event_count},
                fix_hint="Increase cohort size or use a more common outcome definition if appropriate.",
            )
        )

    treated_key = _normalize_discrete_value(treatment_spec.treated)
    control_key = _normalize_discrete_value(treatment_spec.control)
    per_arm = dataframe[[treatment_col, outcome_col]].dropna()
    event_counts_by_arm: dict[str, int] = {"treated": 0, "control": 0}
    for treatment_value, outcome_value in per_arm.itertuples(index=False):
        treatment_key = _normalize_discrete_value(treatment_value)
        outcome_key = _normalize_discrete_value(outcome_value)
        if treatment_key == treated_key and outcome_key == event_key:
            event_counts_by_arm["treated"] += 1
        elif treatment_key == control_key and outcome_key == event_key:
            event_counts_by_arm["control"] += 1

    low_event_arms = {arm: count for arm, count in event_counts_by_arm.items() if count < 10}
    if low_event_arms:
        issues.append(
            _issue(
                severity="WARN",
                message="Some treatment arms have very few observed events.",
                evidence={"event_counts_by_arm": low_event_arms},
                fix_hint="Expect unstable effect estimates unless more outcome events are available.",
            )
        )

    return issues


def _validate_transform_plan(
    *,
    dataframe: pd.DataFrame,
    treatment_spec: BinaryTreatmentSpecModel,
    transform_plan: TransformPlan | None,
    covariates: list[str],
    effect_modifiers: list[str],
    eligible_cols: list[str],
    treatment_col: str,
    outcome_col: str,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []

    if not eligible_cols:
        if transform_plan is not None:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Transform plan was provided even though there are no covariates or effect modifiers to encode.",
                    evidence={"plan_columns": [column_plan.column for column_plan in transform_plan.columns]},
                    fix_hint="Drop the transform plan or add adjustment features if encoding is actually needed.",
                )
            )
        return issues

    if transform_plan is None:
        issues.append(
            _issue(
                severity="FAIL",
                message="Transform plan is required when covariates or effect modifiers are present.",
                evidence={"eligible_columns": eligible_cols},
                fix_hint="Generate a transform plan that covers every covariate and effect modifier.",
            )
        )
        return issues

    plan_columns = [column_plan.column for column_plan in transform_plan.columns]
    plan_set = set(plan_columns)
    eligible_set = set(eligible_cols)

    illegal_columns = sorted(plan_set.intersection({treatment_col, outcome_col}))
    if illegal_columns:
        issues.append(
            _issue(
                severity="FAIL",
                message="Transform plan must not include treatment or outcome columns.",
                evidence={"illegal_columns": illegal_columns},
                fix_hint="Keep only covariates and effect modifiers in the transform plan.",
            )
        )

    missing_columns = sorted(eligible_set - plan_set)
    if missing_columns:
        issues.append(
            _issue(
                severity="FAIL",
                message="Transform plan is missing eligible columns.",
                evidence={"missing_columns": missing_columns},
                fix_hint="Include every covariate and effect modifier in the transform plan.",
            )
        )

    extra_columns = sorted(plan_set - eligible_set)
    if extra_columns:
        issues.append(
            _issue(
                severity="FAIL",
                message="Transform plan contains non-eligible columns.",
                evidence={"extra_columns": extra_columns},
                fix_hint="Remove columns that are not declared as covariates or effect modifiers.",
            )
        )

    expected_roles = {
        **{column: "covariate" for column in covariates},
        **{column: "effect_modifier" for column in effect_modifiers},
    }
    for column_plan in transform_plan.columns:
        expected_role = expected_roles.get(column_plan.column)
        if expected_role is not None and column_plan.role != expected_role:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Transform plan assigns the wrong role to a column.",
                    evidence={
                        "column": column_plan.column,
                        "expected_role": expected_role,
                        "actual_role": column_plan.role,
                    },
                    fix_hint="Match transform-plan roles to the causal spec exactly.",
                )
            )

        if column_plan.column not in dataframe.columns:
            continue

        inferred_kind = _infer_kind_from_series(dataframe[column_plan.column])
        preset = str(column_plan.encoding.preset)
        if not _is_encoding_preset_compatible_with_kind(inferred_kind=inferred_kind, preset=preset):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Transform plan preset is incompatible with the observed dataframe column type.",
                    evidence={
                        "column": column_plan.column,
                        "inferred_kind": inferred_kind,
                        "preset": preset,
                    },
                    fix_hint="Choose a preset compatible with the observed column type.",
                )
            )
            continue

        issues.extend(
            _validate_encoding_semantics(
                dataframe=dataframe,
                treatment_spec=treatment_spec,
                column=column_plan.column,
                role=column_plan.role,
                inferred_kind=inferred_kind,
                encoding=column_plan.encoding,
            )
        )

    if any(issue.severity == "FAIL" for issue in issues):
        return issues

    try:
        compile_plan_to_transformers(
            plan=transform_plan,
            effect_modifiers=effect_modifiers,
            covariates=covariates,
            dense_output=True,
            require_full_coverage=True,
        )
    except Exception as exc:
        issues.append(
            _issue(
                severity="FAIL",
                message="Transform plan failed transformer compilation.",
                evidence={"error": repr(exc)},
                fix_hint="Adjust the transform plan so it compiles into sklearn transformers cleanly.",
            )
        )

    return issues


def _validate_encoding_semantics(
    *,
    dataframe: pd.DataFrame,
    treatment_spec: BinaryTreatmentSpecModel,
    column: str,
    role: Literal["covariate", "effect_modifier"],
    inferred_kind: str,
    encoding: Any,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []
    series = dataframe[column]
    non_missing = series.dropna()
    preset = str(encoding.preset)

    if preset == "drop":
        return issues

    if preset == "passthrough":
        issues.extend(
            _maybe_issue_for_unhandled_missingness(
                series=series,
                column=column,
                role=role,
                preset=preset,
            )
        )
        issues.extend(
            _maybe_issue_for_low_cardinality_numeric(
                series=series,
                column=column,
                role=role,
                preset=preset,
                inferred_kind=inferred_kind,
            )
        )
        return issues

    if preset in {"num_standard", "num_minmax"}:
        issues.extend(
            _maybe_issue_for_low_cardinality_numeric(
                series=series,
                column=column,
                role=role,
                preset=preset,
                inferred_kind=inferred_kind,
            )
        )
        return issues

    if isinstance(encoding, NumLog1pParams):
        numeric = pd.to_numeric(non_missing, errors="coerce")
        if numeric.isna().any():
            issues.append(
                _issue(
                    severity="FAIL",
                    message="num_log1p preset requires numeric values.",
                    evidence={"column": column},
                    fix_hint="Convert the column to numeric or choose a categorical preset.",
                )
            )
            return issues
        if (numeric <= -1).any():
            issues.append(
                _issue(
                    severity="FAIL",
                    message="num_log1p preset cannot be applied because some values are <= -1.",
                    evidence={"column": column},
                    fix_hint="Use another numeric preset or transform the raw values before log1p.",
                )
            )
        elif (numeric < 0).any() and not encoding.allow_negative:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="num_log1p preset received negative values while allow_negative is false.",
                    evidence={"column": column},
                    fix_hint="Enable allow_negative or choose a different numeric preset.",
                )
            )
        issues.extend(
            _maybe_issue_for_low_cardinality_numeric(
                series=series,
                column=column,
                role=role,
                preset=preset,
                inferred_kind=inferred_kind,
            )
        )
        return issues

    if isinstance(encoding, DateTimeEpochParams):
        issues.extend(
            _maybe_issue_for_unhandled_missingness(
                series=series,
                column=column,
                role=role,
                preset=preset,
            )
        )
        parsed = pd.to_datetime(non_missing, errors="coerce")
        parse_failures = int(parsed.isna().sum())
        if parse_failures > 0 and encoding.errors == "raise":
            issues.append(
                _issue(
                    severity="FAIL",
                    message="datetime_epoch_seconds preset cannot parse some datetime values.",
                    evidence={"column": column, "parse_failures": parse_failures},
                    fix_hint="Clean the datetime column or switch to coercion-aware preprocessing.",
                )
            )
        elif parse_failures > 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="datetime_epoch_seconds preset will coerce some invalid datetime values.",
                    evidence={"column": column, "parse_failures": parse_failures},
                    fix_hint="Inspect invalid datetime rows before fitting a model.",
                )
            )
        return issues

    if isinstance(encoding, CatOneHotParams):
        if encoding.missing == "error":
            issues.extend(
                _maybe_issue_for_unhandled_missingness(
                    series=series,
                    column=column,
                    role=role,
                    preset=preset,
                )
            )
        distinct = int(non_missing.astype(str).nunique())
        if encoding.max_categories is not None and distinct > int(encoding.max_categories):
            issues.append(
                _issue(
                    severity="WARN",
                    message="cat_onehot preset sees more categories than max_categories.",
                    evidence={
                        "column": column,
                        "distinct_categories": distinct,
                        "max_categories": int(encoding.max_categories),
                    },
                    fix_hint="Expect grouped or truncated categories, or increase max_categories.",
                )
            )
        issues.extend(
            _maybe_issue_for_single_arm_levels(
                dataframe=dataframe,
                treatment_spec=treatment_spec,
                column=column,
                role=role,
                preset=preset,
            )
        )
        return issues

    if isinstance(encoding, MapBinaryParams):
        if encoding.missing == "error":
            issues.extend(
                _maybe_issue_for_unhandled_missingness(
                    series=series,
                    column=column,
                    role=role,
                    preset=preset,
                )
            )
        issues.extend(
            _validate_mapping_encoding(
                column=column,
                series=series,
                allowed_keys=set(encoding.mapping.keys()),
                allow_unknown=encoding.allow_unknown,
                missing_mode=encoding.missing,
                missing_token=encoding.missing_token,
                preset_name="map_binary",
            )
        )
        issues.extend(
            _maybe_issue_for_single_arm_levels(
                dataframe=dataframe,
                treatment_spec=treatment_spec,
                column=column,
                role=role,
                preset=preset,
            )
        )
        return issues

    if isinstance(encoding, MapOrdinalParams):
        allowed_keys = set(encoding.order)
        if encoding.missing == "impute_token" and encoding.missing_token is not None:
            allowed_keys.add(encoding.missing_token)
        if encoding.missing == "error":
            issues.extend(
                _maybe_issue_for_unhandled_missingness(
                    series=series,
                    column=column,
                    role=role,
                    preset=preset,
                )
            )
        issues.extend(
            _validate_mapping_encoding(
                column=column,
                series=series,
                allowed_keys=allowed_keys,
                allow_unknown=encoding.allow_unknown,
                missing_mode=encoding.missing,
                missing_token=encoding.missing_token,
                preset_name="map_ordinal",
            )
        )
        issues.extend(
            _maybe_issue_for_single_arm_levels(
                dataframe=dataframe,
                treatment_spec=treatment_spec,
                column=column,
                role=role,
                preset=preset,
            )
        )
        return issues

    return issues


def _validate_mapping_encoding(
    *,
    column: str,
    series: pd.Series,
    allowed_keys: set[str],
    allow_unknown: bool,
    missing_mode: str,
    missing_token: str | None,
    preset_name: str,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []

    observed_keys = {str(value) for value in series.dropna().tolist()}
    if missing_mode == "impute_token" and missing_token is not None and missing_token not in allowed_keys:
        issues.append(
            _issue(
                severity="FAIL",
                message=f"{preset_name} missing_token is not covered by the declared mapping/order.",
                evidence={"column": column, "missing_token": missing_token},
                fix_hint="Include missing_token in the mapping/order or change missing handling.",
            )
        )

    unknown_keys = sorted(observed_keys - allowed_keys)
    if unknown_keys and not allow_unknown:
        issues.append(
            _issue(
                severity="FAIL",
                message=f"{preset_name} does not cover all observed non-missing values.",
                evidence={"column": column, "unknown_values": unknown_keys[:50]},
                fix_hint="Extend the mapping/order or enable unknown-value handling.",
            )
        )
    elif unknown_keys:
        issues.append(
            _issue(
                severity="WARN",
                message=f"{preset_name} will send some observed values through unknown handling.",
                evidence={"column": column, "unknown_values": unknown_keys[:50]},
                fix_hint="Review whether unknown categories should be mapped explicitly.",
            )
        )

    return issues


def _maybe_issue_for_unhandled_missingness(
    *,
    series: pd.Series,
    column: str,
    role: Literal["covariate", "effect_modifier"],
    preset: str,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []
    missing_count = int(series.isna().sum())
    if missing_count == 0:
        return issues

    missing_rate = float(series.isna().mean()) if len(series) else 0.0
    severity: Literal["WARN", "FAIL"] = "FAIL" if role == "effect_modifier" else "WARN"
    issues.append(
        _issue(
            severity=severity,
            message=(
                "Effect modifier has missing values but the transform preset does not explicitly handle them."
                if role == "effect_modifier"
                else "Covariate has missing values but the transform preset does not explicitly handle them. but its ok because later it will be handled by the estimator's internal imputation. Still, review the missingness and consider cleaning or encoding the column with explicit missing handling for more stable estimates."
            ),
            evidence={
                "column": column,
                "role": role,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "preset": preset,
            },
            fix_hint=(
                "Choose a preset with explicit missing handling or clean the effect modifier before estimation."
                if role == "effect_modifier"
                else "Choose a preset with explicit missing handling or clean the covariate before fitting."
            ),
        )
    )
    return issues


def _maybe_issue_for_single_arm_levels(
    *,
    dataframe: pd.DataFrame,
    treatment_spec: BinaryTreatmentSpecModel,
    column: str,
    role: Literal["covariate", "effect_modifier"],
    preset: str,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []

    treatment_col = str(treatment_spec.column)
    if treatment_col not in dataframe.columns or column not in dataframe.columns:
        return issues

    treated_key = _normalize_discrete_value(treatment_spec.treated)
    control_key = _normalize_discrete_value(treatment_spec.control)

    counts_by_level: dict[str, dict[str, int]] = {}
    for treatment_value, raw_value in dataframe[[treatment_col, column]].dropna().itertuples(index=False):
        treatment_key = _normalize_discrete_value(treatment_value)
        if treatment_key not in {treated_key, control_key}:
            continue
        level_key = _normalize_discrete_value(raw_value)
        level_text = _discrete_key_text(level_key)
        level_counts = counts_by_level.setdefault(level_text, {"treated": 0, "control": 0})
        if treatment_key == treated_key:
            level_counts["treated"] += 1
        elif treatment_key == control_key:
            level_counts["control"] += 1

    levels_missing_by_arm: list[dict[str, Any]] = []
    for level_text, level_counts in counts_by_level.items():
        missing_arms: list[str] = []
        if level_counts["treated"] == 0:
            missing_arms.append("treated")
        if level_counts["control"] == 0:
            missing_arms.append("control")
        if missing_arms:
            levels_missing_by_arm.append(
                {
                    "level": level_text,
                    "missing_arms": missing_arms,
                    "treated_count": level_counts["treated"],
                    "control_count": level_counts["control"],
                }
            )

    if levels_missing_by_arm:
        issues.append(
            _issue(
                severity="WARN",
                message="Categorical or mapped column has levels observed in only one treatment arm.",
                evidence={
                    "column": column,
                    "role": role,
                    "preset": preset,
                    "levels_missing_by_arm": levels_missing_by_arm,
                    "counts_by_level": counts_by_level,
                },
                fix_hint="Review sparse categories and consider regrouping rare levels before estimation.",
            )
        )

    return issues


def _maybe_issue_for_low_cardinality_numeric(
    *,
    series: pd.Series,
    column: str,
    role: Literal["covariate", "effect_modifier"],
    preset: str,
    inferred_kind: str,
) -> list[ValidationIssueModel]:
    issues: list[ValidationIssueModel] = []
    if inferred_kind != "NUMERIC" or preset not in {"num_standard", "num_minmax", "num_log1p", "passthrough"}:
        return issues

    non_missing = series.dropna()
    if non_missing.empty:
        return issues

    distinct_values = sorted({float(value) for value in pd.to_numeric(non_missing, errors="coerce").dropna().tolist()})
    distinct_count = len(distinct_values)
    if distinct_count > 20:
        return issues

    issues.append(
        _issue(
            severity="WARN",
            message="Numeric column has low cardinality and may actually represent coded categories.",
            evidence={
                "column": column,
                "role": role,
                "preset": preset,
                "distinct_non_null_count": distinct_count,
                "distinct_values_sample": distinct_values[:20],
            },
            fix_hint="Review whether this numeric column should use a categorical encoding strategy instead.",
        )
    )
    return issues


def _is_encoding_preset_compatible_with_kind(*, inferred_kind: str, preset: str) -> bool:
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


def _infer_kind_from_series(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series):
        return "BOOLEAN"
    if ptypes.is_numeric_dtype(series):
        return "NUMERIC"
    if ptypes.is_datetime64_any_dtype(series):
        return "DATETIME"
    if ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series) or ptypes.is_categorical_dtype(series):
        return "CATEGORICAL"
    return "OTHER"


def _normalize_discrete_value(value: Any) -> tuple[str, Any]:
    try:
        if pd.isna(value):
            return ("na", None)
    except Exception:
        pass

    if isinstance(value, bool):
        return ("bool", bool(value))
    if isinstance(value, numbers.Real):
        return ("num", float(value))
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered == "true":
            return ("bool", True)
        if lowered == "false":
            return ("bool", False)
        try:
            return ("num", float(lowered))
        except ValueError:
            return ("str", lowered)
    return ("str", str(value).strip().lower())


def _normalized_value_counts(series: pd.Series) -> dict[tuple[str, Any], int]:
    counts: dict[tuple[str, Any], int] = {}
    for value in series.tolist():
        key = _normalize_discrete_value(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _discrete_key_text(key: tuple[str, Any]) -> str:
    return f"{key[0]}:{key[1]!r}"


def _dedup_keep_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _issue(
    *,
    severity: Literal["WARN", "FAIL"],
    message: str,
    evidence: dict[str, Any] | None = None,
    fix_hint: str | None = None,
) -> ValidationIssueModel:
    return ValidationIssueModel(
        severity=severity,
        message=message,
        evidence=dict(evidence or {}),
        fix_hint=fix_hint,
    )


__all__ = ["ValidationBackdoorReport", "ValidationBackdoorTool", "validate_backdoor"]
