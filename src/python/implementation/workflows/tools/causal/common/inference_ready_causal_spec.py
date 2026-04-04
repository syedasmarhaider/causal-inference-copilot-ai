from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    ColumnProfileModel,
    DatasetSummaryModel,
)


class InferenceReadyCausalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    data_summary: DatasetSummaryModel

    @model_validator(mode="after")
    def _validate_consistency(self) -> InferenceReadyCausalSpec:
        covariates = [str(column).strip() for column in self.causal_spec.covariates]
        effect_modifiers = [
            str(column).strip() for column in self.causal_spec.effect_modifiers
        ]
        expected_columns = set(covariates).union(effect_modifiers)

        if not expected_columns:
            raise ValueError(
                "InferenceReadyCausalSpec requires at least one covariate or effect_modifier."
            )

        treatment_column = str(self.causal_spec.treatment_spec.column).strip()
        outcome_column = str(self.causal_spec.outcome_spec.column).strip()
        protected_columns = {treatment_column, outcome_column}
        profile_by_name = self._profile_by_name()
        missing_summary_columns = sorted(
            expected_columns.union(protected_columns) - set(profile_by_name.keys())
        )
        if missing_summary_columns:
            raise ValueError(
                "data_summary is missing causal_spec/transformation_plan columns: "
                f"{missing_summary_columns}"
            )

        plan_columns = [
            str(column_plan.column).strip()
            for column_plan in self.transformation_plan.columns
        ]
        plan_roles = {
            str(column_plan.column).strip(): str(column_plan.role)
            for column_plan in self.transformation_plan.columns
        }

        protected_in_plan = sorted(expected_columns.intersection(protected_columns))
        if protected_in_plan:
            raise ValueError(
                "causal_spec covariates/effect_modifiers must not include treatment or outcome "
                f"columns: {protected_in_plan}"
            )

        invalid_protected_plan_columns = sorted(set(plan_columns).intersection(protected_columns))
        if invalid_protected_plan_columns:
            raise ValueError(
                "transformation_plan must not include treatment or outcome columns: "
                f"{invalid_protected_plan_columns}"
            )

        extra_columns = sorted(set(plan_columns) - expected_columns)
        if extra_columns:
            raise ValueError(
                "transformation_plan contains columns outside causal_spec covariates/"
                f"effect_modifiers: {extra_columns}"
            )

        missing_covariates = sorted(set(covariates) - set(plan_columns))
        if missing_covariates:
            raise ValueError(
                "transformation_plan is missing causal_spec covariates: "
                f"{missing_covariates}"
            )

        missing_effect_modifiers = sorted(set(effect_modifiers) - set(plan_columns))
        if missing_effect_modifiers:
            raise ValueError(
                "transformation_plan is missing causal_spec effect_modifiers: "
                f"{missing_effect_modifiers}"
            )

        wrong_roles: list[dict[str, str]] = []
        for column in covariates:
            actual_role = plan_roles.get(column)
            if actual_role != "covariate":
                wrong_roles.append(
                    {
                        "column": column,
                        "expected_role": "covariate",
                        "actual_role": str(actual_role),
                    }
                )
        for column in effect_modifiers:
            actual_role = plan_roles.get(column)
            if actual_role != "effect_modifier":
                wrong_roles.append(
                    {
                        "column": column,
                        "expected_role": "effect_modifier",
                        "actual_role": str(actual_role),
                    }
                )
        if wrong_roles:
            raise ValueError(
                "transformation_plan assigned roles inconsistent with causal_spec: "
                f"{wrong_roles}"
            )

        return self

    def _profile_by_name(self) -> dict[str, ColumnProfileModel]:
        profile_by_name: dict[str, ColumnProfileModel] = {}
        duplicate_profile_names: set[str] = set()
        for profile in self.data_summary.profiles:
            profile_name = str(profile.name).strip()
            if not profile_name:
                continue
            if profile_name in profile_by_name:
                duplicate_profile_names.add(profile_name)
                continue
            profile_by_name[profile_name] = profile

        if duplicate_profile_names:
            raise ValueError(
                "data_summary contains duplicate profile names: "
                f"{sorted(duplicate_profile_names)}"
            )
        return profile_by_name

    def get_covariates_order(self) -> list[str]:
        return [
            str(column_plan.column)
            for column_plan in self.transformation_plan.columns
            if str(column_plan.role) == "covariate"
        ]

    def get_effect_modifiers_order(self) -> list[str]:
        return [
            str(column_plan.column)
            for column_plan in self.transformation_plan.columns
            if str(column_plan.role) == "effect_modifier"
        ]

    def is_covariates_missing(self) -> bool:
        profile_by_name = self._profile_by_name()
        return any(
            profile_by_name[column].n_missing > 0
            for column in self.get_covariates_order()
        )


__all__ = ["InferenceReadyCausalSpec"]
