from __future__ import annotations

from typing import List, Set, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.utils.validation import ValidationIssueModel


def validate_dataframes_match_protocols(
    *,
    df_before: pd.DataFrame,
    protocol_before: ProtocolSpec,
    df_after: pd.DataFrame,
    protocol_after: ProtocolSpec,
) -> List[str]:
    """
    Validation logic (strict):

    BEFORE:
      - df_before must contain ALL and ONLY:
          {treatment column, outcome column, covariates, effect modifiers}
        from protocol_before.

    AFTER:
      - df_after must contain ALL and ONLY:
          {treatment column, outcome column, covariates, effect modifiers}
        from protocol_after.

    INVARIANT:
      - treatment column name must remain unchanged (protocol_before == protocol_after)
      - outcome column name must remain unchanged (protocol_before == protocol_after)

    Returns: list[str] issues (empty => valid).
    """
    issues: List[str] = []

    # -----------------------------
    # helpers
    # -----------------------------
    def _treatment_and_outcome_columns(p: ProtocolSpec) -> Tuple[str, str]:
        return p.treatment_spec.column, p.outcome_spec.column

    def _duplicates(xs: List[str]) -> List[str]:
        seen: Set[str] = set()
        dups: Set[str] = set()
        for x in xs:
            if x in seen:
                dups.add(x)
            seen.add(x)
        return sorted(dups)

    def _required_columns(p: ProtocolSpec) -> Set[str]:
        treatment_column, outcome_column = _treatment_and_outcome_columns(p)
        cols = [treatment_column, outcome_column, *p.covariates, *p.effect_modifiers]

        dups = _duplicates(cols)
        if dups:
            issues.append(f"Protocol has duplicate column names across roles: {dups}")

        return set(cols)

    def _check_no_duplicate_dataframe_columns(df: pd.DataFrame, label: str) -> None:
        if df.columns.has_duplicates:
            dup_cols = sorted(set(df.columns[df.columns.duplicated()].tolist()))
            issues.append(f"{label} has duplicate column labels (pandas allows this): {dup_cols}")

    def _check_exact_column_set(df: pd.DataFrame, required: Set[str], label: str) -> None:
        have = set(df.columns.tolist())
        missing = sorted(required - have)
        extra = sorted(have - required)
        if missing:
            issues.append(f"{label} is missing required columns: {missing}")
        if extra:
            issues.append(f"{label} has extra columns not allowed by protocol: {extra}")

    # -----------------------------
    # dataframe structural sanity
    # -----------------------------
    _check_no_duplicate_dataframe_columns(df_before, "df_before")
    _check_no_duplicate_dataframe_columns(df_after, "df_after")

    # -----------------------------
    # treatment/outcome must be unchanged
    # -----------------------------
    treatment_before, outcome_before = _treatment_and_outcome_columns(protocol_before)
    treatment_after, outcome_after = _treatment_and_outcome_columns(protocol_after)

    if treatment_before != treatment_after:
        issues.append(
            f"Treatment column changed between protocols: before='{treatment_before}' after='{treatment_after}'"
        )
    if outcome_before != outcome_after:
        issues.append(
            f"Outcome column changed between protocols: before='{outcome_before}' after='{outcome_after}'"
        )

    # -----------------------------
    # strict column-set checks
    # -----------------------------
    required_before = _required_columns(protocol_before)
    required_after = _required_columns(protocol_after)

    _check_exact_column_set(df_before, required_before, "df_before")
    _check_exact_column_set(df_after, required_after, "df_after")

    return issues


def validate_covariates_and_effect_modifiers_numeric_only(
    *,
    df_after: pd.DataFrame,
    protocol_after: ProtocolSpec,
) -> List[ValidationIssueModel]:
    """
    Minimal post-transform validation:
      - Checks covariates + effect modifiers exist in df_after
      - Checks they are numeric dtype (bool NOT allowed)
      - Does NOT validate treatment/outcome
    """
    issues: List[ValidationIssueModel] = []

    cols = list(protocol_after.covariates) + list(protocol_after.effect_modifiers)

    missing = sorted([c for c in cols if c not in df_after.columns])
    if missing:
        issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Missing covariate/effect-modifier columns in transformed dataframe.",
                evidence={
                    "missing_columns": missing,
                    "expected_covariates": list(protocol_after.covariates),
                    "expected_effect_modifiers": list(protocol_after.effect_modifiers),
                },
                fix_hint="Ensure the transform plan produces all covariates and effect modifiers listed in the protocol.",
            )
        )
        return issues

    non_numeric = [{"column": c, "dtype": str(df_after[c].dtype)} for c in cols if not is_numeric_dtype(df_after[c])]
    if non_numeric:
        issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Some covariates/effect modifiers are not numeric after transformation.",
                evidence={"non_numeric_columns": non_numeric},
                fix_hint="Encode these covariates/effect modifiers to numeric before model fitting (e.g., one-hot / ordinal / to_numeric).",
            )
        )

    return issues