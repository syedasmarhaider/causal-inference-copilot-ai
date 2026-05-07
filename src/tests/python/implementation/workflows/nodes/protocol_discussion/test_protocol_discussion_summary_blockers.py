from __future__ import annotations

from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_summary_blockers import (
    build_summary_blocker_follow_up_message,
    scan_protocol_summary_blockers,
    unresolved_summary_blockers,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


def _summary_payload() -> DatasetSummaryModel:
    return DatasetSummaryModel.model_validate(
        {
            "n_rows": 100,
            "profiles": [
                {
                    "name": "btransf",
                    "dtype": "int32",
                    "n_rows": 100,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "distinct_count": 2,
                    "inferred_kind": "NUMERIC",
                    "summary": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5, "quantiles": None},
                },
                {
                    "name": "istatus",
                    "dtype": "int32",
                    "n_rows": 100,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "distinct_count": 3,
                    "inferred_kind": "NUMERIC",
                    "summary": {"min": 1.0, "max": 3.0, "mean": 1.9, "std": 0.6, "quantiles": None},
                },
                {
                    "name": "isbp",
                    "dtype": "float64",
                    "n_rows": 100,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "distinct_count": 80,
                    "inferred_kind": "NUMERIC",
                    "summary": {"min": 60.0, "max": 200.0, "mean": 98.0, "std": 22.0, "quantiles": None},
                },
                {
                    "name": "iage",
                    "dtype": "float64",
                    "n_rows": 100,
                    "n_missing": 1,
                    "missing_rate": 0.01,
                    "distinct_count": 70,
                    "inferred_kind": "NUMERIC",
                    "summary": {"min": 18.0, "max": 95.0, "mean": 44.0, "std": 14.0, "quantiles": None},
                },
                {
                    "name": "isex",
                    "dtype": "object",
                    "n_rows": 100,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "distinct_count": 3,
                    "inferred_kind": "CATEGORICAL",
                    "summary": {
                        "top_categories": [
                            {"value": "Male", "count": 60},
                            {"value": "Female", "count": 39},
                            {"value": "Other/Unknown", "count": 1},
                        ],
                        "other_count": 0,
                    },
                },
            ],
        }
    )


def test_scan_protocol_summary_blockers_finds_selected_column_issues() -> None:
    blockers = scan_protocol_summary_blockers(
        dataset_summary=_summary_payload(),
        treatment_column="btransf",
        outcome_column="istatus",
        covariates=["isbp"],
        effect_modifiers=["iage", "isex"],
    )

    blocker_pairs = {(blocker.column, blocker.issue) for blocker in blockers}

    assert ("istatus", "outcome_mapping_required") in blocker_pairs
    assert ("iage", "missing_values") in blocker_pairs
    assert ("isex", "unknown_category_present") in blocker_pairs
    assert ("isbp", "missing_values") not in blocker_pairs


def test_unresolved_summary_blockers_requires_explicit_q14_q15_answers() -> None:
    blockers = scan_protocol_summary_blockers(
        dataset_summary=_summary_payload(),
        treatment_column="btransf",
        outcome_column="istatus",
        covariates=["isbp"],
        effect_modifiers=["iage", "isex"],
    )
    discussion = "\n".join(
        [
            "8) Outcome specification: Death defined by istatus.",
            "14) Treatment/outcome data-quality decisions: For istatus, map 1 to Dead and 2/3 to Alive before modeling.",
            "15) Baseline feature preparation decisions: UNCLEAR",
        ]
    )

    unresolved = unresolved_summary_blockers(protocol_discussion=discussion, blockers=blockers)
    unresolved_pairs = {(blocker.column, blocker.issue) for blocker in unresolved}

    assert ("istatus", "outcome_mapping_required") not in unresolved_pairs
    assert ("iage", "missing_values") in unresolved_pairs
    assert ("isex", "unknown_category_present") in unresolved_pairs


def test_unresolved_summary_blockers_accepts_global_baseline_unknown_decision() -> None:
    blockers = scan_protocol_summary_blockers(
        dataset_summary=_summary_payload(),
        treatment_column="btransf",
        outcome_column="istatus",
        covariates=[],
        effect_modifiers=["isex"],
    )
    discussion = "\n".join(
        [
            "8) Outcome specification: Death defined by istatus.",
            "14) Treatment/outcome data-quality decisions: For istatus, map 1 to Dead and 2/3 to Alive before modeling.",
            (
                "15) Baseline feature preparation decisions: Keep Unknown and Other/Unknown "
                "as their own category for all selected covariates and effect modifiers."
            ),
        ]
    )

    unresolved = unresolved_summary_blockers(protocol_discussion=discussion, blockers=blockers)
    unresolved_pairs = {(blocker.column, blocker.issue) for blocker in unresolved}

    assert ("isex", "unknown_category_present") not in unresolved_pairs


def test_build_summary_blocker_follow_up_message_is_direct_and_actionable() -> None:
    blockers = scan_protocol_summary_blockers(
        dataset_summary=_summary_payload(),
        treatment_column="btransf",
        outcome_column="istatus",
        covariates=["isbp"],
        effect_modifiers=["iage"],
    )

    message = build_summary_blocker_follow_up_message(blockers)

    assert "Before I lock this protocol" in message
    assert "deterministic cleaning and encoding instructions" in message
    assert "validation/refutation results" in message
    assert "istatus" in message
    assert "iage" in message
    assert "explicit cleaning instructions" in message
