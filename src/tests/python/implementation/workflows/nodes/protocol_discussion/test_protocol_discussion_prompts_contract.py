from __future__ import annotations

from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_review_summary_prompt,
    get_protocol_discussion_update_prompt,
    get_questions,
    initial_user_message,
    summarize_upstream_data_prep_decisions,
)


def test_protocol_discussion_questions_include_upstream_data_prep_items() -> None:
    questions = get_questions()

    assert any(
        question.startswith("14) Treatment/outcome data-quality decisions:")
        for question in questions
    )
    assert any(
        question.startswith("15) Baseline feature preparation decisions:")
        for question in questions
    )


def test_protocol_discussion_update_prompt_mentions_blockers_and_grounded_baseline_prep() -> None:
    prompt = get_protocol_discussion_update_prompt()

    assert "Surface only blockers that would prevent safe compilation" in prompt
    assert "Carry forward the confirmed upstream data-preparation decisions" in prompt
    assert "approved imputation or unknown-category handling explicitly" in prompt
    assert "Do not invent filters, drops, mappings, imputations, or normalization rules" in prompt


def test_protocol_discussion_review_prompt_mentions_upstream_data_prep_decisions() -> None:
    prompt = get_protocol_discussion_review_summary_prompt()

    assert "approved upstream data-preparation decisions" in prompt
    assert "baseline feature preparation decisions" in prompt


def test_initial_user_message_mentions_upstream_data_handling() -> None:
    message = initial_user_message()

    assert "upstream data-handling decisions" in message
    assert "treatment, outcome, or baseline features" in message


def test_summarize_upstream_data_prep_decisions_collects_questions_14_and_15() -> None:
    discussion = "\n".join(
        [
            "1) Causal question: effect of transfusion on mortality.",
            "14) Treatment/outcome data-quality decisions: Map istatus so 1=Dead and 2/3=Alive.",
            "15) Baseline feature preparation decisions: Impute missing iage values before modeling and keep isex unknown as its own category.",
        ]
    )

    summary = summarize_upstream_data_prep_decisions(discussion)

    assert summary is not None
    assert "Map istatus so 1=Dead and 2/3=Alive." in summary
    assert "Impute missing iage values before modeling" in summary


def test_summarize_upstream_data_prep_decisions_skips_unclear_items() -> None:
    discussion = "\n".join(
        [
            "14) Treatment/outcome data-quality decisions: UNCLEAR",
            "15) Baseline feature preparation decisions: UNCLEAR",
        ]
    )

    assert summarize_upstream_data_prep_decisions(discussion) is None
