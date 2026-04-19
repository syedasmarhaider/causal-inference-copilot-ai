from __future__ import annotations

from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    summarize_upstream_data_prep_decisions,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_review_summary_prompt,
    get_protocol_discussion_update_prompt,
    get_questions,
    initial_user_message,
)


def test_protocol_discussion_questions_include_identifier_question_as_q16() -> None:
    questions = get_questions()

    assert questions[-1].startswith("16) Identifier column (optional):")
    assert any(
        question.startswith("14) Treatment/outcome data-quality decisions:")
        for question in questions
    )
    assert any(
        question.startswith("15) Baseline feature preparation decisions:")
        for question in questions
    )


def test_protocol_discussion_update_prompt_mentions_identifier_handling_rules() -> None:
    prompt = get_protocol_discussion_update_prompt()

    assert "identifier_column_candidates" in prompt
    assert "suggested_identifier_column" in prompt
    assert "Identifier column handling is optional and non-blocking." in prompt
    assert "set answer 16 to __auto_id__" in prompt
    assert "Never invent an identifier column." in prompt


def test_protocol_discussion_review_prompt_mentions_identifier_handling() -> None:
    prompt = get_protocol_discussion_review_summary_prompt()

    assert "Summarize the identifier column choice when grounded." in prompt
    assert "suggested_identifier_column" in prompt
    assert "confirming this review will accept that identifier choice" in prompt
    assert "__auto_id__ will be used" in prompt


def test_initial_user_message_mentions_identifier_selection() -> None:
    message = initial_user_message()

    assert "identifier column" in message
    assert "patient or unit" in message
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
