from __future__ import annotations

from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_causal_draft_prompt,
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_response_prompt,
    get_protocol_discussion_template,
    get_protocol_discussion_update_prompt,
    get_protocol_discussion_validation_suggestion_prompt,
    initial_user_message,
)


def test_protocol_discussion_info_describes_discussion_and_final_draft_compile() -> None:
    info = get_protocol_discussion_get_node_info()

    assert "protocol discussion string" in info
    assert "DISCUSSING, REVIEW, and READY" in info
    assert "causal specification draft" in info


def test_protocol_discussion_update_prompt_uses_protocol_string_contract() -> None:
    prompt = get_protocol_discussion_update_prompt()

    assert "previous_protocol_discussion" in prompt
    assert "latest_user_message" in prompt
    assert "Source: user" in prompt
    assert "Source: data" in prompt
    assert "status" in prompt
    assert "Treatment must be binary" in prompt
    assert "Outcome must be binary or continuous" in prompt


def test_protocol_discussion_template_uses_expected_questions_and_auto_id() -> None:
    template = get_protocol_discussion_template()

    assert "Q1: Treatment" in template
    assert "Q2: Outcome" in template
    assert "Q5: ID column" in template
    assert "auto_id" in template
    assert "Source: unclear" in template


def test_protocol_discussion_response_prompt_is_plain_text_not_json() -> None:
    prompt = get_protocol_discussion_response_prompt()

    assert "Return only the user-facing assistant message as plain text" in prompt
    assert "Do not wrap the message in JSON" in prompt
    assert "Output JSON exactly" not in prompt


def test_protocol_discussion_causal_draft_prompt_is_grounded() -> None:
    prompt = get_protocol_discussion_causal_draft_prompt()

    assert "signed-off protocol discussion" in prompt
    assert "exact dataset column names" in prompt
    assert "auto_id" in prompt
    assert "Return only JSON matching the requested causal draft schema" in prompt


def test_protocol_discussion_validation_suggestion_prompt_requires_update_dataset() -> None:
    prompt = get_protocol_discussion_validation_suggestion_prompt()

    assert "validation_issues" in prompt
    assert "Every dataset-change suggestion must start" in prompt
    assert "update dataset" in prompt
    assert "Return plain text only" in prompt


def test_initial_user_message_starts_protocol_discussion() -> None:
    message = initial_user_message()

    assert "Welcome" in message
    assert "treatment" in message
    assert "outcome" in message
    assert "negative-control outcome" in message
