from __future__ import annotations

from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_get_node_info,
    get_protocol_discussion_update_prompt,
    initial_user_message,
)


def test_protocol_discussion_info_describes_draft_only_node() -> None:
    info = get_protocol_discussion_get_node_info()

    assert "Draft-only causal specification node" in info
    assert "accepted causal draft artifact" in info


def test_protocol_discussion_update_prompt_uses_structured_draft_as_source_of_truth() -> None:
    prompt = get_protocol_discussion_update_prompt()

    assert "current_draft" in prompt
    assert "authoritative in-progress causal draft" in prompt
    assert "treatment_column" in prompt
    assert "outcome_column" in prompt
    assert "covariates" in prompt
    assert "effect_modifiers" in prompt
    assert "target_population" in prompt
    assert "study_type" in prompt
    assert "negative_control_outcome" in prompt
    assert "time_zero" in prompt


def test_protocol_discussion_update_prompt_keeps_target_trial_guidance_without_cleaning_questions() -> (
    None
):
    prompt = get_protocol_discussion_update_prompt()

    assert "Ask about time zero conceptually" in prompt
    assert "when follow-up starts and treatment assignment is anchored" in prompt
    assert "Covariates are baseline adjustment variables" in prompt
    assert "Effect modifiers are baseline variables used for heterogeneity" in prompt
    assert "Do not ask treatment/outcome value mapping questions." in prompt
    assert (
        "Do not ask imputation, missingness, category-handling, recoding, or cleaning questions."
        in prompt
    )


def test_initial_user_message_mentions_only_draft_fields() -> None:
    message = initial_user_message()

    assert "causal draft" in message
    assert "treatment column" in message
    assert "outcome column" in message
    assert "target population" in message
    assert "study type" in message
    assert "time zero" in message
    assert "cleaning" not in message.lower()
