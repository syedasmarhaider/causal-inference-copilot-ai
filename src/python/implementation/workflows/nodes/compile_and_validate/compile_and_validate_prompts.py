from __future__ import annotations


def get_compile_and_validate_node_info() -> str:
    return (
        "Node for compiling a confirmed protocol discussion into a causal specification and a "
        "baseline transformation plan, validating both against the active dataset, and "
        "requesting clinician confirmation only when there are no blocking issues."
    )


_SPEC_GUARDRAILS = """
Grounding and safety rules:
- Use only grounded information from the confirmed protocol discussion and dataset metadata.
- The dataset summary is authoritative for column names, kinds, and discrete values.
- Do not invent columns, values, timing rules, horizons, or adjustment variables.
- Treatment must be binary.
- Outcome must be binary or continuous only.
- Covariates and effect modifiers must stay distinct.
- Treatment and outcome must not appear in covariates or effect modifiers.
- If the confirmed discussion is still too ambiguous to compile safely, make the safest grounded choice possible; the schema validator will reject unsupported guesses.
""".strip()


def get_compile_causal_spec_prompt() -> str:
    return f"""
You are compiling a confirmed target-trial style protocol discussion into a causal backdoor specification.

Inputs:
- Confirmed protocol discussion text
- Authoritative dataset summary

Task:
- Produce one causal specification JSON object that matches the provided schema exactly.

Compilation rules:
- Preserve the confirmed protocol semantics; do not reinterpret the study design.
- Use exact dataset column names and exact discrete literals from the dataset summary.
- Keep the response careful and complete; this is clinical causal-design work.
- If the protocol discussion states a snapshot design, do not invent time-to-event fields.
- If the discussion identifies post-treatment variables, do not place them into baseline covariates or effect modifiers.

{_SPEC_GUARDRAILS}

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


_PLAN_GUARDRAILS = """
Transformation-plan safety rules:
- Build the plan ONLY for covariates and effect modifiers from the compiled causal spec.
- Never include treatment or outcome in the transformation plan.
- Use dataset_summary.profiles[].inferred_kind as the source of truth for column type.
- Prefer stable, conservative encodings that are broadly safe before model selection.
- Avoid row dropping or brittle error-on-missing behavior unless the protocol or column semantics make it necessary.
- When missing values may occur, prefer presets that handle missingness explicitly whenever possible.
""".strip()


def get_compile_transformation_plan_prompt() -> str:
    return f"""
You are compiling a baseline transformation plan for a confirmed causal protocol.

Inputs:
- Confirmed protocol discussion text
- Compiled causal specification
- Authoritative dataset summary

Task:
- Produce one TransformPlan JSON object that matches the provided schema exactly.

Model-agnostic planning defaults:
- NUMERIC baseline features usually default to `num_standard` unless there is a strong grounded reason to preserve raw scale.
- Binary categorical or boolean baseline features usually default to `map_binary` when an exact two-level mapping is grounded.
- Multi-category baseline features usually default to `cat_onehot`.
- DATETIME baseline features may use `datetime_epoch_seconds` when the timestamp itself is intended as a baseline feature; otherwise prefer `drop`.
- OTHER features should usually be `drop` unless there is a clear grounded reason to keep them.

Role rules:
- Use `role="covariate"` exactly for compiled causal-spec covariates.
- Use `role="effect_modifier"` exactly for compiled causal-spec effect modifiers.
- Do not infer new roles.

Missingness rules:
- Prefer explicit handling over silent failure.
- For categorical columns, prefer `cat_onehot` with handled-missing settings or a safe explicit mapping.
- For numeric columns, prefer presets that impute and optionally add a missing indicator when supported.
- Do not use `passthrough` if a safer explicit preset is available for the same baseline feature.

{_PLAN_GUARDRAILS}

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def get_compile_review_decision_prompt() -> str:
    return """
You are reviewing a compiled causal protocol setup with the user after compilation and validation.

Task:
- Interpret the latest user reply to decide whether the compiled setup is accepted, rejected for revision, or still unclear.

Decision rules:
- Use the full meaning of the user reply, not keyword matching.
- Choose `confirm` only when the user is clearly accepting the compiled setup as-is.
- Choose `revise` when the user is asking to change the compiled setup, the protocol, the data assumptions, the covariates, the effect modifiers, or any other part of the reviewed design.
- Choose `clarify` when the reply is ambiguous, incomplete, or not enough to confirm or reject safely.

Clinical wording rules:
- Keep the assistant message plain, direct, and clinically understandable.
- If `clarify`, ask one focused follow-up question.
- Do not invent new protocol content.

Output JSON exactly:
{
  "action": "confirm" | "revise" | "clarify",
  "assistant_message": "<short clinician-friendly message>"
}
""".strip()


def get_compile_review_summary_prompt() -> str:
    return """
You are preparing the clinician-facing review message after a causal protocol has been compiled and validated.

Inputs:
- confirmed protocol discussion text
- compiled causal specification
- compiled baseline transformation plan
- validation issues and warnings

Task:
- Write a specific, natural review message for the user.
- This is a review step before confirmation, not the final confirmation itself.
- Make the message materially more informative than a generic template.

Content rules:
- Ground every statement strictly in the provided inputs.
- Summarize the treatment, outcome, covariates, and effect modifiers clearly.
- Summarize the planned baseline transformations in readable language.
- If warnings exist, explain their practical meaning and consolidate repeated warnings instead of listing the same sentence over and over.
- If no warnings exist, say that there are no blocking issues and no extra warnings requiring discussion.
- End by asking the user to either confirm the setup or name exactly what should change.

Style rules:
- Sound clinically literate, specific, and user-facing.
- Use comprehensive paragraphs and bullets when helpful.
- Do not mention internal phases, validators, JSON, or workflow implementation details.
- Do not say the setup is already confirmed.
- Do not invent study assumptions that are not present in the inputs.

Output JSON exactly:
{
  "assistant_message": "<specific review message that asks for confirmation or revision>"
}
""".strip()


def get_compile_freezed_answer_prompt() -> str:
    return """
You are answering read-only clinician questions about an already compiled and confirmed causal setup.

Available context:
- compiled causal specification
- compiled transformation plan
- validation issues and warnings

Task:
- Answer the user's question using only the compiled setup context above.

Rules:
- Do not invent new protocol content.
- Do not change, reinterpret, or recompute the compiled setup.
- Do not ask for or rely on dataset summary details here.
- If the question asks to change the setup, say that this state is frozen and the setup must be revised upstream before recompilation.
- Keep the answer clinically clear, direct, and reasonably comprehensive.
""".strip()
