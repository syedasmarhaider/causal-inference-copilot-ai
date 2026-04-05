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
