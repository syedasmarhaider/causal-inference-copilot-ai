from __future__ import annotations


def compile_protocol_node_info() -> str:
    return (
        "Compiles the user discussion plus dataset summary into a structured CausalSpec. "
        "This node extracts only the causal protocol (treatment, outcome, covariates, effect modifiers, etc.). "
        "Exclusion rules are compiled separately by a dedicated structured call."
    )


def compile_protocol_prompt() -> str:
    return """
You are compiling a causal inference protocol into a strict JSON object that must match the CausalSpec schema exactly.

Your job:
- Read the protocol discussion carefully.
- Read the dataset summary carefully.
- Produce ONLY the causal_spec specification.
- Do NOT include exclusion rules here. Exclusions are compiled separately.
- Use only column names that exist in the dataset summary.
- Use exact literal values when treatment/outcome values are categorical or boolean-like.
- Do not invent columns, categories, encodings, thresholds, or assumptions that are not grounded in the discussion or dataset summary.
- If the discussion is ambiguous, choose the most conservative, simplest, schema-valid interpretation.
- Respect the data type please. Always respect datatype and use exact values from the dataset summary for categorical/binary fields.
- Output JSON only.

Important rules:
- Treatment and outcome columns must be real dataset columns.
- For binary treatment/outcome values, use exact values consistent with the dataset summary whenever possible.
- For continuous outcomes, select a numeric outcome column only.
- Do not emit explanatory text.
- Do not wrap JSON in markdown.

Protocol discussion:
{{PROTOCOL_TEXT}}

Dataset summary JSON:
{{DATASET_SUMMARY_JSON}}
""".strip()


def compile_protocol_repair_prompt() -> str:
    return """
You are repairing a previously generated CausalSpec JSON.

Your job:
- Read the protocol discussion.
- Read the dataset summary.
- Read the previous CausalSpec JSON.
- Read the validation errors.
- Return a corrected CausalSpec JSON only.
- Do NOT include exclusion rules here. Exclusions are compiled separately.
- Fix only what is necessary to make the protocol valid and faithful to the discussion and dataset.
- Do not invent columns, values, or assumptions.
- Output JSON only.

Repair priorities:
1. Use only real dataset columns.
2. Use exact literal values grounded in the dataset summary.
3. Keep the protocol semantically faithful to the user discussion.
4. Prefer omission over speculation.

Protocol discussion:
{{PROTOCOL_TEXT}}

Dataset summary JSON:
{{DATASET_SUMMARY_JSON}}

Previous CausalSpec JSON:
{{PREVIOUS_CAUSAL_SPEC_JSON}}

Previous exclusion JSON (for context only; do not output exclusions here):
{{PREVIOUS_EXCLUSION_JSON}}

Validation errors:
{{VALIDATION_ERRORS}}
""".strip()


def compile_exclusion_repair_prompt() -> str:
    return """
You are repairing a previously generated ExclusionRulesModel JSON.

Your job:
- Read the protocol discussion.
- Read the compiled causal_spec JSON.
- Read the dataset summary.
- Read the previous exclusion JSON.
- Read the validation errors.
- Return a corrected ExclusionRulesModel JSON only.
- Output JSON only.

Repair priorities:
1. Use only real dataset columns.
2. Use exact values grounded in the dataset summary.
3. Keep only true row-level exclusion logic.
4. Prefer fewer rules over speculative rules.
5. Preserve valid rules and fix only what is broken.

Protocol discussion:
{{PROTOCOL_TEXT}}

Compiled causal_spec JSON:
{{CAUSAL_SPEC_JSON}}

Dataset summary JSON:
{{DATASET_SUMMARY_JSON}}

Previous exclusion JSON:
{{PREVIOUS_EXCLUSION_JSON}}

Validation errors:
{{VALIDATION_ERRORS}}
""".strip()