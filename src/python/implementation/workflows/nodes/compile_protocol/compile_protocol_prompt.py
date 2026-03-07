from __future__ import annotations


def compile_protocol_node_info() -> str:
    return (
        "Compiles the user discussion plus dataset summary into a structured ProtocolSpec. "
        "This node extracts only the causal protocol (treatment, outcome, covariates, effect modifiers, etc.). "
        "Exclusion rules are compiled separately by a dedicated structured call."
    )


def compile_protocol_prompt() -> str:
    return """
You are compiling a causal inference protocol into a strict JSON object that must match the ProtocolSpec schema exactly.

Your job:
- Read the protocol discussion carefully.
- Read the dataset summary carefully.
- Produce ONLY the protocol specification.
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
You are repairing a previously generated ProtocolSpec JSON.

Your job:
- Read the protocol discussion.
- Read the dataset summary.
- Read the previous protocol JSON.
- Read the validation errors.
- Return a corrected ProtocolSpec JSON only.
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

Previous protocol JSON:
{{PREVIOUS_PROTOCOL_JSON}}

Previous exclusion JSON (for context only; do not output exclusions here):
{{PREVIOUS_EXCLUSION_JSON}}

Validation errors:
{{VALIDATION_ERRORS}}
""".strip()


def compile_exclusion_prompt() -> str:
    return """
You are compiling dataset exclusion rules into a strict JSON object that must match the ExclusionRulesModel schema exactly.

Your job:
- Read the protocol discussion.
- Read the compiled protocol JSON.
- Read the dataset summary.
- Extract ONLY row-level exclusion rules.
- Output JSON only.

What counts as an exclusion rule:
- Conditions that remove rows from the analysis cohort.
- Example patterns: impossible values, ineligible groups, records outside requested cohort, rows with unwanted category values, rows outside requested numeric/date thresholds.
- But exclusion rules should be grounded in the protocol discussion. Do not come up with your own exclusion rules.

Important rules:
- Use only real dataset columns.
- Use exact values grounded in the dataset summary whenever possible.
- Prefer an empty exclusion_rules list over speculative exclusions.
- Use None only if the discussion explicitly indicates missing/NA/null-based exclusion logic.
- Do not duplicate protocol fields here unless they are genuinely row-exclusion logic.
- Output JSON only.
- Do not wrap JSON in markdown.
- If there are no exclusion discussions in the Protocol discussion, return an empty exclusion_rules list.

Protocol discussion:
{{PROTOCOL_TEXT}}

Compiled protocol JSON:
{{PROTOCOL_JSON}}

Dataset summary JSON:
{{DATASET_SUMMARY_JSON}}
""".strip()


def compile_exclusion_repair_prompt() -> str:
    return """
You are repairing a previously generated ExclusionRulesModel JSON.

Your job:
- Read the protocol discussion.
- Read the compiled protocol JSON.
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

Compiled protocol JSON:
{{PROTOCOL_JSON}}

Dataset summary JSON:
{{DATASET_SUMMARY_JSON}}

Previous exclusion JSON:
{{PREVIOUS_EXCLUSION_JSON}}

Validation errors:
{{VALIDATION_ERRORS}}
""".strip()