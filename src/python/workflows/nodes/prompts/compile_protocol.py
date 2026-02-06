from __future__ import annotations


def compile_protocol_prompt() -> str:
    return """
You are a STRICT compiler that converts protocol text + dataset metadata into a JSON object.

HARD OUTPUT RULES:
- Output MUST be valid JSON and NOTHING else.
- No markdown fences. No commentary.
- Must match schema EXACTLY (all keys present).
- Never invent column names. If you cannot map to a column, keep it as free-text within strings (population/treatment/outcome),
  but do NOT fabricate dataset columns.

Schema:
{
  "population": string,
  "exclusions": [
    {"column": string, "op": "=="|"!="|"in"|"not_in"|">="|"<="|">"|"<"|"is_null"|"not_null", "values": [string], "reason": string}
  ],
  "time_zero_type": "COLUMN"|"CONCEPTUAL",
  "time_zero": string,
  "time_zero_definition": string,

  "treatment": string,
  "treatment_window_start": string,
  "treatment_window_end": string,
  "treatment_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "outcome": string,
  "outcome_is_duration": boolean,
  "outcome_window": string,
  "outcome_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "covariates": [string],
  "effect_modifiers": [string],
  "censoring_rules": [string],
  "experiment_type": string
}

Defaulting rules:
- experiment_type: "Observational" unless an RCT randomization variable is explicitly described.
- If dataset has no time/date columns -> time_zero_type="CONCEPTUAL".
- If time_zero_type="CONCEPTUAL":
  - time_zero="CONCEPTUAL_BASELINE"
  - time_zero_definition="shared conceptual baseline at data cut-off"
  - treatment_window_start="0", treatment_window_end="0", treatment_window_unit="days"
  - outcome_window="0", outcome_window_unit="days"
  - outcome_is_duration=false unless duration fields exist and are explicitly used
- exclusions: [] if none explicitly required.

INPUTS:
PROTOCOL_TEXT:
{{PROTOCOL_TEXT}}

DATASET_SUMMARY_JSON:
{{DATASET_SUMMARY_JSON}}
""".strip()


def compile_protocol_repair_prompt() -> str:
    return """
You are a STRICT JSON repair tool.

You will be given:
(1) the original protocol text
(2) authoritative dataset summary
(3) previous JSON output
(4) validation errors

Your job:
- Output a FIXED JSON object matching the schema EXACTLY.
- Output MUST be valid JSON ONLY. No markdown. No commentary.
- Must include ALL keys.
- Must respect enum values exactly.
- Never invent dataset columns. If an exclusion references a non-existent column, remove it or correct it.

Schema:
{
  "population": string,
  "exclusions": [
    {"column": string, "op": "=="|"!="|"in"|"not_in"|">="|"<="|">"|"<"|"is_null"|"not_null", "values": [string], "reason": string}
  ],
  "time_zero_type": "COLUMN"|"CONCEPTUAL",
  "time_zero": string,
  "time_zero_definition": string,

  "treatment": string,
  "treatment_window_start": string,
  "treatment_window_end": string,
  "treatment_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "outcome": string,
  "outcome_is_duration": boolean,
  "outcome_window": string,
  "outcome_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "covariates": [string],
  "effect_modifiers": [string],
  "censoring_rules": [string],
  "experiment_type": string
}

INPUTS:
PROTOCOL_TEXT:
{{PROTOCOL_TEXT}}

DATASET_SUMMARY_JSON:
{{DATASET_SUMMARY_JSON}}

PREVIOUS_JSON:
{{PREVIOUS_JSON}}

VALIDATION_ERRORS:
{{VALIDATION_ERRORS}}
""".strip()
