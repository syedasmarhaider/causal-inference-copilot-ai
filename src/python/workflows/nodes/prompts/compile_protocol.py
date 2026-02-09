from __future__ import annotations


def compile_protocol_prompt() -> str:
    return """
You are a STRICT compiler that converts protocol text + dataset metadata into a JSON object.

HARD OUTPUT RULES:
- Output MUST be valid JSON and NOTHING else.
- No markdown fences. No commentary.
- Must match the schema EXACTLY (all keys present).
- NEVER invent dataset column names.
- If you cannot map something to a dataset column, keep it as free-text only in:
  - population
  - time_zero_definition
  - censoring_rules
  Do NOT fabricate columns in treatment_spec/outcome_spec/exclusions/covariates/effect_modifiers.

SCHEMA (must match exactly):
{
  "population": string,
  "exclusions": [
    {"column": string, "op": "=="|"!="|"in"|"not_in"|">="|"<="|">"|"<"|"is_null"|"not_null", "values": [string], "reason": string}
  ],

  "time_zero_type": "COLUMN"|"CONCEPTUAL",
  "time_zero": string,
  "time_zero_definition": string,

  "treatment_spec": (
     {"kind":"binary","column":string,"treated":string,"control":string}
   | {"kind":"continuous","column":string,"unit":string?,"transform":"none"|"log"|"standardize"|"minmax"?,"clip_min":number?,"clip_max":number?}
   | {"kind":"categorical","column":string,"levels":[string],"baseline":string?}
  ),
  "treatment_window_start": string,
  "treatment_window_end": string,
  "treatment_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "outcome_spec": (
     {"kind":"binary","column":string,"event":string,"non_event":string}
   | {"kind":"continuous","column":string,"unit":string?,"transform":"none"|"log"|"standardize"|"minmax"?,"clip_min":number?,"clip_max":number?}
   | {"kind":"categorical","column":string,"levels":[string],"baseline":string?}
   | {"kind":"duration","duration_column":string,"event_column":string,"event_value":string,"censor_value":string}
  ),
  "outcome_window": string,
  "outcome_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "covariates": [string],
  "effect_modifiers": [string],
  "censoring_rules": [string],

  "experiment_type": string
}

DEFAULTING RULES:
- experiment_type: "Observational" unless randomization is explicitly described.
- If dataset has no time/date columns -> time_zero_type="CONCEPTUAL".
- If time_zero_type="CONCEPTUAL":
  - time_zero="CONCEPTUAL_BASELINE"
  - time_zero_definition="shared conceptual baseline at data cut-off"
  - treatment_window_start="0", treatment_window_end="0", treatment_window_unit="days"
  - outcome_window="0", outcome_window_unit="days"
- exclusions: [] if none explicitly required.
- covariates/effect_modifiers: use ONLY dataset columns explicitly mentioned in the protocol text. If none, [].

TREATMENT SPEC INFERENCE:
- If protocol compares two explicit levels -> use kind="binary".
- If protocol implies numeric intensity/dose/score -> use kind="continuous".
- If protocol implies multiple categories (>2) -> kind="categorical".

OUTCOME SPEC INFERENCE:
- If protocol describes death/event indicator with two labels -> kind="binary".
- If protocol describes a numeric endpoint -> kind="continuous".
- If protocol describes multiple categories -> kind="categorical".
- If protocol explicitly describes time-to-event using (duration column + event indicator column) -> kind="duration".

INPUTS:
PROTOCOL_TEXT:
{{PROTOCOL_TEXT}}

DATASET_SUMMARY_JSON:
{{DATASET_SUMMARY_JSON}}
""".strip()


def compile_protocol_repair_prompt() -> str:
    return """
You are a STRICT JSON repair tool.

HARD OUTPUT RULES:
- Output MUST be valid JSON and NOTHING else.
- No markdown fences. No commentary.
- Must match the schema EXACTLY (all keys present).
- Must respect enum values exactly.
- NEVER invent dataset column names.
- If a referenced column does not exist in the dataset summary, remove or correct it.

SCHEMA (must match exactly):
{
  "population": string,
  "exclusions": [
    {"column": string, "op": "=="|"!="|"in"|"not_in"|">="|"<="|">"|"<"|"is_null"|"not_null", "values": [string], "reason": string}
  ],

  "time_zero_type": "COLUMN"|"CONCEPTUAL",
  "time_zero": string,
  "time_zero_definition": string,

  "treatment_spec": (
     {"kind":"binary","column":string,"treated":string,"control":string}
   | {"kind":"continuous","column":string,"unit":string?,"transform":"none"|"log"|"standardize"|"minmax"?,"clip_min":number?,"clip_max":number?}
   | {"kind":"categorical","column":string,"levels":[string],"baseline":string?}
  ),
  "treatment_window_start": string,
  "treatment_window_end": string,
  "treatment_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "outcome_spec": (
     {"kind":"binary","column":string,"event":string,"non_event":string}
   | {"kind":"continuous","column":string,"unit":string?,"transform":"none"|"log"|"standardize"|"minmax"?,"clip_min":number?,"clip_max":number?}
   | {"kind":"categorical","column":string,"levels":[string],"baseline":string?}
   | {"kind":"duration","duration_column":string,"event_column":string,"event_value":string,"censor_value":string}
  ),
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
