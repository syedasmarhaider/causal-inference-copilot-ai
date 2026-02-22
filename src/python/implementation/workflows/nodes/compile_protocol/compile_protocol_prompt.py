from __future__ import annotations


def compile_protocol_node_info() -> str:
 return """
        "Convert the confirmed discussion record into a strict ProtocolState. "
        "Enforce schema/enums; do not invent columns/windows/semantics. "
        "If compilation fails, route back to PROTOCOL_DISCUSSION with precise fix instructions."
    """

def compile_protocol_prompt() -> str:
    return """
You are a STRICT compiler that converts protocol text + dataset metadata into a JSON object.
If there is a router message, follow the routers intructions.

HARD OUTPUT RULES:
- Output MUST be valid JSON and NOTHING else.
- No markdown. No commentary.
- Must match the schema EXACTLY (all keys present).
- NEVER invent dataset column names.
- For any field that requires a column name (exclusions/treatment_spec/outcome_spec/covariates/effect_modifiers),
  you MUST choose from DATASET_COLUMNS only.
- For any field that requires categorical values (treated_values/control_values/event_values/non_event_values/included_levels),
  you MUST choose EXACT string tokens from DATASET_VALUE_VOCAB for that column.

STRICT VALUE RULE:
- If protocol says "Former/Current" but the dataset tokens are ["Former","Current"], you must output:
  treated_values=["Former","Current"] (exact tokens).
- Do NOT output labels that do not appear in the dataset vocabulary for that column.

EXCLUSIONS SEMANTICS:
- "exclusions" are ROWS TO DROP.
- Do NOT use exclusions to implement treatment/outcome inclusion.
- If no explicit exclusions -> exclusions: [].
- Nul/miising values are automatically excluded, do NOT add exclusions for that.

SCHEMA (must match exactly):
{
  "exclusions": [
    {"column": string, "op": "=="|"in"|"not_in"|">="|"<="|">"|"<", "values": [string], "reason": string}
  ],

  "time_zero_type": "COLUMN"|"CONCEPTUAL",
  "time_zero": string,
  "time_zero_definition": string,

  "treatment_spec": (
     {"kind":"binary","column":string,"treated_values":[string],"control_values":[string]}
   | {"kind":"categorical","column":string,"included_levels":[string]}
  ),
  "treatment_window_start": string,
  "treatment_window_end": string,
  "treatment_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "outcome_spec": (
     {"kind":"binary","column":string,"event_values":[string],"non_event_values":[string]}
   | {"kind":"continuous","column":string,"valid_min":number?,"valid_max":number?}
   | {"kind":"categorical","column":string,"included_levels":[string]}
  ),
  "outcome_window": string,
  "outcome_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "covariates": [string],
  "effect_modifiers": [string],

  "experiment_type": "RCT"|"Observational"
}

DEFAULTS:
- experiment_type: "Observational" unless randomization is explicitly described.
- If dataset has no time/date columns -> time_zero_type="CONCEPTUAL".
- If time_zero_type="CONCEPTUAL":
  - time_zero="CONCEPTUAL_BASELINE"
  - time_zero_definition="shared conceptual baseline at data cut-off"
  - treatment_window_start="0", treatment_window_end="0", treatment_window_unit="days"
  - outcome_window="0", outcome_window_unit="days"
- covariates/effect_modifiers: ONLY dataset columns explicitly mentioned in the protocol text; else [].

INPUTS:
PROTOCOL_TEXT:
{{PROTOCOL_TEXT}}
{{ROUTER_MESSAGE}}

DATASET_COLUMNS (ONLY allowed column names):
{{DATASET_COLUMNS_JSON}}

DATASET_VALUE_VOCAB (EXACT allowed categorical tokens per column):
{{DATASET_VALUE_VOCAB_JSON}}

DATASET_SUMMARY_JSON:
{{DATASET_SUMMARY_JSON}}
""".strip()


def compile_protocol_repair_prompt() -> str:
    return """
You are a STRICT JSON repair tool.

HARD OUTPUT RULES:
- Output MUST be valid JSON and NOTHING else.
- No markdown. No commentary.
- Must match the schema EXACTLY (all keys present).
- NEVER invent dataset column names (must use DATASET_COLUMNS).
- NEVER invent categorical tokens (must use DATASET_VALUE_VOCAB for that column).

Fix the JSON so that:
- all required keys exist
- enums are correct
- columns exist in DATASET_COLUMNS
- categorical tokens are EXACT members of DATASET_VALUE_VOCAB[column]

SCHEMA (must match exactly):
{
  "exclusions": [
    {"column": string, "op": "=="|"in"|"not_in"|">="|"<="|">"|"<", "values": [string], "reason": string}
  ],

  "time_zero_type": "COLUMN"|"CONCEPTUAL",
  "time_zero": string,
  "time_zero_definition": string,

  "treatment_spec": (
     {"kind":"binary","column":string,"treated_values":[string],"control_values":[string]}
   | {"kind":"categorical","column":string,"included_levels":[string]}
  ),
  "treatment_window_start": string,
  "treatment_window_end": string,
  "treatment_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "outcome_spec": (
     {"kind":"binary","column":string,"event_values":[string],"non_event_values":[string]}
   | {"kind":"continuous","column":string,"valid_min":number?,"valid_max":number?}
   | {"kind":"categorical","column":string,"included_levels":[string]}
  ),
  "outcome_window": string,
  "outcome_window_unit": "minutes"|"hours"|"days"|"weeks"|"months"|"years",

  "covariates": [string],
  "effect_modifiers": [string],

  "experiment_type": "RCT"|"Observational"
}

INPUTS:
PROTOCOL_TEXT:
{{PROTOCOL_TEXT}}

DATASET_COLUMNS:
{{DATASET_COLUMNS_JSON}}

DATASET_VALUE_VOCAB:
{{DATASET_VALUE_VOCAB_JSON}}

DATASET_SUMMARY_JSON:
{{DATASET_SUMMARY_JSON}}

PREVIOUS_JSON:
{{PREVIOUS_JSON}}

VALIDATION_ERRORS:
{{VALIDATION_ERRORS}}
""".strip()



def protocol_validate_through_llm_prompt() -> str:
    return """You are a STRICT protocol validator.
  You will see the protcol JSON, the original protocol discussion 
  text, and the dataset summary.
  Validate that the JSON is a correct interpretation of the protocol discussion and is semantically consistent with the dataset summary.
  Check if columes specified exists their types are correct, inclusion crtieria is good.
  return detail response what is wrong and how to fix it if there are any issues otherwise return only one token "VALID".
INPUTS:
PROTOCOL_JSON:
{{PROTOCOL_JSON}}
PROTOCOL_DISCUSSION_TEXT:
{{PROTOCOL_DISCUSSION_TEXT}}
DATASET_SUMMARY_JSON:
{{DATASET_SUMMARY_JSON}}
""".strip()
  