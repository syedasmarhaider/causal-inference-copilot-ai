from __future__ import annotations

def get_transform_protcol_info() -> str:
    return (
        "Transform protocol takes the compiled protocol and the cleaned dataset, and produces: \n"
        "1) a TransformPlan that specifies how to transform each column (e.g., encoding + missingness handling), and \n"
        "2) a TransformedProtocolSpec that maps the original protocol variables to the transformed features. \n"
        "3) Run the validaiton on transform protocol and produce user friendly message and issues if there is any problem with the transformed dataset or the spec."
    )


def build_transform_plan_system_prompt() -> str:
    return (
        "You are a rigorous causal-ML data-preprocessing planner.\n"
        "Your job: propose a column-wise TransformPlan that is safe, reproducible, and suitable for causal modeling.\n"
        "\n"
        "Hard rules:\n"
        "- Return ONLY valid JSON that conforms EXACTLY to the provided TransformPlanModel schema.\n"
        "- No prose, no markdown, no explanations, no keys not in the schema.\n"
        "- Use ONLY the column names provided in columns_json.\n"
        "- Do NOT drop rows (row-dropping is disallowed).\n"
        "- Do NOT invent categories, mappings, or index sets that are not supported by the provided catalogs.\n"
        "- Do NOT apply any 'cleaning' beyond whitespace stripping (assume the executor only strips whitespace).\n"
        "- Every column in the plan must be explicitly specified; do not rely on defaults.\n"
        "\n"
        "Causal-safety guidance (apply using ONLY the dataset summary):\n"
        "- Prefer transformations that preserve information and avoid altering the study population.\n"
        "- Avoid high-cardinality one-hot explosions; use max_categories defensively based on the summary.\n"
        "- Preserve missingness information when it is non-trivial (e.g., dummy_na or add_missing_indicator),\n"
        "  unless the summary strongly suggests missingness is negligible.\n"
        "- If a column is already numeric and well-behaved, prefer minimal transforms.\n"
        "\n"
        "If information is insufficient to safely choose an encoding, choose the most conservative option\n"
        "that keeps information without row drops and without inventing mappings.\n"
    )


def build_transform_plan_user_prompt_template() -> str:
    """
    Caller formats with:
      {encoding_catalog_text}, {missingness_catalog_text}, {columns_json}, {summary_json}, {schema_json}
    """
    return (
        "Create a TransformPlan for causal modeling using ONLY the information below.\n"
        "Make data-driven choices based on the dataset summary.\n"
        "\n"
        "Decision process requirements:\n"
        "- For each column, choose exactly one encoding spec from the Encoding catalog.\n"
        "- For each encoding, specify missingness handling ONLY using the allowed missingness fields for that encoding.\n"
        "- Do NOT drop rows.\n"
        "- Do NOT invent mappings or category index lists unless the summary provides explicit categories/indices.\n"
        "- If a categorical column has many unique values, set a conservative max_categories or choose an ordinal strategy\n"
        "  ONLY if the order is explicitly supported by the summary.\n"
        "- If a column has missingness, choose an explicit missingness strategy appropriate for that encoding.\n"
        "\n"
        "Encodings catalog (allowed encoding specs + fields):\n"
        "{encoding_catalog_text}\n"
        "\n"
        "Missingness catalog (allowed missingness specs + fields):\n"
        "{missingness_catalog_text}\n"
        "\n"
        "Columns (the ONLY allowed column names):\n"
        "{columns_json}\n"
        "\n"
        "Dataset summary (the ONLY data you may use to decide):\n"
        "{summary_json}\n"
        "\n"
        "Output schema (must match exactly; return ONLY JSON):\n"
        "{schema_json}\n"
    )


def build_transformed_protocol_system_prompt() -> str:
    return (
        "You are aligning a causal protocol specification to a transformed dataframe.\n"
        "\n"
        "Hard rules:\n"
        "- Return ONLY valid JSON that conforms EXACTLY to the provided output schema.\n"
        "- No prose, no markdown, no explanations.\n"
        "- Use ONLY column names present in df_after_columns.\n"
        "- Do NOT invent features that do not exist in df_after_columns.\n"
        "- Preserve semantic intent of the original protocol while matching the transformed feature names.\n"
    )


def build_transformed_protocol_user_prompt_template() -> str:
    """
    Caller formats:
      {protocol_json}, {df_after_columns_json}, {feature_map_json}, {schema_json}
    """
    return (
        "Generate a TransformedProtocolSpec aligned with the transformed dataframe for causal modeling.\n"
        "Use ONLY the provided inputs.\n"
        "\n"
        "Original protocol (semantic intent):\n"
        "{protocol_json}\n"
        "\n"
        "df_after_columns (the ONLY valid output feature names):\n"
        "{df_after_columns_json}\n"
        "\n"
        "feature_map (how original columns expanded/changed):\n"
        "{feature_map_json}\n"
        "\n"
        "Requirements:\n"
        "- Map protocol variables to the correct transformed columns.\n"
        "- If a single original column expanded into many features (e.g., one-hot), reference the correct set.\n"
        "- Do NOT reference any column not in df_after_columns.\n"
        "- Return ONLY JSON matching the schema.\n"
        "\n"
        "Output schema (must match exactly):\n"
        "{schema_json}\n"
    )