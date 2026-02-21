from __future__ import annotations

def get_transform_protocol_node_info() -> str:
    return (
        "Transforms the protocol spec to match the transformed dataset. and make inference ready for user validation\n"
        "This includes updating column references and encoding decisions based on the feature map.\n"
        "The node runs after the dataset has been transformed and a feature map is available.\n"
        "The output is a TransformedProtocolSpec that can be used for downstream modeling.\n"
    )

def build_encoding_plan_system_prompt() -> str:
    return (
        "You are a data-encoding planner for causal inference pipelines.\n"
        "Your job: choose an encoding specification for each column to prepare it for modeling.\n"
        "You MUST output ONLY valid JSON that conforms exactly to the provided schema.\n"
        "Do not include explanations outside JSON.\n"
        "\n"
        "Hard rules:\n"
        "- Only reference columns from the provided column list.\n"
        "- At most one decision per column.\n"
        "- Do NOT invent new encoding types.\n"
        "- Prefer encodings that preserve causal interpretability.\n"
        "- Do not leak target variables into features; treat Y/T columns specially.\n"
    )


def build_encoding_plan_user_prompt_template() -> str:
    """
    Template only. NO parsing/serialization here.
    Caller must format with:
      {encoding_catalog_text}, {columns_json}, {protocol_json}, {roles_json}, {summary_json}
    """
    return (
        "You will create an encoding plan for the dataset.\n"
        "Return ONLY JSON matching the schema.\n"
        "\n"
        "Supported encodings (whitelist):\n"
        "{encoding_catalog_text}\n"
        "\n"
        "Columns (choose ONLY from this list):\n"
        "{columns_json}\n"
        "\n"
        "Protocol (JSON):\n"
        "{protocol_json}\n"
        "\n"
        "Roles (raw columns):\n"
        "{roles_json}\n"
        "\n"
        "Dataset summary (JSON):\n"
        "{summary_json}\n"
        "\n"
        "Rules for decisions:\n"
        "- Provide exactly one decision per column you decide to encode.\n"
        "- Never invent columns.\n"
        "- For categorical columns: use one-hot only when cardinality is small; otherwise choose an encoding that avoids high dimensionality.\n"
        "- For boolean columns: keep boolean/binary when possible.\n"
        "- For numeric columns: keep numeric; consider standardization only if the downstream model benefits.\n"
        "- For datetime columns: do NOT pass raw timestamps; prefer extracting meaningful parts (year/month/day) only if justified.\n"
        "- Be conservative: prefer simpler encodings.\n"
    )
    
def build_transformed_protocol_system_prompt() -> str:
    return (
        "You output ONLY valid JSON that matches the provided schema.\n"
        "Use ONLY column names provided in df_after_columns.\n"
        "Do not invent columns.\n"
        "Do not include explanations outside JSON.\n"
    )


def build_transformed_protocol_user_prompt_template() -> str:
    """
    Template only. Caller formats:
      {protocol_json}, {df_after_columns_json}, {feature_map_json}
    """
    return (
        "Generate a TransformedProtocolSpec for the transformed dataframe.\n"
        "Return ONLY JSON.\n"
        "\n"
        "Original protocol (JSON):\n"
        "{protocol_json}\n"
        "\n"
        "df_after_columns (JSON list):\n"
        "{df_after_columns_json}\n"
        "\n"
        "feature_map (JSON):\n"
        "{feature_map_json}\n"
    )    