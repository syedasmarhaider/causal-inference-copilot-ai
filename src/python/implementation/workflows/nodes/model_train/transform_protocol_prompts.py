from __future__ import annotations

def get_transform_protocol_node_info() -> str:
    return (
        "Transform protocol takes the compiled protocol and the cleaned dataset, and produces: \n"
        "1) a TransformPlan that specifies how to transform each column (e.g., encoding + missingness handling), and \n"
        "2) a ProtocolSpec that maps the original protocol variables to the transformed features. \n"
        "3) Run the validation on transform protocol and produce user friendly message and issues if there is any problem with the transformed dataset or the spec."
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
        "- Only create plans for covariates and effect modifiers.\n"
        "- Do NOT drop rows (row-dropping is disallowed).\n"
        "- Do NOT invent categories, mappings, or index sets that are not supported by the provided catalogs.\n"
        "- Do NOT apply any 'cleaning' beyond whitespace stripping (assume the executor only strips whitespace).\n"
        "- Every column in the plan must be explicitly specified; do not rely on defaults.\n"
        "\n"
        "Causal-safety guidance (apply using ONLY the dataset summary):\n"
        "- Prefer transformations that preserve information and avoid altering the study population.\n" 
        "- Preserve missingness information when it is non-trivial\n"
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
        "Encodings catalog:\n"
        "Protocol Specs:\n"
        "{protocol_json}\n"
        "\n"
        "Dataset summary:\n"
        "{summary_json}\n"
        "\n"
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
    Caller formats with:
      {protocol_json}, {df_after_columns_json}
    """
    return (
        "Align the original protocol specification to the transformed dataframe using ONLY the information below.\n"
        "For each protocol variable (treatment, outcome, adjustment), map it to the corresponding column(s) in the transformed dataframe.\n"
        "\n"
        "Protocol Specs:\n"
        "{protocol_json}\n"
        "\n"
        "Transformed dataframe columns:\n"
        "{df_after_columns_json}\n"
        "\n"
    )
    
def  build_hard_validation_system_prompt() -> str:
    return (
        "Describe the validaton issues to the clinicians in a clear and concise manner. For each issue, explain which column is affected, what the issue is, and why it matters for causal inference. Use non-technical language that clinicians can understand, and provide actionable recommendations for how to address each issue. If there are multiple issues, present them in a bulleted list format for easy reading."
        "Ask to discuss the protocol again and to resolve the issues. Or select different causal queiton."
        "Focus on fail part but include shed subtle light on warn part if there is any warn issue as well."
        "if Not validation issues then simply say tried to tranfrom data but cannot"
        "{validation_issues_json}"
    ) 
    
    
def build_user_friendly_message_for_transform_protocol_system_prompt() -> str:
    return (
        "You are an assistant and you need to do two things:\n"
        "1. If users suggest and proceed with the transformation plan and you think it would work, then return the user all plan summary by starting exactly with {causal_transformation_summary}: and then give the summary of the plan in a clear and concise manner.\n"
        "2. Otherwise, if the user wants to continue discussion and refining the transformation, you can continue the discussion.\n"
        "Important note: as the user is a clinician, try to explain things in that way, not as a data scientist. Use simple language and avoid technical jargon. Always relate the transformation and its implications to clinical practice and patient outcomes to make it more relevant and understandable for the clinician."
    )