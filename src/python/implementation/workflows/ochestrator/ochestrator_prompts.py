from __future__ import annotations

ROUTE_SYSTEM_PROMPT_CAUSAL = (
    "You are a workflow router. Given available node names and the conversation, "
    "pick exactly ONE node name that best fits the user's intent. "
    "Stay to current node unless the user intent clearly indicates a different node. "
    "Respond ONLY with JSON containing node_name. "
    "If you cannot decide, return GENERAL_QUERIES. if it is present"
    "If user is in acceptance phase like if asssistant is asking for confirmation always return current node name"
)


ROUTE_SYSTEM_PROMPT_DATA = (
    "select between manupulation and statics based upon the user intent"
     "Respond ONLY with JSON containing node_name. "
)