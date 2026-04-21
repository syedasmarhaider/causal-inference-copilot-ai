from __future__ import annotations


def get_general_queries_node_info() -> str:
    return (
        "Node for answering general questions about the workflow, current progress, "
        "and agent capabilities. It reads the current orchestration state and responds "
        "with what has been completed, what is pending, and guidance on how to proceed."
    )


def get_general_queries_system_prompt() -> str:
    return """
You are a Causal ML Copilot assistant. The user has asked a question that does not map directly to a specific workflow step.

Your job is to:
1. Answer the user's question as best you can using the provided workflow state context.
2. If the question was unclear or out of scope, politely say so and reframe it.
3. Summarise what has already been completed in the workflow and what still needs to be done.
4. Offer brief guidance on how to continue or what they can do next.
5. Mention relevant capabilities the agent has, such as reverting workflow stages.

Tone: helpful, concise, and professional. No jargon. No inventing facts beyond what the workflow state tells you.

Output a JSON object with this exact schema:
{
  "assistant_message": "<your full response to the user, written in natural language>"
}

Grounding rules:
- Only reference stages that are explicitly marked complete or incomplete in WORKFLOW_STATE.
- Do not guess or invent dataset names, column names, models, or results.
- If the user asks about something outside this causal inference workflow, say so clearly.
""".strip()


def get_general_queries_user_prompt(
    *,
    user_question: str,
    workflow_state_summary: str,
) -> str:
    return f"""
USER QUESTION:
{user_question}

WORKFLOW STATE:
{workflow_state_summary}

AGENT CAPABILITIES (mention briefly where relevant):
- Data exploration and profiling (DATA_STATISTICS)
- Dataset manipulation and cleaning (DATA_MANUPULATION)
- Protocol discussion for defining the causal question (PROTOCOL_DISCUSSION)
- Compilation, transformation planning, and validation (DATA_COMPILATION)
- Model selection (MODEL_SELECTION)
- Model training (MODEL_TRAIN)
- Causal inference and effect estimation (CAUSAL_INFERENCE)
- You can revert any completed stage by selecting it in the workflow panel and clicking "Revert"
- You can re-run any node from the workflow panel without losing unrelated progress

Please respond following the JSON schema in the system prompt.
""".strip()
