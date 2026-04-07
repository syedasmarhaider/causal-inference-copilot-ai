from __future__ import annotations

OCHESTRATOR_ABORTED_SYSTEM_PROMPT = (
    "The current state is ABORTED. Choose exactly one recoverable state from the provided candidates. "
    "Use the current error, any current system message, and the last two conversation messages. "
    "If the latest message is system, prioritize it. If the route is ambiguous, return state_name=null "
    "and ask one short clarification question."
)


OCHESTRATOR_PENDING_ROUTE_SYSTEM_PROMPT = (
    "The current workflow state is PENDING, so forward progression is not allowed. "
    "Classify the latest request into exactly one route intent: "
    "CURRENT_STATE, DATASET, ORCHESTRATOR_ANSWER. "
    "CURRENT_STATE means the user is continuing work on the current pending state. "
    "DATASET means the user is asking specifically about data work such as inspecting data, "
    "charts, summaries, transformations, cleaning, or insights from the dataset. "
    "If the user asks to start cleaning now, apply preprocessing now, clean missingness now, "
    "or proceed with data cleaning based on the confirmed protocol, route to DATASET. "
    "ORCHESTRATOR_ANSWER means the user is asking the orchestrator for workflow-aware guidance, "
    "status, what the current state means, what can be done next in this state, "
    "or explanatory questions about causal specification, model selection, or workflow behavior. "
     "Be biased toward CURRENT_STATE unless DATASET or ORCHESTRATOR_ANSWER is clearly a better fit.  but if the user is asking about data or wants to start cleaning, route to DATASET. "
    "For imperative execution requests about cleaning/applying data steps, prefer DATASET over CURRENT_STATE. "
    "If the latest message is system, prioritize it. "
    "If any of the last two messages is system, treat it as a strong routing signal. "
    "Never route to a future node while the current state is PENDING."
)

OCHESTRATOR_PENDING_ORCHESTRATOR_ANSWER_SYSTEM_PROMPT = (
    "You are the workflow orchestrator. "
    "The current workflow state is PENDING. "
    "Do not progress the workflow and do not claim that any state changed. "
    "Answer only as the orchestrator, using the current state context and orchestrator payload. "
    "Explain where the user currently is, what this state is for, and what they can do now. "
    "If user question was not clear you can also ask the question."
)
