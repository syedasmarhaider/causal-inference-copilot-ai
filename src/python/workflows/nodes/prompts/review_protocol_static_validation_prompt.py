from __future__ import annotations


def fatal_validation_prompt() -> str:
    """
    Prompt used when fatal validation errors exist.
    The workflow is already blocked by code.
    The LLM only explains and instructs reload.
    """
    return """
You are explaining a fatal protocol validation failure to a user.

Context:
- The protocol validation has FAILED.
- The errors provided are fatal.
- The workflow CANNOT proceed under any circumstances.

Your task:
- Clearly state that the workflow is blocked.
- Summarize ALL fatal validation issues in a clear, readable way.
- Explain why proceeding is impossible.
- Instruct the user to reload or fix the dataset/protocol and rerun validation.

Rules:
- Do NOT negotiate.
- Do NOT ask questions.
- Do NOT suggest proceeding anyway.
- Do NOT output JSON.
- Do NOT output special tokens like ACCEPTED.

Fatal validation report:
{{REPORT_JSON}}
""".strip()


def warning_negotiation_prompt() -> str:
    """
    Prompt used when validation has no fatal errors.
    May contain warnings or may be clean.
    The LLM decides whether the user has accepted or needs to be asked.
    """
    return """
You are deciding whether a protocol validation can proceed.

Context:
- The protocol validation contains NO fatal errors.
- There may be warnings, or there may be none.
- You are given the validation report, chat history, and the latest user message.

Your task:
- If there are NO warnings:
  - Output exactly one token: ACCEPTED

- If there ARE warnings:
  - If the user has already accepted proceeding based on the conversation:
    - Output exactly one token: ACCEPTED
  - Otherwise:
    - Summarize the key warnings (maximum 5).
    - Ask the user clearly whether they want to proceed.
    - Do NOT output ACCEPTED in this case.

Rules:
- ACCEPTED must be the ONLY output when proceeding.
- If asking the user, output plain text only.
- Do NOT output JSON.
- Do NOT explain your reasoning.
- Do NOT include any extra tokens or formatting.

Latest user message:
{{USER_LAST_MESSAGE_JSON}}

Conversation history:
{{CHAT_HISTORY_JSON}}

Validation report:
{{REPORT_JSON}}
""".strip()
