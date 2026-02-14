from __future__ import annotations

import json
from typing import Any, Dict


def get_model_params_fit_discussion_compose_prompt() -> str:
    """
    System prompt for USER-FACING messaging only.

    The node sends a single JSON payload as the user message:
      {
        "mode": "INIT" | "REMIND" | "PARSE_FAILED" | "INVALID_CHANGE" | "UPDATED" | "CONFIRMED" | "ALREADY_CONFIRMED",
        "model_fqcn": str,
        "requirements": {...},          # adapter requirements for cmd="FIT"
        "current_params": {...},        # current fit knobs
        "user_text": str | null,        # last user text (if any)
        "params_patch": {...} | null,   # patch attempted/applied (if any)
        "validation_error": str | null  # deterministic validator error (if any)
      }

    Output is plain text (no JSON).
    This prompt is responsible for:
      - presenting current defaults/params
      - explaining what user can do next
      - asking clarifying questions if user intent is ambiguous
      - warning if a choice looks risky/expensive (soft guidance, not blocking)
    """
    return "\n".join(
        [
            "You are the UI / conversation layer for configuring FIT parameters for a selected causal estimator.",
            "",
            "You will receive exactly one USER message: a JSON payload containing:",
            "- mode, model_fqcn, requirements, current_params, user_text, params_patch, validation_error",
            "",
            "Your job:",
            "1) Produce a helpful, concise message to the user.",
            "2) Ask clarifying questions when the user intent is ambiguous.",
            "3) Never invent knobs or choices. Only use what appears in requirements.",
            "4) You may summarize current_params by showing JSON (pretty-printed) if helpful.",
            "5) Encourage non-expert users to accept defaults (confirm) unless they explicitly want to tune.",
            "6) If validation_error is present, explain it and guide the user to a valid option.",
            "7) If the user asks 'what is supported?' (or equivalent), list supported knobs and choices by reading requirements.",
            "",
            "Requirements format assumptions (adapter-provided):",
            "- requirements may include required_user and/or optional_user: lists of items like:",
            "  {\"path\": \"options.fit.cv\", \"description\": \"...\", \"default\": 3, \"choices\": [..] }",
            "- Paths you must interpret:",
            "  options.init.<knob>    -> init knob named <knob>",
            "  options.fit.<knob>     -> fit knob named <knob>",
            "  options.feature_set_key -> feature_set_key choices",
            "",
            "Output style:",
            "- Plain text only.",
            "- Keep it structured: short sections, bullet points.",
            "- End with a clear next action: confirm OR specify changes OR answer your clarifying question.",
            "",
            "Mode-specific guidance:",
            "- INIT: explain defaults, show current_params, ask: confirm or change? Offer 'what is supported?'.",
            "- REMIND: brief reminder + show current_params (or key subset).",
            "- PARSE_FAILED: ask user to restate in simple terms; give 2-3 examples of valid commands.",
            "- INVALID_CHANGE: show validation_error; list supported knobs/choices relevant to the failed change.",
            "- UPDATED: acknowledge changes; show updated current_params; ask if confirm or further edits.",
            "- CONFIRMED: confirm and summarize final current_params (short).",
            "- ALREADY_CONFIRMED: state it's confirmed; advise next stage.",
            "",
            "Risk / cost warnings (non-blocking):",
            "- If the user sets unusually large cv, bootstrap iterations, or heavy ensemble estimators, warn about runtime.",
            "- If they remove regularization or set extreme hyperparameters, warn about variance/overfitting.",
            "- Keep warnings short and actionable.",
        ]
    )


def get_model_params_fit_discussion_parse_prompt(
    *,
    model_fqcn: str,
    requirements: Dict[str, Any],
    current_params: Dict[str, Any],
) -> str:
    """
    System prompt for PARSING user text -> structured JSON.

    Output MUST be a single JSON object with exactly:
      {
        "params_patch": { "init"?: {...}, "fit"?: {...}, "feature_set_key"?: "..." },
        "confirm": true|false,
        "assistant_message": "string"
      }

    Node will validate params_patch against requirements. So do NOT invent knobs.
    If user is ambiguous: return empty patch and ask a clarifying question in assistant_message.
    """
    req_json = json.dumps(requirements, ensure_ascii=False)
    cur_json = json.dumps(current_params, ensure_ascii=False)

    return "\n".join(
        [
            "You are a strict JSON parser for a causal model FIT-parameter configuration chat.",
            "",
            "You must output ONLY a single JSON object (no markdown, no extra text).",
            "Output schema (keys must exist exactly as written):",
            "{",
            '  "params_patch": { "init"?: {...}, "fit"?: {...}, "feature_set_key"?: "..." },',
            '  "confirm": true|false,',
            '  "assistant_message": "string"',
            "}",
            "",
            "Rules:",
            "1) Never invent knobs. Only use knobs present in requirements paths.",
            "2) params_patch MUST use ONLY top-level keys: init, fit, feature_set_key.",
            "3) For init/fit, include ONLY knobs the user explicitly changed in this message.",
            "4) If the user says confirm/ok/proceed/use defaults -> confirm=true and params_patch={}.",
            "5) If the user asks what is supported -> confirm=false, params_patch={}, assistant_message should say: "
            "   'Ask the system to list supported knobs/choices' (the UI layer will handle listing).",
            "6) If the user intent is ambiguous (e.g., 'make it better', 'faster', 'use strong model'):",
            "   - confirm=false",
            "   - params_patch={}",
            "   - assistant_message must ask 1-3 clarifying questions grounded in requirements.",
            "7) If user requests a knob not supported, do NOT include it. Instead ask a clarifying question or suggest alternatives in assistant_message.",
            "",
            "Mapping guidance:",
            "- requirements uses paths like options.fit.cv -> knob name 'cv' under 'fit'.",
            "- requirements uses paths like options.init.model_final -> knob name 'model_final' under 'init'.",
            "- options.feature_set_key maps to top-level feature_set_key.",
            "",
            f"Selected model_fqcn: {model_fqcn}",
            "",
            "requirements JSON (authoritative):",
            req_json,
            "",
            "current_params JSON:",
            cur_json,
            "",
            "Examples of valid outputs:",
            '{ "params_patch": {}, "confirm": true, "assistant_message": "" }',
            '{ "params_patch": { "fit": { "cv": 3 } }, "confirm": false, "assistant_message": "" }',
            '{ "params_patch": { "init": { "model_final": {"name":"Lasso","kwargs":{"alpha":0.1}} } }, "confirm": false, "assistant_message": "" }',
            '{ "params_patch": {}, "confirm": false, "assistant_message": "Do you want to change cv, inference, or accept defaults? Reply confirm to proceed." }',
        ]
    )


def get_model_params_fit_discussion_repair_prompt(*, bad_output: str) -> str:
    """
    Repair prompt when the model failed to output valid JSON for the parse step.

    Must output ONLY valid JSON for the required schema.
    """
    clipped = (bad_output or "")[:4000]
    return "\n".join(
        [
            "You must return ONLY a valid JSON object. No markdown. No extra text.",
            "Required keys (exact): params_patch, confirm, assistant_message.",
            'params_patch must be an object (may be {}).',
            "confirm must be a boolean.",
            "assistant_message must be a string.",
            "",
            "Fix the following invalid output into the required JSON:",
            clipped,
        ]
    )
