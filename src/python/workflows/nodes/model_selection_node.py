from __future__ import annotations

import json
import logging
import re
from typing import Any, List, cast
from uuid import UUID

from python.domain.service.llm_service import  LLMConfig, LLMService
from python.workflows.nodes.prompts.model_selection import (
    PromptInputs,
    get_econml_allowed_estimators,
    get_model_selection_prompt_1,
    get_model_selection_prompt_2,
    get_model_selection_prompt_3,
)
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.model_selection_state import ModelSelectionState
from python.workflows.state.validate_protocol_state import ProtocolStaticValidationState, ProtocolValidationReport

log = logging.getLogger(__name__)


def make_model_selection_node(*, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(user_id=user_id, conversation_id=conversation_id, state=state, llm=llm, model_name=model_name)

    return node


def _run(
    *,
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    llm: LLMService,
    model_name: str,
) -> ConversationState:
    ms: ModelSelectionState = ModelSelectionState()
    state["model_selection"] = ms

    allowed_list = list(get_econml_allowed_estimators())
    allowed_set = set(allowed_list)
    ms["allowed_estimators"] = allowed_list
    ms["allowed_estimators_map"] = {fqcn: True for fqcn in allowed_list}

    dataset = state.get("dataset")

    # TODO: filter data summary here
    dataset_summary = dataset.get("summary")
    if dataset_summary is None:
        return _abort(state, ms, "Dataset summary is missing. Reload dataset is required.")

    inference_obj = (
        state.get("inference_ready_state")
        or state.get("inference_ready")
        or state.get("inference")
        or state.get("inference_ready_state_summary")
    )
    if inference_obj is None:
        return _abort(state, ms, "InferenceReadyState is missing from state. Run inference-ready stage is required.")

    protocol_obj = state.get("protocol_state") or state.get("protocol") or state.get("protocol_discussion")
    if protocol_obj is None:
        return _abort(state, ms, "Protocol state is missing from state. Run protocol compilation is required.")

    validation_state: ProtocolStaticValidationState | None = state.get("protocol_static_validation")
    if validation_state is None:
        return _abort(state, ms, "ProtocolStaticValidationState is missing from state. Run protocol validation stage is required.")
    
    report : ProtocolValidationReport | None =  validation_state.get("report")
    if report is None:
        return _abort(state, ms, "Protocol validation report is missing from state. Run protocol validation stage is required.")
    
    validation_issues: str = "\n".join([
        f"- {issue.get('description', 'No description')} (severity: {issue.get('severity', 'UNKNOWN')})"
        for issue in report.get("issues", [])
    ])
    
    inference_str = _normalize_to_text(inference_obj)
    dataset_str = _normalize_to_text(dataset_summary)
    protocol_str = _normalize_to_text(protocol_obj)
    

    prompt_inputs_base = PromptInputs(
        inference_ready_state_summary=inference_str,
        dataset_summary=dataset_str,
        validation_notes=validation_issues,
        protocol_state=protocol_str,
    )

    # -------------------------
    # LLM #1: Prompt 1 (draft) - retry once on failure/empty
    # -------------------------
    try:
        prompt_1 = get_model_selection_prompt_1(prompt_inputs_base)
        draft_text = _llm_call_prompt_retry(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            prompt=prompt_1,
            empty_err="LLM#1 returned empty draft output",
            max_attempts=2,
            log_label="MODEL_SELECTION:LLM#1",
        )
        ms["draft_text"] = draft_text
    except Exception as e:
        log.exception("MODEL_SELECTION: LLM#1 failed")
        return _abort(state, ms, f"Model selection (draft) failed: {e}")

    # -------------------------
    # LLM #2: Prompt 2 (final JSON) - parse/validate; repair once if needed
    # -------------------------
    try:
        prompt_inputs_2 = PromptInputs(
            inference_ready_state_summary=inference_str,
            dataset_summary=dataset_str,
            protocol_state=protocol_str,
            validation_notes=validation_issues,
            paste_from_previous_step=draft_text,
        )
        prompt_2 = get_model_selection_prompt_2(prompt_inputs_2)

        final_json_raw = _llm_call_prompt_retry(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            prompt=prompt_2,
            empty_err="LLM#2 returned empty JSON output",
            max_attempts=2,  # retry on transport/empty
            log_label="MODEL_SELECTION:LLM#2",
        )
        ms["final_json_raw"] = final_json_raw

        # Parse + validate; if fails once, ask for a repair JSON-only output and parse again
        parsed = None
        last_err: Exception | None = None

        for attempt in range(2):
            try:
                parsed = _parse_json_strictish(ms["final_json_raw"])
                ms["final_json"] = parsed

                selected_top3 = _extract_and_validate_top3(parsed, allowed=allowed_set)
                ms["selected_top3"] = selected_top3
                ms["top3_validated"] = True
                ms["top3_invalid"] = []

                ms["selection_notes"] = cast(List[str], parsed.get("selection_notes") or [])
                ms["rejected"] = cast(List[dict[str, Any]], parsed.get("rejected") or [])
                ms["unknowns"] = cast(List[str], parsed.get("unknowns") or [])
                break
            except Exception as e:
                last_err = e
                ms["top3_validated"] = False
                # On first failure, do one “repair” re-ask
                if attempt == 0:
                    repair_prompt = _build_prompt2_json_repair_prompt(
                        original_prompt=prompt_2,
                        bad_output=ms["final_json_raw"],
                        error=str(e),
                    )
                    repaired = _llm_call_prompt_retry(
                        llm=llm,
                        model_name=model_name,
                        temperature=0.0,
                        prompt=repair_prompt,
                        empty_err="LLM#2 repair returned empty JSON output",
                        max_attempts=1,  # single shot; we already had retries above
                        log_label="MODEL_SELECTION:LLM#2_REPAIR",
                    )
                    ms["final_json_raw"] = repaired
                    continue
                raise

        if parsed is None and last_err is not None:
            raise last_err

    except Exception as e:
        log.exception("MODEL_SELECTION: LLM#2 failed")
        return _abort(state, ms, f"Model selection (finalize) failed: {e}")

    # -------------------------
    # LLM #3: Prompt 3 (rationale)
    # -------------------------
    try:
        prompt_inputs_3 = PromptInputs(
            inference_ready_state_summary=inference_str,
            dataset_summary=dataset_str,
            protocol_state=protocol_str,
            validation_notes=validation_issues,
            final_selection_json=json.dumps(ms["final_json"], ensure_ascii=False, indent=2), # pyright: ignore[reportTypedDictNotRequiredAccess]
        )
        prompt_3 = get_model_selection_prompt_3(prompt_inputs_3)
        rationale_text = _llm_call_prompt_retry(
            llm=llm,
            model_name=model_name,
            temperature=0.5,
            prompt=prompt_3,
            empty_err="LLM#3 returned empty rationale output",
            max_attempts=2,
            log_label="MODEL_SELECTION:LLM#3",
        )
        ms["rationale_text"] = rationale_text
    except Exception as e:
        log.exception("MODEL_SELECTION: LLM#3 failed")
        return _abort(state, ms, f"Model selection (rationale) failed: {e}")

    ConversationStateHelpers.append_ai_message(state=state, content=rationale_text)
    return ConversationStateHelpers.set_done(state=state, action="NEEDS_INPUT", msg=rationale_text)


def _abort(state: ConversationState, ms: ModelSelectionState, msg: str) -> ConversationState:
    ms.setdefault("errors", []).append(msg)
    ms["top3_validated"] = False
    ConversationStateHelpers.append_ai_message(state=state, content=msg)
    return ConversationStateHelpers.set_abort(state=state, action="NONE", msg=msg)


def _normalize_to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        return str(obj)


def _parse_json_strictish(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Empty JSON string")

    # Strip common code fences if model violates the instruction
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)

    # Try direct
    try:
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object at top-level")
        return cast(dict[str, Any], parsed)
    except Exception:
        pass

    # Extract outermost {...}
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object boundaries in model output")

    candidate = s[start : end + 1]

    # Minor cleanup: remove trailing commas before } or ]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    parsed2 = json.loads(candidate)
    if not isinstance(parsed2, dict):
        raise ValueError("Expected a JSON object after extraction")
    return cast(dict[str, Any], parsed2)


def _extract_and_validate_top3(parsed: dict[str, Any], *, allowed: set[str]) -> List[str]:
    v = parsed.get("selected_top3")
    if not isinstance(v, list):
        raise ValueError("final_json.selected_top3 is missing or not a list")

    selected = [str(x).strip() for x in v if str(x).strip()] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    invalid = [x for x in selected if x not in allowed]
    selected = [x for x in selected if x in allowed]

    # Must be exactly 3 unique
    uniq: List[str] = []
    for x in selected:
        if x not in uniq:
            uniq.append(x)

    if len(uniq) != 3:
        raise ValueError(f"Expected exactly 3 valid estimators, got {len(uniq)}: {uniq}. Invalid: {invalid}")

    return uniq


def _build_prompt2_json_repair_prompt(*, original_prompt: str, bad_output: str, error: str) -> str:
    # Keep it brutally strict: JSON only, same schema, no prose.
    return (
        "Your previous output for Prompt 2 was invalid.\n"
        f"ERROR: {error}\n"
        "\n"
        "You MUST output ONLY a valid JSON object matching the required schema.\n"
        "No markdown. No code fences. No commentary.\n"
        "\n"
        "=== ORIGINAL PROMPT 2 ===\n"
        + original_prompt
        + "\n\n"
        "=== YOUR PREVIOUS (INVALID) OUTPUT ===\n"
        + bad_output
        + "\n\n"
        "NOW OUTPUT ONLY THE CORRECT JSON OBJECT:\n"
    )


def _llm_call_prompt_retry(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    prompt: str,
    empty_err: str,
    max_attempts: int,
    log_label: str,
) -> str:
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            out = _llm_call_prompt(
                llm=llm,
                model_name=model_name,
                temperature=temperature,
                prompt=prompt,
                empty_err=empty_err,
            )
            return out
        except Exception as e:
            last = e
            log.warning("%s attempt %d/%d failed: %s", log_label, attempt, max_attempts, e)
    assert last is not None
    raise last


def _llm_call_prompt(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    prompt: str,
    empty_err: str,
) -> str:
    cfg = LLMConfig(model=model_name, temperature=temperature)
    raw = _llm_text(llm, config=cfg, system_prompt="you are causal inference copilot", user_prompt=prompt)
    out = (raw or "").strip()
    if not out:
        raise ValueError(empty_err)
    return out


def _llm_text(
    llm: LLMService,
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
        resp = llm.generate( 
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=None,
        ).content
        return  resp
