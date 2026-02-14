from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TypedDict, cast
from uuid import UUID

from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.model_params_fit_discussion_system_prompt import (
    get_model_params_fit_discussion_compose_prompt,
    get_model_params_fit_discussion_parse_prompt,
    get_model_params_fit_discussion_repair_prompt,
)
from python.workflows.state.control_state import ACTION
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.model_state import ModelParamsFitState, ModelState
from python.workflows.tools.inference.causal_inference import CausalInference
from python.workflows.tools.inference.causal_inference_factory import CausalInferenceFactory


_PROMPTS: Dict[str, Callable[..., str]] = {
    "compose": get_model_params_fit_discussion_compose_prompt,
    "parse": get_model_params_fit_discussion_parse_prompt,
    "repair": get_model_params_fit_discussion_repair_prompt,
}


class _FitStateView(TypedDict):
    # required view (Pylance-safe)
    params: Dict[str, Any]
    confirmed: bool


@dataclass(frozen=True)
class _ParseResult:
    params_patch: Dict[str, Any]
    confirm: bool
    assistant_message: str


def make_model_params_fit_discussion_node(
    *,
    llm: LLMService,
    model_name: str,
    causal_factory: CausalInferenceFactory,
) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=state,
            llm=llm,
            model_name=model_name,
            causal_factory=causal_factory,
        )

    return node


def _run(
    *,
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    llm: LLMService,
    model_name: str,
    causal_factory: CausalInferenceFactory,
) -> ConversationState:
    ir: Any | None = state.get("inference_ready")
    if not ir:
        msg: str = "Missing inference_ready state."
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg=msg)

    model_state: ModelState | None = cast(Optional[ModelState], state.get("model"))
    if model_state is None:
        msg: str = "ModelState missing. Run MODEL_SELECTION first."
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg=msg)

    model_fqcn: Optional[str] = model_state.get("selected_model_fqcn")
    if not model_fqcn:
        msg: str = "No model selected yet. Select a model first."
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_pending(state=state, action=cast(ACTION, "NEEDS_INPUT"), msg=msg)

    inference: CausalInference | None = causal_factory.resolve(model_fqcn)
    if inference is None:
        msg: str = f"Selected model is not supported by any adapter: reselect the model {model_fqcn}"
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg=msg)

    req:Dict[str, Any]  = inference.get_input_requirements(cmd="FIT", ir=ir)

    if req.get("status") == "UNSUPPORTED":
        msg: str = str(
            req.get("reason")
            or (
                f"Unsupported: {model_fqcn} does not support FIT stage. "
                "Select a different model, or change inference requirements by discussing the protocol."
            )
        )
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg=msg)

    def _view(
        *,
        mode: str,
        user_text: Optional[str],
        params_patch: Optional[Dict[str, Any]],
        validation_error: Optional[str],
        current_params: Dict[str, Any],
    ) -> str:
        return _llm_compose_message(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            mode=mode,
            model_fqcn=model_fqcn,
            requirements=req,
            current_params=current_params,
            user_text=user_text,
            params_patch=params_patch,
            validation_error=validation_error,
        )

    def _pending(msg: str) -> ConversationState:
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_pending(state=state, action=cast(ACTION, "NEEDS_INPUT"), msg=msg)

    def _done(msg: str) -> ConversationState:
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_done(state=state, action=cast(ACTION, "NONE"), msg=msg)

    # --- fit_state load/init (Pylance-safe via _FitStateView) ---
    fit_state_any: Any = model_state.get("model_params_fit")
    is_new_fit_state: bool = not isinstance(fit_state_any, dict)

    if is_new_fit_state:
        fit_state_any = {"params": {}, "confirmed": False}
        model_state["model_params_fit"] = cast(ModelParamsFitState, fit_state_any)

    fit_state: ModelParamsFitState = cast(ModelParamsFitState, fit_state_any)
    fit_view: _FitStateView = _fit_state_view(fit_state)

    if is_new_fit_state:
        _apply_defaults_in_place(fit_view["params"], req)
        msg: str = _view(
            mode="INIT",
            user_text=None,
            params_patch=None,
            validation_error=None,
            current_params=fit_view["params"],
        )
        return _pending(msg)

    if fit_view["confirmed"] is True:
        msg: str = _view(
            mode="ALREADY_CONFIRMED",
            user_text=None,
            params_patch=None,
            validation_error=None,
            current_params=fit_view["params"],
        )
        return _done(msg)

    user_text: str = (ConversationStateHelpers.last_human_text(state) or "").strip()
    if not user_text:
        msg: str = _view(
            mode="REMIND",
            user_text=None,
            params_patch=None,
            validation_error=None,
            current_params=fit_view["params"],
        )
        return _pending(msg)

    parse: Optional[_ParseResult] = _llm_parse_user_message(
        llm=llm,
        model_name=model_name,
        temperature=0.0,
        model_fqcn=model_fqcn,
        requirements=req,
        current_params=fit_view["params"],
        user_text=user_text,
    )
    if parse is None:
        msg: str = _view(
            mode="PARSE_FAILED",
            user_text=user_text,
            params_patch=None,
            validation_error="Could not parse your message into a valid JSON patch.",
            current_params=fit_view["params"],
        )
        return _pending(msg)

    ok: bool
    reason: Optional[str]
    ok, reason = _validate_patch_against_requirements(parse.params_patch, req)
    if not ok:
        msg: str = _view(
            mode="INVALID_CHANGE",
            user_text=user_text,
            params_patch=parse.params_patch,
            validation_error=reason or "Unsupported change.",
            current_params=fit_view["params"],
        )
        return _pending(msg)

    _deep_merge_in_place(fit_view["params"], parse.params_patch)
    _apply_defaults_in_place(fit_view["params"], req)
    model_state["model_params_fit"] = fit_state

    if parse.confirm is True:
        fit_view["confirmed"] = True
        model_state["model_params_fit"] = fit_state

        msg: str = _view(
            mode="CONFIRMED",
            user_text=user_text,
            params_patch=parse.params_patch,
            validation_error=None,
            current_params=fit_view["params"],
        )
        return _done(msg)

    msg: str = _view(
        mode="UPDATED",
        user_text=user_text,
        params_patch=parse.params_patch,
        validation_error=None,
        current_params=fit_view["params"],
    )
    return _pending(msg)


# -------------------------
# FitState "required view" (fixes Pylance NotRequiredAccess)
# -------------------------
def _fit_state_view(fit_state: ModelParamsFitState) -> _FitStateView:
    params_any: Any = fit_state.get("params")
    if not isinstance(params_any, dict):
        fit_state["params"] = {}  # materialize
    confirmed_any: Any = fit_state.get("confirmed")
    if not isinstance(confirmed_any, bool):
        fit_state["confirmed"] = False  # materialize

    return cast(_FitStateView, fit_state)


# -------------------------
# LLM helpers (LLMConfig + generate)
# -------------------------
def _llm_compose_message(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    mode: str,
    model_fqcn: str,
    requirements: Dict[str, Any],
    current_params: Dict[str, Any],
    user_text: Optional[str],
    params_patch: Optional[Dict[str, Any]],
    validation_error: Optional[str],
) -> str:
    sys_prompt: str = _PROMPTS["compose"]()
    payload: Dict[str, Any] = _prompt_payload(
        mode=mode,
        model_fqcn=model_fqcn,
        requirements=requirements,
        current_params=current_params,
        user_text=user_text,
        params_patch=params_patch,
        validation_error=validation_error,
    )

    try:
        out: str = _llm_call_prompt(
            llm=llm,
            model_name=model_name,
            temperature=temperature,
            system_prompt=sys_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            empty_err="Compose prompt returned empty output.",
        )
        return out
    except Exception:
        return "Please confirm the defaults or specify what you want to change."


def _llm_parse_user_message(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    model_fqcn: str,
    requirements: Dict[str, Any],
    current_params: Dict[str, Any],
    user_text: str,
) -> Optional[_ParseResult]:
    sys_prompt: str = _PROMPTS["parse"](
        model_fqcn=model_fqcn,
        requirements=requirements,
        current_params=current_params,
    )

    try:
        raw: str = _llm_call_prompt(
            llm=llm,
            model_name=model_name,
            temperature=temperature,
            system_prompt=sys_prompt,
            user_prompt=user_text,
            empty_err="Parse prompt returned empty output.",
        )
    except Exception:
        return None

    parsed: Optional[Dict[str, Any]] = _parse_json_object(raw)
    if parsed is None:
        repair_prompt: str =  _PROMPTS["repair"](bad_output=str(raw or ""))
        try:
            raw2: str = _llm_call_prompt(
                llm=llm,
                model_name=model_name,
                temperature=temperature,
                system_prompt=repair_prompt,
                user_prompt="",
                empty_err="Repair prompt returned empty output.",
            )
        except Exception:
            return None

        parsed = _parse_json_object(raw2)
        if parsed is None:
            return None

    params_patch_any: Any = parsed.get("params_patch")
    confirm_any: Any = parsed.get("confirm")
    assistant_message_any: Any = parsed.get("assistant_message")

    params_patch: Dict[str, Any] = params_patch_any if isinstance(params_patch_any, dict) else {} # pyright: ignore[reportUnknownVariableType]
    confirm: bool = confirm_any if isinstance(confirm_any, bool) else False
    assistant_message: str = assistant_message_any if isinstance(assistant_message_any, str) else ""

    return _ParseResult(
        params_patch=params_patch,
        confirm=confirm,
        assistant_message=assistant_message,
    )


def _llm_call_prompt(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    empty_err: str,
) -> str:
    cfg: LLMConfig = LLMConfig(model=model_name, temperature=temperature)
    raw: str = _llm_text(llm, config=cfg, system_prompt=system_prompt, user_prompt=user_prompt)
    out: str = (raw or "").strip()
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
    resp_any: Any = llm.generate(
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        history=None,
    )
    content: str = str(getattr(resp_any, "content", "") or "").strip()
    return content


def _prompt_payload(
    *,
    mode: str,
    model_fqcn: str,
    requirements: Dict[str, Any],
    current_params: Dict[str, Any],
    user_text: Optional[str],
    params_patch: Optional[Dict[str, Any]],
    validation_error: Optional[str],
) -> Dict[str, Any]:
    return {
        "mode": mode,
        "model_fqcn": model_fqcn,
        "requirements": requirements,
        "current_params": current_params,
        "user_text": user_text,
        "params_patch": params_patch,
        "validation_error": validation_error,
    }


def _parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    s: str = str(raw or "").strip()
    if not s:
        return None
    try:
        out: Any = json.loads(s)
        return out
    except Exception:
        return None


# -------------------------
# Validation / defaults / merge
# -------------------------
def _validate_patch_against_requirements(patch: Dict[str, Any], req: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if not isinstance(patch, dict):
        return False, "params_patch must be an object."

    allowed_init_keys: set[str]
    allowed_fit_keys: set[str]
    feature_choices: Optional[list[Any]]
    knob_choice_map: Dict[str, Optional[list[Any]]]
    allowed_init_keys, allowed_fit_keys, feature_choices, knob_choice_map = _extract_allowed(req)

    allowed_top: set[str] = {"init", "fit", "feature_set_key"}
    for k in patch.keys():
        if k not in allowed_top:
            return False, f"Unsupported top-level key '{k}'. Allowed: {sorted(allowed_top)}"

    init_any: Any = patch.get("init")
    if init_any is not None:
        if not isinstance(init_any, dict):
            return False, "init must be an object."
        for k, v in init_any.items():
            if k not in allowed_init_keys:
                return False, f"Unsupported init knob '{k}'. Supported: {sorted(allowed_init_keys)}"
            ok, reason = _validate_value_against_choices(
                path=f"options.init.{k}",
                value=v,
                choice_map=knob_choice_map,
            )
            if not ok:
                return False, reason

    fit_any: Any = patch.get("fit")
    if fit_any is not None:
        if not isinstance(fit_any, dict):
            return False, "fit must be an object."
        for k, v in fit_any.items():
            if k not in allowed_fit_keys:
                return False, f"Unsupported fit knob '{k}'. Supported: {sorted(allowed_fit_keys)}"
            ok, reason = _validate_value_against_choices(
                path=f"options.fit.{k}",
                value=v,
                choice_map=knob_choice_map,
            )
            if not ok:
                return False, reason

    if "feature_set_key" in patch:
        v: Any = patch.get("feature_set_key")
        if feature_choices is not None and v not in feature_choices:
            return False, f"feature_set_key must be one of {feature_choices}."

    return True, None


def _validate_value_against_choices(
    *,
    path: str,
    value: Any,
    choice_map: Dict[str, Optional[list[Any]]],
) -> Tuple[bool, Optional[str]]:
    choices: Optional[list[Any]] = choice_map.get(path)
    if not choices:
        return True, None

    # Special-case: estimator spec may be {"name": "...", "kwargs": {...}}
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        name: str = cast(str, value["name"])
        if name not in choices:
            return False, f"'{path}': unsupported estimator '{name}'. Supported: {choices}"
        return True, None

    if value not in choices:
        return False, f"'{path}': value must be one of {choices}."
    return True, None


def _apply_defaults_in_place(params: Dict[str, Any], req: Dict[str, Any]) -> None:
    items: list[Any] = []
    for key in ("required_user", "optional_user"):
        v_any: Any = req.get(key)
        if isinstance(v_any, list):
            items.extend(v_any)

    for it in items:
        if not isinstance(it, dict):
            continue
        path_any: Any = it.get("path")
        if not isinstance(path_any, str):
            continue
        if "default" not in it:
            continue
        default_any: Any = it.get("default")
        _set_default_by_path(params, path_any, default_any)


def _set_default_by_path(params: Dict[str, Any], path: str, default: Any) -> None:
    if not path.startswith("options."):
        return

    tail: str = path[len("options.") :]
    parts: list[str] = tail.split(".")
    if not parts:
        return

    cur: Any = params
    for p in parts[:-1]:
        if not isinstance(cur, dict):
            return
        if p not in cur or not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]

    leaf: str = parts[-1]
    if isinstance(cur, dict) and leaf not in cur:
        cur[leaf] = default


def _deep_merge_in_place(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            dst_any: Any = dst.get(k)
            if isinstance(dst_any, dict):
                _deep_merge_in_place(cast(Dict[str, Any], dst_any), v)
            else:
                dst[k] = v
        else:
            dst[k] = v
