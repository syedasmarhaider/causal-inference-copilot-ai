from __future__ import annotations

from dataclasses import dataclass
import json
from typing import  ClassVar, List, Mapping, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_deps import (
    ConfirmTransformedProtocolDeps,
)
from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_prompts import (
    confirm_transformed_protocol_node_info,
    llm1_fail_system_prompt,
    llm2_warn_system_prompt,
    llm3_decision_system_prompt,
)
from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_state import (
    ConfirmTransformedProtocolPayloadModel,
    ConfirmTransformedProtocolState,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel


# -----------------------------------------------------------------------------
# Deterministic helpers (no heuristics)
# -----------------------------------------------------------------------------
def _last_user_text(messages_history: Optional[Sequence[ChatMessage]]) -> Optional[str]:
    if not messages_history:
        return None
    for m in reversed(messages_history):
        if getattr(m, "role", None) == "user":
            txt = (m.content or "").strip()
            return txt if txt else None
    return None


def _issues_pack_text(issues: Sequence[ValidationIssueModel]) -> str:
    # Stable, readable “issues pack” for prompts.
    # (Keeping it simple; no extra schema.)
    payload = [i.model_dump(mode="json") for i in issues]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _call_llm_text(
    *,
    llm: LLMService,
    model_name: str,
    messages_history: Optional[Sequence[ChatMessage]],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    last_10 = list(messages_history[-10:]) if messages_history else None
    cfg = LLMConfig(model=model_name, temperature=temperature)
    resp = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=cfg,
        history=last_10,
    )
    return resp.content.strip()


# -----------------------------------------------------------------------------
# Decision schema for LLM3
# -----------------------------------------------------------------------------
class _DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # None => still discussing / needs clarification
    user_accepted: Optional[bool] = None

    # Always something to show to user (either confirmation, rejection, or a clarifying question)
    user_message: str = Field(..., min_length=1)

    # When rejecting / requesting changes, capture operational instructions.
    # If empty/None, treat as "still discussing" (keeps workflow PENDING).
    improvement_instructions: Optional[str] = None


# -----------------------------------------------------------------------------
# Node
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfirmTransformedProtocolNode(Node):
    NAME: ClassVar[str] = ConfirmTransformedProtocolState.NAME

    llm: LLMService
    model_name: str

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return confirm_transformed_protocol_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        deps = ConfirmTransformedProtocolDeps.from_loaded(previous_state_dependencies)
        tp = deps.transform_protocol.payload

        clean_issues = tp.cleaned_dataset_validation_issues or []
        transform_issues = tp.transformation_issues or []
        all_issues: List[ValidationIssueModel] = list(clean_issues) + list(transform_issues)

        has_fail = any(i.severity == "FAIL" for i in all_issues)
        has_warn = (not has_fail) and any(i.severity == "WARN" for i in all_issues)

        # ------------------------------------------------------------
        # FAIL: blocking -> explain and abort
        # ------------------------------------------------------------
        if has_fail:
            system_prompt = llm1_fail_system_prompt()
            user_prompt = "Validation issues pack (JSON):\n" + _issues_pack_text(all_issues)

            msg = _call_llm_text(
                llm=self.llm,
                model_name=self.model_name,
                messages_history=None,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )

            return ConfirmTransformedProtocolState(
                ConfirmTransformedProtocolPayloadModel(
                    user_accepted=None,
                    user_message=msg,
                    error_message=msg,
                )
            )

        # ------------------------------------------------------------
        # WARN: 2-step handshake
        #
        # Step A (first time): produce explanation, ask user to decide (PENDING).
        # Step B (next time, after user reply): classify decision via LLM3.
        # ------------------------------------------------------------
        if has_warn:
            prev_payload: Optional[ConfirmTransformedProtocolPayloadModel] = None
            if isinstance(state, ConfirmTransformedProtocolState):
                prev_payload = state.payload

            last_user = _last_user_text(messages_history)

            # ---- Step B: we already asked, and now we have a user reply
            if prev_payload is not None and prev_payload.user_accepted is None and prev_payload.user_message and last_user:
                assistant_explanation = prev_payload.user_message

                system_prompt = llm3_decision_system_prompt()
                user_prompt = json.dumps(
                    {
                        "issues_pack": [i.model_dump(mode="json") for i in all_issues],
                        "assistant_explanation": assistant_explanation,
                        "user_reply": last_user,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )

                decision = self.llm.generate_json(
                    schema=_DecisionModel,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    config=LLMConfig(model=self.model_name, temperature=0.0),
                    history=list(messages_history[-10:]) if messages_history else None,
                    max_attempts=2,
                )

                # If model returns "false" but WITHOUT concrete change instructions,
                # treat as "still discussing" to avoid aborting on ambiguity.
                if decision.user_accepted is False:
                    instr = (decision.improvement_instructions or "").strip()
                    if not instr:
                        return ConfirmTransformedProtocolState(
                            ConfirmTransformedProtocolPayloadModel(
                                user_accepted=None,
                                user_message=decision.user_message,
                                error_message=None,
                            )
                        )

                    # Explicit rejection / change request
                    error_blob = json.dumps(
                        {
                            "issues_pack": [i.model_dump(mode="json") for i in all_issues],
                            "assistant_explanation": assistant_explanation,
                            "user_reply": last_user,
                            "improvement_instructions": instr,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    return ConfirmTransformedProtocolState(
                        ConfirmTransformedProtocolPayloadModel(
                            user_accepted=False,
                            user_message=decision.user_message,
                            error_message=error_blob,
                        )
                    )

                if decision.user_accepted is True:
                    return ConfirmTransformedProtocolState(
                        ConfirmTransformedProtocolPayloadModel(
                            user_accepted=True,
                            user_message=decision.user_message,
                            error_message=None,
                        )
                    )

                # decision.user_accepted is None => keep discussing
                return ConfirmTransformedProtocolState(
                    ConfirmTransformedProtocolPayloadModel(
                        user_accepted=None,
                        user_message=decision.user_message,
                        error_message=None,
                    )
                )

            # ---- Step A: produce the warning explanation and ask for a choice
            system_prompt = llm2_warn_system_prompt()
            user_prompt = "Validation issues pack (JSON):\n" + _issues_pack_text(all_issues)

            explanation = _call_llm_text(
                llm=self.llm,
                model_name=self.model_name,
                messages_history=messages_history,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
            )

            return ConfirmTransformedProtocolState(
                ConfirmTransformedProtocolPayloadModel(
                    user_accepted=None,
                    user_message=explanation,
                    error_message=None,
                )
            )

        # ------------------------------------------------------------
        # PASS: no issues -> proceed
        # ------------------------------------------------------------
        return ConfirmTransformedProtocolState(
            ConfirmTransformedProtocolPayloadModel(
                user_accepted=True,
                user_message="No validation issues found. Proceeding.",
                error_message=None,
            )
        )