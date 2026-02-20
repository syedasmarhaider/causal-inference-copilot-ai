from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, cast
from uuid import UUID

import pandas as pd
from pydantic import ValidationError

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidateCompiledInferenceNode(Node):
    """
    Loads the cleaned dataset produced by COMPILE_INFERENCE and runs pre-transform validations
    against the compiled ProtocolSpec.

    - On FAIL issues: returns VALIDATE_COMPILED_INFERENCE with FAIL issues (pipeline ABORT).
    - On only WARN or no issues: returns DONE with an actionable user message.
    - On unexpected exceptions: returns ABORT with a synthetic FAIL + validation_error.

    Notes:
    - This node does not transform/encode. It validates raw, post-filtered data.
    - It attempts to use an LLM (via ToolFactory) to produce a user-facing summary.
      If unavailable, it falls back to a deterministic summarizer.
    """

    data_repo: DataRepo

    # TODO: Fix late r
    NAME: ClassVar[str] = 'FIX NAME LATER'

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return (
            "Validate compiled inference inputs (clean dataset + compiled protocol) prior to transform/encoding. "
            "Produces FAIL/WARN issues and a user-facing summary."
        )

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: Optional[ToolFactory],
        previous_state_dependencies: Mapping[str, State],
        user_message: Optional[str],
        router_message: Optional[str],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        try:
            deps = ValidateCompiledInferenceDeps.from_loaded(previous_state_dependencies)

            # ----------------------------
            # Guardrails: upstream state sanity
            # ----------------------------
            proto = deps.compile_protocol.protocol
            if proto is None or deps.compile_protocol.compile_error:
                return self._abort(
                    tool_factory=tool_factory,
                    messages_history=messages_history,
                    validation_error="CompileProtocolState missing protocol or has compile_error.",
                    issues=[
                        {
                            "severity": "FAIL",
                            "message": "Protocol is not available for validation.",
                            "evidence": {
                                "compile_error": deps.compile_protocol.compile_error,
                                "compile_issues": deps.compile_protocol.compile_issues,
                            },
                            "fix_hint": "Fix protocol compilation first (COMPILE_PROTOCOL).",
                        }
                    ],
                )

            if deps.compile_inference.cleaning_error is not None:
                return self._abort(
                    tool_factory=tool_factory,
                    messages_history=messages_history,
                    validation_error="CompileInferenceState indicates cleaning_error.",
                    issues=[
                        {
                            "severity": "FAIL",
                            "message": "Clean dataset is not available because dataset cleaning failed.",
                            "evidence": {"cleaning_error": deps.compile_inference.cleaning_error},
                            "fix_hint": "Fix dataset cleaning in COMPILE_INFERENCE before running validation.",
                        }
                    ],
                )

            clean_id = deps.compile_inference.clean_dataset_id
            if clean_id is None:
                return self._abort(
                    tool_factory=tool_factory,
                    messages_history=messages_history,
                    validation_error="CompileInferenceState missing clean_dataset_id.",
                    issues=[
                        {
                            "severity": "FAIL",
                            "message": "Clean dataset id is missing; cannot load data for validation.",
                            "evidence": {"clean_dataset_id": None},
                            "fix_hint": "Ensure COMPILE_INFERENCE saves a cleaned dataset_id.",
                        }
                    ],
                )

            # ----------------------------
            # Load cleaned dataframe
            # ----------------------------
            df = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_id,
                limit=None,
            )
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"DataRepo.get_csv_data returned {type(df).__name__}, expected DataFrame.")

            # ----------------------------
            # Build modeling view and validate
            # ----------------------------
            key_cols = extract_key_columns(proto)
            view = select_modeling_view(df, key_cols, include_time_zero=True, copy=True)

            all_issues: List[Dict[str, Any]] = []
            metrics: Dict[str, Any] = {
                "clean_dataset_id": str(clean_id),
                "n_rows": int(view.shape[0]),
                "n_cols": int(view.shape[1]),
                "columns": list(map(str, view.columns)),
            }

            # 2) Structural invariants
            issues, m = validate_min_rows(view, min_rows_fail=1)
            all_issues.extend(issues)
            metrics["min_rows"] = m

            all_issues.extend(validate_column_list_invariants(key_cols))

            issues, m = validate_time_zero_semantics(view, proto, key_cols)
            all_issues.extend(issues)
            metrics["time_zero"] = m

            # 3) Treatment validations
            issues, m = validate_treatment_missingness(df=view, protocol=proto)
            all_issues.extend(issues)
            metrics["treatment_missingness"] = m

            issues, m = validate_treatment_domain_integrity(df=view, protocol=proto)
            all_issues.extend(issues)
            metrics["treatment_domain"] = m

            issues, m = validate_treatment_variation(df=view, protocol=proto)
            all_issues.extend(issues)
            metrics["treatment_variation"] = m

            # 4) Outcome validations
            issues, m = validate_outcome_missingness(df=view, protocol=proto)
            all_issues.extend(issues)
            metrics["outcome_missingness"] = m

            issues, m = validate_outcome_domain_integrity(df=view, protocol=proto)
            all_issues.extend(issues)
            metrics["outcome_domain"] = m

            issues, m = validate_outcome_variation(df=view, protocol=proto)
            all_issues.extend(issues)
            metrics["outcome_variation"] = m

            # 5) W/X validations
            issues, m = validate_WX_presence(view, key_cols, require_W=False)
            all_issues.extend(issues)
            metrics["wx_presence"] = m

            W_prof = profile_feature_block(view, key_cols.W_cols, label="W")
            X_prof = profile_feature_block(view, key_cols.X_cols, label="X")
            WX_prof = profile_feature_block(view, [*key_cols.W_cols, *key_cols.X_cols], label="WX")

            metrics["W_profile"] = W_prof.to_dict()
            metrics["X_profile"] = X_prof.to_dict()
            metrics["WX_profile"] = WX_prof.to_dict()

            for prof in (W_prof, X_prof, WX_prof):
                issues, m = validate_feature_missingness(prof)
                all_issues.extend(issues)
                metrics[f"{prof.label}_missingness"] = m

                issues, m = validate_feature_constantness(prof)
                all_issues.extend(issues)
                metrics[f"{prof.label}_constantness"] = m

                issues, m = validate_feature_cardinality(prof)
                all_issues.extend(issues)
                metrics[f"{prof.label}_cardinality"] = m

                issues, m = validate_feature_type_risks(prof)
                all_issues.extend(issues)
                metrics[f"{prof.label}_type_risks"] = m

            # 6) Overlap / positivity diagnostics
            try:
                arm_masks = compute_arm_masks(view, proto, key_cols)
                metrics["arm_masks"] = arm_masks.to_dict()

                feat_cols = [*key_cols.W_cols, *key_cols.X_cols]
                issues, m = overlap_support_check(view, feat_cols=feat_cols, arm_masks=arm_masks)
                all_issues.extend(issues)
                metrics["overlap_support"] = m

                # Optional: propensity proxy (binary only)
                issues, m = overlap_propensity_proxy(
                    view,
                    W_cols=key_cols.W_cols,
                    treatment_col=key_cols.treatment_col,
                    arm_masks=arm_masks,
                )
                all_issues.extend(issues)
                metrics["propensity_proxy"] = m
            except Exception as e:
                # Overlap diagnostics are non-fatal if the earlier basics already produced FAIL/WARN.
                all_issues.append(
                    {
                        "severity": "WARN",
                        "message": "Overlap diagnostics failed to run (non-fatal).",
                        "evidence": {"error": repr(e)},
                        "fix_hint": "Inspect treatment column typing and feature columns; overlap diagnostics require valid arm masks.",
                    }
                )

            # ----------------------------
            # Normalize into Pydantic issues
            # ----------------------------
            issue_models: List[InferenceReadyValidationIssueModel] = []
            for it in all_issues:
                issue_models.append(InferenceReadyValidationIssueModel.model_validate(it))

            has_fail = any(i.severity == "FAIL" for i in issue_models)

            # User-facing message: LLM if possible, else deterministic
            msg = self._make_user_message(
                tool_factory=tool_factory,
                messages_history=messages_history,
                protocol_summary=self._protocol_summary(proto, key_cols),
                metrics=metrics,
                issues=[i.model_dump(mode="json") for i in issue_models],
                has_fail=has_fail,
            )

            payload = InferenceReadyValidationPayloadModel(
                issues=issue_models,
                validation_error=None,
                user_message=msg,
            )
            return InferenceReadyValidationState(payload=payload)

        except ValidationError as e:
            # Pydantic parse/validation failure inside our own models.
            return self._abort(
                tool_factory=tool_factory,
                messages_history=messages_history,
                validation_error=f"Pydantic validation error: {e.errors()}",
                issues=[
                    {
                        "severity": "FAIL",
                        "message": "Validation node produced an invalid payload (internal error).",
                        "evidence": {"errors": e.errors()},
                        "fix_hint": "Fix the validation node / issue schema mismatch.",
                    }
                ],
            )
        except Exception as e:
            return self._abort(
                tool_factory=tool_factory,
                messages_history=messages_history,
                validation_error=repr(e),
                issues=[
                    {
                        "severity": "FAIL",
                        "message": "Server error while validating compiled inference inputs.",
                        "evidence": {"error": repr(e)},
                        "fix_hint": "Check server logs and dataset/protocol consistency.",
                    }
                ],
            )

    # =============================================================================
    # Internals
    # =============================================================================

    def _abort(
        self,
        *,
        tool_factory: Optional[ToolFactory],
        messages_history: Optional[Sequence[ChatMessage]],
        validation_error: str,
        issues: List[Dict[str, Any]],
    ) -> InferenceReadyValidationState:
        # Ensure abort always triggers ABORTED in your State.status property:
        # it only checks FAIL issues, so we ALWAYS include at least one FAIL issue.
        safe_issues = issues or [
            {
                "severity": "FAIL",
                "message": "Validation aborted due to an internal error.",
                "evidence": {"validation_error": validation_error},
                "fix_hint": "Inspect server logs.",
            }
        ]

        msg = self._make_user_message(
            tool_factory=tool_factory,
            messages_history=messages_history,
            protocol_summary=None,
            metrics={"validation_error": validation_error},
            issues=safe_issues,
            has_fail=True,
        )

        issue_models = [InferenceReadyValidationIssueModel.model_validate(x) for x in safe_issues]

        payload = InferenceReadyValidationPayloadModel(
            issues=issue_models,
            validation_error=validation_error,
            user_message=msg,
        )
        return InferenceReadyValidationState(payload=payload)

    def _protocol_summary(self, proto: Any, key_cols: Any) -> Dict[str, Any]:
        # Keep this JSON-friendly (LLM prompt + debugging).
        return {
            "treatment_col": key_cols.treatment_col,
            "outcome_cols": list(key_cols.outcome_cols),
            "time_zero_type": getattr(proto, "time_zero_type", None),
            "time_zero_col": key_cols.time_zero_col,
            "n_W": len(list(key_cols.W_cols)),
            "n_X": len(list(key_cols.X_cols)),
        }

    def _make_user_message(
        self,
        *,
        tool_factory: Optional[ToolFactory],
        messages_history: Optional[Sequence[ChatMessage]],
        protocol_summary: Optional[Dict[str, Any]],
        metrics: Dict[str, Any],
        issues: List[Dict[str, Any]],
        has_fail: bool,
    ) -> str:
        # Try LLM first; fallback to deterministic summary.
        try:
            llm_text = self._try_llm_summary(
                tool_factory=tool_factory,
                messages_history=messages_history,
                protocol_summary=protocol_summary,
                metrics=metrics,
                issues=issues,
                has_fail=has_fail,
            )
            if isinstance(llm_text, str) and llm_text.strip():
                return llm_text.strip()
        except Exception:
            pass

        return self._fallback_summary(issues=issues, has_fail=has_fail)

    def _try_llm_summary(
        self,
        *,
        tool_factory: Optional[ToolFactory],
        messages_history: Optional[Sequence[ChatMessage]],
        protocol_summary: Optional[Dict[str, Any]],
        metrics: Dict[str, Any],
        issues: List[Dict[str, Any]],
        has_fail: bool,
    ) -> Optional[str]:
        if tool_factory is None:
            return None

        llm = self._resolve_llm(tool_factory)
        if llm is None:
            return None

        system = (
            "You are a rigorous assistant for a causal inference pipeline.\n"
            "Task: produce a short, actionable validation report for the user.\n"
            "Rules:\n"
            "- If there are FAIL issues: clearly say the pipeline must stop and list the top blockers.\n"
            "- If only WARN: say validation passed with warnings and list what will happen next.\n"
            "- Keep it compact: 6–12 bullet points.\n"
            "- Mention treatment/outcome/W/X/time_zero when relevant.\n"
            "- Use concrete fix hints (copy the provided fix_hint when useful).\n"
        )

        payload = {
            "has_fail": bool(has_fail),
            "protocol_summary": protocol_summary,
            "metrics": metrics,
            "issues": issues,
        }

        user = (
            "Generate the user-facing message for these validation results.\n"
            "Return plain text (no JSON).\n\n"
            f"INPUT:\n{json.dumps(payload, ensure_ascii=False)[:60000]}"
        )

        msgs: List[ChatMessage] = []
        msgs.append(cast(ChatMessage, {"role": "system", "content": system}))
        msgs.append(cast(ChatMessage, {"role": "user", "content": user}))

        # Minimal context carry-over (optional)
        if messages_history:
            # Keep a small tail only (don’t blow up tokens)
            tail = list(messages_history)[-4:]
            for m in tail:
                msgs.append(m)

        # Duck-typed call: support multiple llm method names.
        for fn_name in ("generate", "chat", "complete", "__call__"):
            fn = getattr(llm, fn_name, None)
            if callable(fn):
                out = fn(msgs)
                if isinstance(out, str):
                    return out
                # Some APIs return dict-like objects
                if isinstance(out, dict) and isinstance(out.get("content"), str):
                    return cast(str, out["content"])
                if hasattr(out, "content") and isinstance(getattr(out, "content"), str):
                    return cast(str, getattr(out, "content"))

        return None

    def _resolve_llm(self, tool_factory: ToolFactory) -> Optional[Any]:
        # Best-effort resolver without hard-coding your concrete ToolFactory shape.
        for attr in (
            "llm",
            "llm_service",
            "get_llm",
            "get_llm_service",
            "create_llm",
            "create_llm_service",
        ):
            v = getattr(tool_factory, attr, None)
            if callable(v):
                try:
                    return v()
                except Exception:
                    continue
            if v is not None:
                return v
        return None

    def _fallback_summary(self, *, issues: List[Dict[str, Any]], has_fail: bool) -> str:
        # Deterministic, always available.
        fails = [x for x in issues if str(x.get("severity")) == "FAIL"]
        warns = [x for x in issues if str(x.get("severity")) == "WARN"]

        lines: List[str] = []
        if has_fail:
            lines.append("Validation failed. Fix the following blockers before continuing:")
            top = fails[:8]
        else:
            lines.append("Validation passed.")
            if warns:
                lines.append("Warnings detected (you can continue, but results may be unstable):")
            top = warns[:8]

        for i, it in enumerate(top, start=1):
            msg = str(it.get("message", "")).strip()
            hint = it.get("fix_hint")
            if isinstance(hint, str) and hint.strip():
                lines.append(f"{i}. {msg}  →  {hint.strip()}")
            else:
                lines.append(f"{i}. {msg}")

        if not has_fail:
            lines.append("Next: run TRANSFORM (encoding/typing) and then the post-transform validation stage.")

        return "\n".join(lines)
