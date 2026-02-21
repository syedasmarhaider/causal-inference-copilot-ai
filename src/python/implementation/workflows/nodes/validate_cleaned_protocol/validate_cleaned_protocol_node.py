from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from pydantic import ValidationError

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory  # kept for Node.run signature

from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_prompts import (
    system_prompt_validate_cleaned_protocol,
    validate_cleaned_protocol_get_info,
)
    
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_deps import (
    ValidateCleanProtocolDeps,
)
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import (
    ValidateCleanProtocolPayloadModel,
    ValidateCleanProtocolState,
)

from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_utils import (  # noqa: E501
    ValidationIssue,
    compute_arm_masks,
    extract_key_columns,
    overlap_propensity_proxy,
    overlap_support_check,
    profile_feature_block,
    select_modeling_view,
    validate_column_list_invariants,
    validate_feature_cardinality,
    validate_feature_constantness,
    validate_feature_missingness,
    validate_feature_type_risks,
    validate_min_rows,
    validate_outcome_domain_integrity,
    validate_outcome_missingness,
    validate_outcome_variation,
    validate_time_zero_semantics,
    validate_treatment_domain_integrity,
    validate_treatment_missingness,
    validate_treatment_variation,
    validate_WX_presence,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidateCleanProtocolNode(Node):
    """
    Pre-transform validation over the *cleaned, filtered* dataset and compiled ProtocolSpec.

    Output:
      - Returns InferenceReadyValidationState with payload. State.status will be:
          DONE    if no FAIL issues
          ABORTED if any FAIL issues
      - On server/internal errors: returns ABORTED with a synthetic FAIL issue + validation_error.
    """

    data_repo: DataRepo
    llm: LLMService
    model_name: str

    NAME: ClassVar[str] = ValidateCleanProtocolState.NAME

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return validate_cleaned_protocol_get_info()


    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,  
        previous_state_dependencies: Mapping[str, Any],
        user_message: Optional[str],
        router_message: Optional[str],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        try:
            deps = ValidateCleanProtocolDeps.from_loaded(previous_state_dependencies)

            # ----------------------------
            # Guardrails: upstream sanity
            # ----------------------------
            proto = deps.compile_protocol.payload.protocol
            assert proto is not None, "CleanProtocolState must have a compiled protocol for validation."
            clean_id = deps.clean_protocol.payload.clean_dataset_id
            assert clean_id is not None, "CleanProtocolState must have a clean_dataset_id for validation."
            # ----------------------------
            # Load cleaned dataframe
            # ----------------------------
            df = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_id,
                limit=None,
            )
            
            # ----------------------------
            # Build modeling view and validate
            # ----------------------------
            key_cols = extract_key_columns(proto)
            view = select_modeling_view(df, key_cols, include_time_zero=True, copy=True)

            all_issues: List[ValidationIssue] = []
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

                issues, m = overlap_propensity_proxy(
                    view,
                    W_cols=key_cols.W_cols,
                    treatment_col=key_cols.treatment_col,
                    arm_masks=arm_masks,
                )
                all_issues.extend(issues)
                metrics["propensity_proxy"] = m
            except Exception as e:
                all_issues.append(
                    {
                        "severity": "FAIL",
                        "message": "Overlap diagnostics failed to run.",
                        "evidence": {"error": repr(e)},
                        "fix_hint": "Inspect treatment column typing and feature columns; overlap diagnostics require valid arm masks.",
                    }
                )

            # ----------------------------
            # Normalize issues -> pydantic models
            # ----------------------------
            issue_models: List[ValidationIssueModel] = [
                ValidationIssueModel.model_validate(it) for it in all_issues
            ]
            has_fail = any(i.severity == "FAIL" for i in issue_models)

            msg = self._make_user_message(
                messages_history=messages_history,
                protocol_summary=self._protocol_summary(proto, key_cols),
                metrics=metrics,
                issues=[i.model_dump(mode="json") for i in issue_models],
                has_fail=has_fail,
            )

            payload = ValidateCleanProtocolPayloadModel(
                issues=issue_models,
                validation_error=None,
                user_message=msg,
            )
            return ValidateCleanProtocolState(payload=payload)

        except ValidationError as e:
            return self._abort(
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
        messages_history: Optional[Sequence[ChatMessage]],
        validation_error: str,
        issues: List[Dict[str, Any]],
    ) -> ValidateCleanProtocolState:
        safe_issues = issues or [
            {
                "severity": "FAIL",
                "message": "Validation aborted due to an internal error.",
                "evidence": {"validation_error": validation_error},
                "fix_hint": "Inspect server logs.",
            }
        ]

        msg = self._make_user_message(
            messages_history=messages_history,
            protocol_summary=None,
            metrics={"validation_error": validation_error},
            issues=safe_issues,
            has_fail=True,
        )

        issue_models = [ValidationIssueModel.model_validate(x) for x in safe_issues]
        payload = ValidateCleanProtocolPayloadModel(
            issues=issue_models,
            validation_error=validation_error,
            user_message=msg,
        )
        return ValidateCleanProtocolState(payload=payload)

    def _protocol_summary(self, proto: Any, key_cols: Any) -> Dict[str, Any]:
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
        messages_history: Optional[Sequence[ChatMessage]],
        protocol_summary: Optional[Dict[str, Any]],
        metrics: Dict[str, Any],
        issues: List[Dict[str, Any]],
        has_fail: bool,
    ) -> str:
        # LLM first; deterministic fallback.
        try:
            txt = self._try_llm_summary(
                messages_history=messages_history,
                protocol_summary=protocol_summary,
                metrics=metrics,
                issues=issues,
                has_fail=has_fail,
            )
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        except Exception:
            pass
        return self._fallback_summary(issues=issues, has_fail=has_fail)

    def _try_llm_summary(
        self,
        *,
        messages_history: Optional[Sequence[ChatMessage]],
        protocol_summary: Optional[Dict[str, Any]],
        metrics: Dict[str, Any],
        issues: List[Dict[str, Any]],
        has_fail: bool,
    ) -> Optional[str]:
        system = system_prompt_validate_cleaned_protocol()
        history_only_last_4_messages = messages_history[-4:] if messages_history else None
        payload = { # pyright: ignore[reportUnknownVariableType]
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
        
        config = LLMConfig(model=self.model_name, temperature=0.7)
        resp = self.llm.generate(
            system_prompt=system,
            user_prompt=user,
            config=config,
            history=history_only_last_4_messages,
        )
        
        return resp.content

    def _fallback_summary(self, *, issues: List[Dict[str, Any]], has_fail: bool) -> str:
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
            lines.append("Next: run TRANSFORM (encoding/typing) and then post-transform validation.")

        return "\n".join(lines)
