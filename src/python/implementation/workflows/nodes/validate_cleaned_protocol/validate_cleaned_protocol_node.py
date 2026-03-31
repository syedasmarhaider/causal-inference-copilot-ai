from __future__ import annotations

import json
from python.implementation.service.logging.default_logging import get_logger
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_deps import (
    ValidateCleanProtocolDeps,
)
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_prompts import (
    VALIDATE_CLEAN_PROTOCOL_PROMPT,
    validate_cleaned_protocol_get_info,
)
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import (
    ValidateCleanProtocolPayloadModel,
    ValidateCleanProtocolState,
)
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_utils import (
    ValidationIssue,
    validate_covariate_and_effect_modifier_constantness,
    validate_covariate_and_effect_modifier_high_cardinality_and_id_like,
    validate_covariate_and_effect_modifier_missingness,
    validate_covariate_and_effect_modifier_missingness_by_treatment,
    # ---- covariates + effect modifiers ----
    validate_covariate_and_effect_modifier_presence,
    validate_covariate_and_effect_modifier_type_risks,
    # ---- structural / protocol invariants ----
    validate_min_rows,
    validate_outcome,
    # ---- overlap / positivity ----
    validate_overlap_and_positivity,
    validate_protocol_role_columns_invariants,
    validate_treatment,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.utils.validation import ValidationIssueModel

log = get_logger(__name__)


@dataclass(frozen=True)
class ValidateCleanProtocolNode(Node):
    data_repo: DataRepo
    llm: LLMService

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
        previous_state_dependencies: Mapping[str, State],
        messages_history: Sequence[ChatMessage] | None
    ) -> State:
        last_4_messages = messages_history[-4:] if messages_history is not None else None
        try:
            deps = ValidateCleanProtocolDeps.from_loaded(previous_state_dependencies)
            causal_specs = deps.causal_spec
            clean_id = deps.dataset_id

            # ----------------------------
            # Load cleaned dataframe
            # ----------------------------
            df = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_id,
                limit=None,
            )
            
            all_issues: list[ValidationIssue] = []
            metrics: dict[str, Any] = {
                "n_rows_df": int(df.shape[0]),
                "n_cols_df": int(df.shape[1]),
                "df_columns_unique": bool(df.columns.is_unique),
                "df_columns_sample": list(map(str, df.columns[:200])),
            }

            # =========================================================================
            # 0) Protocol-only invariants (no df required)
            # =========================================================================
            issues = validate_protocol_role_columns_invariants(causal_specs)
            all_issues.extend(issues)
            metrics["protocol_role_invariants"] = {
                "n_issues": int(len(issues)),
                "n_fail": int(sum(1 for x in issues if x["severity"] == "FAIL")),
                "n_warn": int(sum(1 for x in issues if x["severity"] == "WARN")),
            }

            # =========================================================================
            # 1) Structural df-backed invariants
            # =========================================================================
            issues, m = validate_min_rows(df, min_rows_fail=20)
            all_issues.extend(issues)
            metrics["min_rows"] = m

            # =========================================================================
            # 2) Treatment validations (ProtocolSpec-native)
            # =========================================================================
            issues, m = validate_treatment(df=df, causal_spec=causal_specs)
            all_issues.extend(issues)
            metrics["treatment"] = m

            # =========================================================================
            # 3) Outcome validations (ProtocolSpec-native)
            # =========================================================================
            issues, m = validate_outcome(df=df, causal_spec=causal_specs)
            all_issues.extend(issues)
            metrics["outcome"] = m


            # =========================================================================
            # 4) Covariates / Effect modifiers (ProtocolSpec-native)
            #    IMPORTANT: keep require_covariates=False to match your prior behavior
            #    (warn when empty; don't hard-fail).
            # =========================================================================
            issues, m = validate_covariate_and_effect_modifier_presence(
                df=df,
                causal_spec=causal_specs,
                require_covariates=False,
            )
            all_issues.extend(issues)
            metrics["covariate_effect_modifier_presence"] = m

            issues, m = validate_covariate_and_effect_modifier_missingness(df=df, causal_spec=causal_specs)
            all_issues.extend(issues)
            metrics["covariate_effect_modifier_missingness"] = m

            issues, m = validate_covariate_and_effect_modifier_missingness_by_treatment(df=df, causal_spec=causal_specs)
            all_issues.extend(issues)
            metrics["covariate_effect_modifier_missingness_by_treatment"] = m

            issues, m = validate_covariate_and_effect_modifier_constantness(df=df, causal_spec=causal_specs)
            all_issues.extend(issues)
            metrics["covariate_effect_modifier_constantness"] = m

            issues, m = validate_covariate_and_effect_modifier_high_cardinality_and_id_like(df=df, causal_spec=causal_specs)
            all_issues.extend(issues)
            metrics["covariate_effect_modifier_cardinality_idlike"] = m

            issues, m = validate_covariate_and_effect_modifier_type_risks(df=df, causal_spec=causal_specs)
            all_issues.extend(issues)
            metrics["covariate_effect_modifier_type_risks"] = m

            # =========================================================================
            # 5) Overlap / positivity (advanced)
            #    - This uses your ProtocolSpec + cleaned df only.
            #    - If covariates are empty, we DO NOT fail here (require_covariates=False),
            #      but overlap diagnostics may be less meaningful.
            # =========================================================================
            try:
                issues, m = validate_overlap_and_positivity(
                    df=df,
                    causal_spec=causal_specs,
                    require_covariates=False,
                    use_effect_modifiers_univariate=True,
                    enable_propensity_proxy=True,
                )
                all_issues.extend(issues)
                metrics["overlap_positivity"] = m
            except Exception as e:
                all_issues.append(
                    {
                        "severity": "FAIL",
                        "message": "Overlap/positivity diagnostics failed to run.",
                        "evidence": {"error": repr(e)},
                        "fix_hint": "Inspect treatment typing and feature columns; overlap diagnostics require valid arm masks and non-pathological inputs.",
                    }
                )

            # ----------------------------
            # Normalize issues -> pydantic models
            # ----------------------------
            issue_models: list[ValidationIssueModel] = [
                ValidationIssueModel.model_validate(it) for it in all_issues
            ]
            has_fail = any(i.severity == "FAIL" for i in issue_models)

            msg = self._make_user_message(
                messages_history=last_4_messages,
                protocol_summary=self._protocol_summary(causal_specs),
                metrics=metrics,
                issues=[i.model_dump(mode="json") for i in issue_models],
                has_fail=has_fail,
            )

            payload = ValidateCleanProtocolPayloadModel(
                issues=issue_models,
                user_acceptance=msg.user_acceptance,
                validation_error="validation error occurs and protocol discussion is required as the protocol has some issues." if has_fail else None,
                user_message=msg.message_for_user,
            )
            return ValidateCleanProtocolState(payload=payload)

        except ValidationError as e:
            return self._abort(
                messages_history=last_4_messages,
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
            log.exception("Unexpected error in ValidateCleanProtocolNode", error=repr(e))
            return self._abort(
                messages_history=last_4_messages,
                validation_error=repr(e),
                issues=[
                    {
                        "severity": "FAIL",
                        "message": "Validation aborted due to an internal error.",
                        "evidence": {"error": repr(e)},
                        "fix_hint": "Inspect server logs and the validation node implementation.",
                    }
                ],
            )

    # =============================================================================
    # Internals
    # =============================================================================
    def _abort(
        self,
        *,
        messages_history: Sequence[ChatMessage] | None,
        validation_error: str,
        issues: list[dict[str, Any]],
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
            user_message=msg.message_for_user,
        )
        return ValidateCleanProtocolState(payload=payload)

    def _protocol_summary(self, causal_spec: CausalSpec) -> dict[str, Any]:
        # outcome cols summary
        ospec = getattr(causal_spec, "outcome_spec", None)
        out_cols: list[str] = []
        if ospec is not None:
            if getattr(ospec, "kind", None) == "duration":
                out_cols = [
                    str(getattr(ospec, "duration_column", "")),
                    str(getattr(ospec, "event_column", "")),
                ]
                out_cols = [c for c in out_cols if c and c.strip()]
            else:
                y = getattr(ospec, "column", None)
                if isinstance(y, str) and y.strip():
                    out_cols = [y]

        return {
            "experiment_type": getattr(causal_spec, "experiment_type", None),
            "treatment_col": getattr(getattr(causal_spec, "treatment_spec", None), "column", None),
            "treatment_kind": getattr(getattr(causal_spec, "treatment_spec", None), "kind", None),
            "outcome_kind": getattr(ospec, "kind", None) if ospec is not None else None,
            "outcome_cols": out_cols,
            "time_zero_type": getattr(causal_spec, "time_zero_type", None),
            "time_zero": getattr(causal_spec, "time_zero", None),
            "n_covariates": int(len(list(getattr(causal_spec, "covariates", []) or []))),
            "n_effect_modifiers": int(len(list(getattr(causal_spec, "effect_modifiers", []) or []))),
        }
       

    def _make_user_message(
        self,
        *,
        messages_history: Sequence[ChatMessage] | None,
        protocol_summary: dict[str, Any] | None,
        metrics: dict[str, Any],
        issues: list[dict[str, Any]],
        has_fail: bool,
    ) -> _UserAcceptanceModel:
            user_payload = { # pyright: ignore[reportUnknownVariableType]
                "protocol_summary": protocol_summary,
                "validation_metrics": metrics,
                "validation_issues": issues,
                "has_hard_fail": has_fail,
            }
            msg = self.llm.generate_json(
                schema=_UserAcceptanceModel,
                config=LLMConfig(model="basic", temperature=0.6),
                system_prompt=VALIDATE_CLEAN_PROTOCOL_PROMPT,
                user_prompt=json.dumps(user_payload, ensure_ascii=False),
                history=messages_history,
            )
            return msg

class _UserAcceptanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message_for_user: str
    user_acceptance: bool | None = None      
