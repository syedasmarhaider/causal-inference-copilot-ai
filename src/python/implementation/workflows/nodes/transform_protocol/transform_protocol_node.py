from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import  ClassVar, List, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.transform_protocol.transform_protcol_plan import TransformPlanModel, validate_plan_against_df_columns
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_deps import TransformProtocolDeps


from python.implementation.workflows.nodes.transform_protocol.transform_protocol_encoding import apply_encoding_plan
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_prompts import build_hard_validation_system_prompt, build_transform_plan_system_prompt, build_transform_plan_user_prompt_template, build_transformed_protocol_user_prompt_template, get_transform_protocol_node_info
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import (
    TransformedProtocolSpec,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import TransformProtocolPayloadModel, TransformProtocolState
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_validation import validate_binary_and_one_hot_invariants, validate_constant_or_near_constant_controls, validate_dimensionality_caps, validate_encoding_postconditions, validate_id_like_features_in_controls, validate_input_columns_exist_and_are_unambiguous, validate_model_inputs_are_numeric_dtypes, validate_treatment_outcome_domains_by_kind
from python.implementation.workflows.tools.data.data_profiling_tool import (
    DatasetProfilingStateTool,
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel

@dataclass(frozen=True)
class TransformProtocolNode(Node):
    NAME: ClassVar[str] = TransformProtocolState.NAME

    llm: LLMService
    data_repo: DataRepo
    model_name: str
    max_attempts: int = 3
    # knobs for deterministic steps
    profiling_max_categories: int = 50
    profiling_sample_distinct: int = 50
    fail_fast_apply: bool = False
    fail_fast_validate: bool = False
    validation_policy: Optional["TransformValidationPolicy"] = None  # optional

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_transform_protocol_node_info()

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
        # ------------------------------------------------------------------
        # Dependencies
        # ------------------------------------------------------------------
        data_profiling_tool = cast(DatasetProfilingStateTool, tool_factory.get_tool("DATA_PROFILING_TOOL"))                
        deps = TransformProtocolDeps.from_loaded(previous_state_dependencies)
        clean_state= deps.clean_protocol
        compile_state = deps.compile_protocol
        validate_clean_protcol = deps.validate_cleaned_protocol
        clean_dataset_validation_issues = validate_clean_protcol.payload.issues
        clean_dataset_id = clean_state.payload.clean_dataset_id
        assert clean_dataset_id is not None, "CleanProtocolState.payload.clean_dataset_id is required for TransformProtocolNode"
        protocol = compile_state.payload.protocol
        assert protocol is not None, "CompileProtocolState.payload.protocol is required for TransformProtocolNode"


        # ------------------------------------------------------------------
        # Load dataset once (no need to reload every attempt)
        # ------------------------------------------------------------------
        try:
            df_clean: pd.DataFrame = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_dataset_id,
            )
        except Exception as e:  # noqa: BLE001
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    transformation_issues=clean_dataset_validation_issues + [
                        ValidationIssueModel(
                            severity="FAIL",
                            message=f"Failed to load the cleaned dataset: {e}",
                        )],
                    user_message="Failed to read the cleaned dataset. Please check if the previous steps completed successfully and try again.",
                )
            )

        # ------------------------------------------------------------------
        # Profile once (no need to reprofile every attempt)
        # ------------------------------------------------------------------
        try:
            clean_dataset_summary: DatasetSummaryModel = data_profiling_tool.extract_dataset_summary(
                df_clean,
                max_categories=self.profiling_max_categories,
                sample_distinct=self.profiling_sample_distinct,
                compute_quantiles=True,
                strict=True,
            )
        except Exception as e:  # noqa: BLE001
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    transformation_issues=clean_dataset_validation_issues + [
                        ValidationIssueModel(
                            severity="FAIL",
                            message=f"Failed to profile the cleaned dataset: {e}",
                        )],
                    user_message="Failed to profile the cleaned dataset. Please check if the previous steps completed successfully and try again.",
                )
            )
            
        # ------------------------------------------------------------------
        # Main logic: transform + validate in a loop with error propagation
        # ------------------------------------------------------------------
        transformed_spec, transformation_issues, final_transformed_spec, df_transformed = transform_and_validate_protocol_spec(
            llm=self.llm,
            protocol=protocol,
            df_after=df_clean,
            dataset_summary=clean_dataset_summary,
            max_attempts=self.max_attempts,
        )
        
        if transformed_spec is None or final_transformed_spec is None or (transformation_issues and any(i.severity == "FAIL" for i in transformation_issues)):
            if len(transformation_issues) == 0:
                raise ValueError("transform_and_validate_protocol_spec must return transformation_issues (possibly empty list), but got None")
            message = get_message_for_hard_validation_issue(
                llm=self.llm,
                issue=transformation_issues
            )
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    user_message=message,
                    transformation_issues = transformation_issues,
                )
            )
        
        new_transformed_dataset_id = uuid4()
        self.data_repo.save_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=new_transformed_dataset_id,
            df=df_transformed,
        )
        
        return TransformProtocolState(
            TransformProtocolPayloadModel(
                transform_protocol_plan=transformed_spec,
                transformed_dataset_id=new_transformed_dataset_id,
                transformed_spec=final_transformed_spec,
                cleaned_dataset_id=clean_dataset_id,
                cleaned_dataset_summary=clean_dataset_summary,
                cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                user_message="Data transformation successful. The dataset has been transformed according to the protocol and is ready for the next steps.",
            )
        )
            


def get_message_for_hard_validation_issue(llm: LLMService, issue: List[ValidationIssueModel]) -> str:
    llm_config = LLMConfig(temperature=1.0)
    return llm.generate(
        system_prompt="You are an assistant for generating user-friendly error messages",
        user_prompt=build_hard_validation_system_prompt().format(
            validation_issues_json=json.dumps([i.model_dump(mode="json") for i in issue], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        config=llm_config,
        history=None,
    ).content   
    




def transform_and_validate_protocol_spec(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    df_after: pd.DataFrame,
    dataset_summary: DatasetSummaryModel,
    max_attempts: int,
) -> Tuple[Optional[TransformedProtocolSpec], List[ValidationIssueModel], Optional[TransformedProtocolSpec], pd.DataFrame]:
    message_to_pass_in_case_of_error: List[ChatMessage] = []
    validation_issues: List[ValidationIssueModel] = []
    
    for _ in range(1, max_attempts + 1):
        try:
            plan = get_transform_encoding_plan(
                llm=llm,
                protocol=protocol,
                dataset_summary=dataset_summary,
                history=message_to_pass_in_case_of_error,
            )
            
            issues = validate_plan_against_df_columns(
                plan=plan,
                df_columns=df_after.columns.tolist(),
                require_full_coverage=True,
            )
            
            if len(issues) > 0:
                validation_issues.extend(issues)
                message_to_pass_in_case_of_error.extend(_repair_messages_from_issues(attempt=_, stage="PLAN_VALIDATION", issues=issues))
                continue
            
            df_transformed, issues = apply_plan(
                df=df_after,
                plans=plan,
            )
            
            if df_transformed is None:
                validation_issues.extend(issues)
                continue
            
            validation_issues.extend(issues)
            
            transformed_specs = protocol_spec_to_transformed_spec(
                llm=llm,
                protocol=protocol,
                transformed_df=df_transformed,
                history=message_to_pass_in_case_of_error,
            )
            
            validation_issues = run_transform_validations(
                df_after=df_transformed,
                spec=transformed_specs,
            )
            
            if any(_is_fail(x) for x in validation_issues):
                message_to_pass_in_case_of_error.extend(_repair_messages_from_issues(attempt=_, stage="VALIDATION", issues=validation_issues))
                continue
            
            return transformed_specs, validation_issues, transformed_specs, df_transformed
        except Exception as e:  # noqa: BLE001
            message_to_pass_in_case_of_error.append(
                ChatMessage(
                    role="system",
                    content=f"Error during transformed protocol spec generation: {e}. Please fix the JSON output to match the TransformedProtocolSpec schema and ensure it is consistent with the transformations applied. Output JSON only.",
                )
            )
    
    return None, validation_issues, None, df_after



@dataclass(frozen=True)
class TransformValidationPolicy:
    # --- Validation #2
    allow_bool_inputs: bool = True

    # --- Validation #3
    binary_allowed_values: tuple[int, int] = (0, 1)
    min_variance: float = 1e-12
    duration_min_value: float = 0.0

    # --- Validation #4
    value_tol: float = 1e-9
    check_one_hot_group_row_sums: bool = True
    allow_zero_sum_rows: bool = True

    # --- Validation #5
    uniqueness_ratio_threshold: float = 0.98
    max_allowed_id_like: int = 0  # 0 => FAIL if any

    # --- Validation #6
    controls_min_variance: float = 1e-12
    max_constant_allowed: Optional[int] = None
    skip_binary_one_hot_in_constant_check: bool = True

    # --- Validation #7
    max_total_features: Optional[int] = 5000
    max_w_features: Optional[int] = None
    max_x_features: Optional[int] = None
    max_features_per_source_raw: Optional[int] = None

    # --- Validation #8
    minmax_tol: float = 1e-6
    zscore_warn_abs: float = 10.0
    zscore_fail_abs: Optional[float] = None


def _is_fail(issue: ValidationIssueModel) -> bool:
    return issue.severity == "FAIL"

def protocol_spec_to_transformed_spec(
    llm: LLMService,
    protocol: ProtocolSpec,
    transformed_df: pd.DataFrame,
    history: Optional[Sequence[ChatMessage]],
) -> TransformedProtocolSpec:
    user_prompt = build_transformed_protocol_user_prompt_template().format(
        protocol_json=json.dumps(protocol.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        df_after_columns_json=json.dumps(transformed_df.columns.tolist(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )   
    llm_config = LLMConfig(temperature=0.2)
    return llm.generate_json(
        schema=TransformedProtocolSpec,
        system_prompt=get_transform_protocol_node_info(),
        user_prompt=user_prompt,
        config=llm_config,
        history=history,
        max_attempts=2,
    )
    
    


def run_transform_validations(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    policy: Optional[TransformValidationPolicy] = None,
    fail_fast: bool = False,
) -> List[ValidationIssueModel]:
    """
    Runs validation suite in deterministic order.

    Order matters:
      #1 schema referential integrity
      #2 dtype numeric
      #3 Y/T domain checks
      #4 binary/one-hot invariants
      #5 id-like controls
      #6 constant-like controls
      #7 dimensionality caps
      #8 encoding post-conditions

    If fail_fast=True, returns immediately after first FAIL appears.
    """
    pol = policy or TransformValidationPolicy()
    issues: List[ValidationIssueModel] = []

    def _extend(new_issues: List[ValidationIssueModel]) -> None:
        nonlocal issues
        if not new_issues:
            return
        issues.extend(new_issues)
        if fail_fast and any(_is_fail(x) for x in new_issues):
            raise _StopValidation()

    class _StopValidation(Exception):
        pass

    try:
        _extend(
            validate_input_columns_exist_and_are_unambiguous(
                df_after=df_after,
                spec=spec,
            )
        )

        _extend(
            validate_model_inputs_are_numeric_dtypes(
                df_after=df_after,
                spec=spec,
                allow_bool=pol.allow_bool_inputs,
            )
        )

        _extend(
            validate_treatment_outcome_domains_by_kind(
                df_after=df_after,
                spec=spec,
                binary_allowed_values=pol.binary_allowed_values,
                min_variance=pol.min_variance,
                duration_min_value=pol.duration_min_value,
            )
        )

        _extend(
            validate_binary_and_one_hot_invariants(
                df_after=df_after,
                spec=spec,
                value_tol=pol.value_tol,
                check_one_hot_group_row_sums=pol.check_one_hot_group_row_sums,
                allow_zero_sum_rows=pol.allow_zero_sum_rows,
            )
        )

        _extend(
            validate_id_like_features_in_controls(
                df_after=df_after,
                spec=spec,
                uniqueness_ratio_threshold=pol.uniqueness_ratio_threshold,
                max_allowed_id_like=pol.max_allowed_id_like,
            )
        )

        _extend(
            validate_constant_or_near_constant_controls(
                df_after=df_after,
                spec=spec,
                min_variance=pol.controls_min_variance,
                max_constant_allowed=pol.max_constant_allowed,
                skip_binary_one_hot=pol.skip_binary_one_hot_in_constant_check,
            )
        )

        _extend(
            validate_dimensionality_caps(
                df_after=df_after,
                spec=spec,
                max_total_features=pol.max_total_features,
                max_w_features=pol.max_w_features,
                max_x_features=pol.max_x_features,
                max_features_per_source_raw=pol.max_features_per_source_raw,
            )
        )

        _extend(
            validate_encoding_postconditions(
                df_after=df_after,
                spec=spec,
                minmax_tol=pol.minmax_tol,
                zscore_warn_abs=pol.zscore_warn_abs,
                zscore_fail_abs=pol.zscore_fail_abs,
            )
        )

    except _StopValidation:
        pass

    return issues

def apply_plan(
    *,
    df: pd.DataFrame,
    plans: TransformPlanModel,
) -> Tuple[Optional[pd.DataFrame], List[ValidationIssueModel]]:
    """
    Applies the encoding plan to the dataframe.
    Returns the transformed dataframe and any issues encountered during application.
    """
    all_issues: List[ValidationIssueModel] = []
    
    for plan_decision in plans.columns:
        df, issues =apply_encoding_plan(
            df=df,
            plan=plan_decision,
        )
        if issues is not None:
            all_issues.append(issues)

    return df, all_issues



def get_transform_encoding_plan(
    llm: LLMService,
    protocol: ProtocolSpec,
    history: Optional[Sequence[ChatMessage]],
    dataset_summary: DatasetSummaryModel) -> TransformPlanModel:
    
    system_prompt = build_transform_plan_system_prompt()
    user_prompt = build_transform_plan_user_prompt_template().format(
        protocol_json=json.dumps(protocol.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        summary_json=json.dumps(dataset_summary.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return  llm.generate_json(
        schema=TransformPlanModel,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=LLMConfig(temperature=0.4),
        history=history,
        max_attempts=2,   
    )
    
def _repair_messages_from_issues(
    *,
    attempt: int,
    stage: str,
    issues: List[ValidationIssueModel],
) -> List[ChatMessage]:
    """
    Minimal “propagate error” into next attempt.
    Kept small: only FAIL issues + up to N.
    """
    fails = [x for x in issues if x.severity == "FAIL"]
    sample = (fails or issues)[:10]
    payload = { # pyright: ignore[reportUnknownVariableType]
        "attempt": attempt,
        "stage": stage,
        "n_issues": len(issues),
        "issues_sample": [x.model_dump(mode="json") for x in sample],
        "instruction": "On the next attempt, fix the JSON output to satisfy schema and constraints. Output JSON only.",
    }
    return [ChatMessage(role="system", content=f"REPAIR_CONTEXT={payload}")]    
    