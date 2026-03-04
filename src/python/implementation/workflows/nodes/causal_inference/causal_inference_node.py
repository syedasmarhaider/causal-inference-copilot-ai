from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, List, Optional, Sequence, cast
import pandas as pd
from typing_extensions import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import CausalInferenceDeps
from python.implementation.workflows.nodes.causal_inference.causal_inference_prompts import (
    CATE_GENERAL_PROMPT,
    CATE_INCLUSION_PROMPT,
    CATE_SUMMARY_PROMPT,
    CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT,
    CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE,
    CAUSAL_INFERENCE_MAIN_SYSTEM_PROMPT,
    SMALL_ROUTER_PROMPT,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferenceState,
)
from python.implementation.workflows.tools.causal.causal_model import CausalModel
from python.implementation.workflows.tools.causal.causal_spec import (
    CausalSpec,
    BinaryTreatmentSpecModel as CausalBinaryTreatmentSpecModel,
    BinaryOutcomeSpecModel as CausalBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as CausalContinuousOutcomeSpecModel,
)

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.tools.causal.causal_command import ATECommand, ATEInputsModel, ATEResult, ATESuccess, CATECommand, CATEInputs, CATEResult, CATESuccess, CommandFailure  # adjust if needed
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.causal_model_factory_tool import CausalModelFactoryTool

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    ProtocolSpec,
    BinaryTreatmentSpecModel as ProtocolBinaryTreatmentSpecModel,
    BinaryOutcomeSpecModel as ProtocolBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as ProtocolContinuousOutcomeSpecModel,
)

from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.data_processing.data_processing_tool import DataProcessingTool, InclusionRulesModel
from python.implementation.workflows.tools.data_profiling.causal_data_profiling_tool import CausalDataProfilingTool
from python.implementation.workflows.utils.validation import ValidationIssueModel


def _dumps(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, default=str)



def _protocol_to_causal_spec(protocol: ProtocolSpec) -> CausalSpec:
    # -------- Treatment (T) --------
    t = protocol.treatment_spec
    if isinstance(t, ProtocolBinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        t_specs = CausalBinaryTreatmentSpecModel(
            kind="binary",
            column=t.column,
            treated_values=[t.treated],
            control_values=[t.control],
        )
    else:
        raise TypeError(f"Unsupported treatment_spec type: {type(t).__name__}")

    # -------- Outcome (Y) --------
    y = protocol.outcome_spec
    if isinstance(y, ProtocolBinaryOutcomeSpecModel):
        y_specs = CausalBinaryOutcomeSpecModel(
            kind="binary",
            column=y.column,
            event_values=[y.event],
            non_event_values=[y.non_event],
        )
    elif isinstance(y, ProtocolContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        y_specs = CausalContinuousOutcomeSpecModel(
            kind="continuous",
            column=y.column,
            unit=y.unit,
            clip_min=y.clip_min,
            clip_max=y.clip_max,
        )
    else:
        raise TypeError(f"Unsupported outcome_spec type: {type(y).__name__}")

    # -------- Roles --------
    # NOTE: Your CausalSpec validator says W and X must be disjoint and T/Y must not be in W/X/Z.
    # This will raise if ProtocolSpec contains overlaps.
    return CausalSpec(
        Y=y_specs,
        T=t_specs,
        W=list(protocol.covariates),
        X=list(protocol.effect_modifiers),
        Z=[],
    )   

def _serialize_result_to_json_str(res: Any) -> str:
    # Prefer model_dump / dataclass / dict, fall back to repr
    if hasattr(res, "model_dump"):
        return _dumps(res.model_dump(mode="json"))
    if isinstance(res, dict):
        return _dumps(res)
    # dataclass?
    try:
        import dataclasses

        if dataclasses.is_dataclass(res) and not isinstance(res, type):
            return _dumps(dataclasses.asdict(res))
    except Exception:
        pass
    return _dumps({"repr": repr(res)})


@dataclass(frozen=True, slots=True)
class CausalInferenceNode(Node):
    llm: LLMService
    data_repo: DataRepo
    model_name: str

    @property
    def name(self) -> str:
        return CausalInferenceState.NAME

    @classmethod
    def get_info(cls) -> str:
        return "Computes ATE from the trained causal model and answers clinician questions about the result."

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Any,
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        if not isinstance(state, CausalInferenceState):
            raise ValueError(f"{self.name}: invalid state (got {type(state).__name__})")

        deps = CausalInferenceDeps.from_loaded(previous_state_dependencies)

        protocol = deps.compile_protocol.payload.protocol
        assert protocol is not None, "ProtocolSpec is required in CompileProtocolState payload"

        trained_model_id = getattr(deps.model_train.payload, "trained_model_id", None)
        assert trained_model_id is not None, "trained_model_id is required in ModelTrainState payload"

        clean_dataset_id = getattr(deps.clean_protocol.payload, "clean_dataset_id", None)
        assert clean_dataset_id is not None, "clean_dataset_id is required in CleanProtocolState payload"

        selected = deps.model_selection.payload.confirmed_model_selection
        assert selected is not None, "Confirmed model selection is required in ModelSelectionState payload"
        
        selected_model_fqcn = selected.selected_model
        assert selected_model_fqcn is not None, "selected_model (fqcn) is required in confirmed model selection"
        
        data_summary = deps.clean_protocol.payload.summary
        assert data_summary is not None, "dataset_summary is required in CleanProtocolState"
        
        order_X = deps.model_train.payload.order_X
        order_W = deps.model_train.payload.order_W
        assert order_X is not None, "order_X is required in ModelTrainState payload"
        assert order_W is not None, "order_W is required in ModelTrainState payload"

        # Resolve model tool
        mf_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        model_factory = cast(CausalModelFactoryTool, mf_raw)

        model = model_factory.resolve(selected_model_fqcn)
        assert model is not None, f"Model factory could not resolve model for fqcn: {selected_model_fqcn}"
        
        data_profiling_tool_raw = tool_factory.get_tool(CausalDataProfilingTool.NAME)
        data_profiling_tool = cast(CausalDataProfilingTool, data_profiling_tool_raw)
        data_processing_tool_raw = tool_factory.get_tool(DataProcessingTool.NAME)
        data_processing_tool = cast(DataProcessingTool, data_processing_tool_raw)
        

        # Context bundle for prompts
        context: dict[str, Any] = {
            "selected_model": selected_model_fqcn,
            "protocol": protocol.model_dump(mode="json"),
            "dataset_summary": data_summary.model_dump(mode="json"),
        }

        # ============================================================
        # Step 1: Compute ATE once (idempotent)
        # ============================================================
        if state.payload.ate_result_raw_json_str is None:
                spec = _protocol_to_causal_spec(protocol)

                # Build ATECommand (fields must match your dataclass)
                # Required by your backend snippet: command.protocol_specs, command.fitted_model_id, command.inputs.alpha, command.run_id
                cmd = ATECommand(
                    model_name=selected_model_fqcn,
                    dataset_id=clean_dataset_id,
                    run_id=uuid4(),
                    data_summary=data_summary,
                    transformation_plan=deps.model_train.payload.column_transformation_plan,
                    protocol_specs=spec,
                    fitted_model_id=trained_model_id,
                    order_X=order_X,
                    order_W=order_W,
                    inputs=ATEInputsModel(),
                    options={},
                )

                logging.warning(f"Executing ATECommand with model {selected_model_fqcn}, dataset_id {clean_dataset_id}, fitted_model_id {trained_model_id}, command: {cmd}")
                res = model.execute(user_id=user_id, conversation_id=conversation_id, command=cmd)
                logging.warning(f"ATECommand executed with result: {res}")
                
                if not isinstance(res, ATEResult):
                    raise TypeError(f"Expected ATEResult from model.execute, got {type(res).__name__}")
                
                match res:
                    case ATESuccess():
                            result = _serialize_result_to_json_str(res.ate)
                            warnings = res.warnings if hasattr(res, "warnings") else []
                            summary_out = self.llm.generate(
                                system_prompt=CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT,
                                user_prompt=CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE.format(
                                    context_json=_dumps(context),
                                    ate_result_json=result,
                                    warnings_json=_dumps(warnings),
                                ),
                                config=LLMConfig(temperature=0.2, model=self.model_name),
                                history=messages_history[-8:] if messages_history else None,
                            ).content.strip()
                            
                            return CausalInferenceState(
                                payload=state.payload.model_copy(
                                    update={
                                        "ate_result_raw_json_str": result,
                                        "ate_inference_error": None,
                                        "should_abort": False,
                                        "abort_error_message": None,
                                        "user_message": summary_out.strip(),
                                    }
                                )
                            )
                        
                        
                    case CommandFailure():
                            error_message = f"ATE computation failed: {res.error.message}"
                            return CausalInferenceState(
                                payload=state.payload.model_copy(
                                    update={
                                        "ate_result_raw_json_str": None,
                                        "error": res.error.message,
                                        "should_abort": True,
                                        "user_message": error_message,
                                    }
                                )
                            ) 
                    case _:
                        raise TypeError(f"Unhandled ATEResult type: {type(res).__name__}") 
        
        question_type = _is_question_about_ate_or_cate_abort(
            llm=self.llm,
            model_name=self.model_name,
            messages_history=messages_history,
        )

        
        match question_type.type:
            case "ate":
                return _process_ate_question(
                    llm=self.llm,
                    current_state=state, 
                    model_name=self.model_name,
                    ate_model_output_json_str=state.payload.ate_result_raw_json_str,
                    state=state,
                    data_summary=data_summary,
                    selected_model_fqcn=selected_model_fqcn,
                    messages_history=messages_history,
                )    
            
            case "cate":
                return _process_cate_question(
                    llm=self.llm,
                    data_repo=self.data_repo,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    model_name=self.model_name,
                    ate_model_output_json_str=state.payload.ate_result_raw_json_str,
                    messages_history=messages_history,
                    current_state=state,
                    protocol=protocol,
                    clean_dataset_id=clean_dataset_id,
                    data_summary=data_summary,
                    tranformation_plan=deps.model_train.payload.column_transformation_plan,
                    selected_model_fqcn=selected_model_fqcn,
                    trained_model_id=trained_model_id,
                    order_X=order_X,
                    order_W=order_W,
                    model=model,
                    data_profiling_tool=data_profiling_tool,
                    data_processing_tool=data_processing_tool,
                )
            case "other":
                    return CausalInferenceState(
                        payload=state.payload.model_copy(
                            update={
                                "message": "Are you sure you want to go to the prev step? You will lose all the info.",
                            }
                        )
                    )  
            
            case "abort":
                    return CausalInferenceState(
                        payload=state.payload.model_copy(
                            update={
                                "message": "Aborting the workflow as per your request. If you want to start over, please re-run the workflow.",
                                "should_abort": True,
                            }
                        )
                    )
            
            case _:
                raise ValueError(f"Invalid question type: {question_type.type}")                  
                               
        raise ValueError("Unexpected error: question type could not be determined")
              

#===============================================================
# internal small router
#===============================================================

_Question_Type = Literal["ate", "cate", "other","abort"]

class _QuestionDecisonPayload(BaseModel):
    type: Optional[_Question_Type] = None
    reasoning: Optional[str] = None
def _is_question_about_ate_or_cate_abort(
    llm: LLMService,
    model_name: str,
    messages_history: Optional[Sequence[ChatMessage]],
                                        ) -> _QuestionDecisonPayload:
    last_8_messages = messages_history[-8:] if messages_history else None
    return llm.generate_json(
        schema=_QuestionDecisonPayload,
        system_prompt=None,
        user_prompt=SMALL_ROUTER_PROMPT,
        config=LLMConfig(temperature=0.2, model=model_name),
        history=last_8_messages,
        max_attempts=3,
    )

#===============================================================
# ATE questions 
#===============================================================

def _process_ate_question(
    llm: LLMService,
    current_state: CausalInferenceState,
    model_name: str,
    ate_model_output_json_str: str,
    state: CausalInferenceState,
    data_summary: DatasetSummaryModel,
    selected_model_fqcn: str,
    messages_history: Optional[Sequence[ChatMessage]]) -> CausalInferenceState:
    last_8_messages = messages_history[-8:] if messages_history else None
    answer = llm.generate(
        system_prompt=CAUSAL_INFERENCE_MAIN_SYSTEM_PROMPT.format(
            data_summary=data_summary.model_dump(mode="json"),
            ate_model_output_json_str=ate_model_output_json_str,
            selected_model_fqcn=selected_model_fqcn,
        ),
        user_prompt="{user_question}",
        config=LLMConfig(temperature=0.7, model=model_name),
        history=last_8_messages,
    ).content.strip()
    return CausalInferenceState(
        payload=state.payload.model_copy(
            update={
                "message": answer,
            }        )
    )
    
#===============================================================
# Cate  Questions
#===============================================================
class _CateIntentPayload(BaseModel):
    prev_context_relevant: bool 
    answer: Optional[str] = None

def _process_cate_question(
    llm: LLMService,
    data_repo: DataRepo,
    user_id: UUID,
    conversation_id: UUID,
    model_name: str,
    ate_model_output_json_str: str,
    messages_history: Optional[Sequence[ChatMessage]],
    current_state: CausalInferenceState,
    protocol: ProtocolSpec,
    clean_dataset_id: UUID,
    data_summary: DatasetSummaryModel,
    tranformation_plan: Optional[TransformPlan],
    selected_model_fqcn: str,
    trained_model_id: UUID,
    order_X: List[str],
    order_W: List[str],
    model: CausalModel,
    data_profiling_tool: CausalDataProfilingTool,
    data_processing_tool: DataProcessingTool) -> CausalInferenceState:
    
    last_8_messages = messages_history[-8:] if messages_history else None
    intent = llm.generate_json(
        schema=_CateIntentPayload,
        user_prompt=CATE_GENERAL_PROMPT,
        system_prompt=None,
        config=LLMConfig(temperature=0.2, model=model_name),
        history=last_8_messages,
        max_attempts=3,
     )
    
    if intent.prev_context_relevant and intent.answer is not None:
        return CausalInferenceState(
            payload=current_state.payload.model_copy(
                update={
                    "message": intent.answer,
                }
             )
        )
    
    encoding_plan: Optional[InclusionRulesModel] = None
    error_message: Optional[str] = None
    for attempt in range(3):
            encoding_plan = llm.generate_json(
                schema=InclusionRulesModel, 
                system_prompt=None,
                user_prompt= CATE_INCLUSION_PROMPT.format(
                    PROTOCOL_SPEC_JSON=protocol.model_dump(mode="json"),
                    DATA_SUMMARY_JSON=data_summary.model_dump(mode="json"),
                ) + (f"\nPrevious error message: {error_message}" if error_message else ""),
                config=LLMConfig(temperature=0.2, model=model_name),
                max_attempts=3,
                history=last_8_messages,
            )
            issues = _validate_inclusion_rules_semantic(effect_modifiers=protocol.effect_modifiers, plan=encoding_plan)
            if len(issues) == 0:
                break
            else:
                logging.warning(f"Invalid inclusion plan generated by LLM on attempt {attempt+1}: {encoding_plan}")
                error_message = "The inclusion rules you provided have the following issues:\n" + "\n".join(f"- {issue.message}" for issue in issues) + "\nPlease revise your inclusion rules to fix these issues."
                encoding_plan = None
    
    if encoding_plan is None:
        return CausalInferenceState(
            payload=current_state.payload.model_copy(
                update={
                    "message": "Sorry, I was not able to process your response. Please clarify your question or try again.",
                }
             )
        )
    
    df = data_repo.get_csv_data(
        user_id=user_id,
        conversation_id=conversation_id,
        dataset_id=clean_dataset_id,
    )
    df_effect_modifier= _extract_cols_data(df=df, cols=order_X) 
    df_effect_modifier = data_processing_tool.apply_inclusion_rules(
        df=df_effect_modifier,
        rules=encoding_plan.rules,
    )
    
    cmd = CATECommand(
                    model_name=selected_model_fqcn,
                    dataset_id=clean_dataset_id,
                    run_id=uuid4(),
                    data_summary=data_summary,
                    transformation_plan=tranformation_plan,
                    protocol_specs=_protocol_to_causal_spec(protocol),
                    fitted_model_id=trained_model_id,
                    order_X=order_X,
                    order_W=order_W,
                    inputs=CATEInputs(x_rows=df_effect_modifier),
                    options={},
                )
    res = model.execute(
        user_id=user_id,
        conversation_id=conversation_id,
        command=cmd,
    )
    
    if not isinstance(res, CATEResult):
        raise TypeError(f"Expected CATEResult from model.execute, got {type(res).__name__}")
    
    match res:
        case CATESuccess():
            result = _serialize_result_to_json_str(res.effects) + _serialize_result_to_json_str(res.warnings)
            answer = llm.generate(
                system_prompt=CATE_SUMMARY_PROMPT,
                user_prompt=result,
                config=LLMConfig(temperature=0.2, model=model_name),
                history=last_8_messages,
            ).content.strip()
            return CausalInferenceState(
                payload=current_state.payload.model_copy(
                    update={
                        "message": answer,
                    }
                )
            )
        
        case CommandFailure():
            error_message = f"CATE computation failed: {res.error.message} Please try again sorry for inconvenience."
            return CausalInferenceState(
                payload=current_state.payload.model_copy(
                    update={
                        "message": error_message,
                    }
                )
            )
        case _:
            raise TypeError(f"Unhandled CATEResult type: {type(res).__name__}")     
            
            
def _validate_inclusion_rules_semantic(
    *,
    plan: InclusionRulesModel,
    effect_modifiers: Sequence[str],
) -> List[ValidationIssueModel]:
    issues: List[ValidationIssueModel] = []
    allowed_x = {str(c) for c in effect_modifiers}

    for idx, r in enumerate(plan.rules):
        col = str(r.column)

        # (1) X-only column restriction
        if col not in allowed_x:
            issues.append(
                ValidationIssueModel(
                    severity="FAIL",
                    message=f"Inclusion rule column '{col}' is not an effect modifier (X).",
                    evidence={
                        "rule_index": idx,
                        "column": col,
                        "op": r.op,
                        "values": r.values,
                        "allowed_effect_modifiers": sorted(allowed_x),
                    },
                    fix_hint="Use only columns from protocol.effect_modifiers (X).",
                )
            )

        # (2) values cardinality by op
        if r.op in ("==", ">=", "<=", ">", "<"):
            if len(r.values) != 1:
                issues.append(
                    ValidationIssueModel(
                        severity="FAIL",
                        message=f"Rule {idx} on '{col}' with op '{r.op}' requires exactly 1 value; got {len(r.values)}.",
                        evidence={
                            "rule_index": idx,
                            "column": col,
                            "op": r.op,
                            "values": r.values,
                        },
                        fix_hint="For scalar ops (==, >=, <=, >, <), set values=[single_value].",
                    )
                )
        elif r.op in ("in", "not_in"):
            if len(r.values) < 1:
                issues.append(
                    ValidationIssueModel(
                        severity="FAIL",
                        message=f"Rule {idx} on '{col}' with op '{r.op}' requires a non-empty values list.",
                        evidence={
                            "rule_index": idx,
                            "column": col,
                            "op": r.op,
                            "values": r.values,
                        },
                        fix_hint="For membership ops (in, not_in), set values=[v1, v2, ...].",
                    )
                )
        else:
            # Should be unreachable due to Literal typing, but keep safety if data bypasses typing.
            issues.append(
                ValidationIssueModel(
                    severity="FAIL",
                    message=f"Unsupported operator '{r.op}' in rule {idx} for column '{col}'.",
                    evidence={
                        "rule_index": idx,
                        "column": col,
                        "op": r.op,
                        "values": r.values,
                    },
                    fix_hint="Use one of: ==, in, not_in, >=, <=, >, <",
                )
            )

    return issues
 

    


def _extract_cols_data(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Return a dataframe with exactly `cols` in the given order.

    Strict behavior:
      - Raises KeyError if any requested column is missing.
      - Returns a shallow copy of the selected columns (safe to mutate columns without touching `df`).
      - Preserves original index.

    Notes:
      - Use this to build X_query from effect modifier column names.
    """
    cols_list = [str(c) for c in cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise KeyError(
            f"Requested columns not found in df: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep exact order requested
    return df.loc[:, cols_list].copy()
   
    
    
    
    
    

    
     
    



    
    
    
                        
                
            
            


                            
                
     

                                        


                  
                
 
            
      