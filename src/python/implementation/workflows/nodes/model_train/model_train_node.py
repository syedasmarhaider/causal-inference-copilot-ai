from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from typing import Any, Optional, Sequence, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    ProtocolSpec,
    BinaryTreatmentSpecModel as ProtocolBinaryTreatmentSpecModel,
    CategoricalTreatmentSpecModel as ProtocolCategoricalTreatmentSpecModel,
    BinaryOutcomeSpecModel as ProtocolBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as ProtocolContinuousOutcomeSpecModel,
)
from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.nodes.model_train.model_train_prompts import (
    ENCODING_PLAN_SYSTEM_PROMPT,
    ENCODING_PLAN_USER_PROMPT_TEMPLATE,
    FIT_SUCCESS_FAILURE_SYSTEM_PROMPT,
    get_model_train_node_info,
)
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainPayload, ModelTrainState

from python.implementation.workflows.tools.causal.causal_command import CommandFailure, FitCommand, FitInputs, FitResult, FitSuccess
from python.implementation.workflows.tools.causal.causal_model_factory_tool import CausalModelFactoryTool
from python.implementation.workflows.tools.causal.causal_spec import (
    CausalSpec,
    BinaryTreatmentSpecModel as CausalBinaryTreatmentSpecModel,
    CategoricalTreatmentSpecModel as CausalCategoricalTreatmentSpecModel,
    BinaryOutcomeSpecModel as CausalBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as CausalContinuousOutcomeSpecModel,
)

from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetSummaryModel
from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan


# ---------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------
class EncodingPlanLLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    plan: Optional[TransformPlan] = None
    message: str = Field(..., min_length=1)
    needs_user_input: bool = False

    @model_validator(mode="after")
    def _coherence(self) -> "EncodingPlanLLMOutput":
        if self.needs_user_input:
            if self.plan is not None:
                raise ValueError("needs_user_input=True requires plan=null.")
        else:
            if self.plan is None:
                raise ValueError("needs_user_input=False requires a non-null plan.")
        return self


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _safe_model_dump(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    return x

def _protocol_to_causal_spec(protocol: ProtocolSpec) -> CausalSpec:
    # -------- Treatment (T) --------
    t = protocol.treatment_spec
    if isinstance(t, ProtocolBinaryTreatmentSpecModel):
        t_specs = CausalBinaryTreatmentSpecModel(
            kind="binary",
            column=t.column,
            treated_values=[t.treated],
            control_values=[t.control],
        )
    elif isinstance(t, ProtocolCategoricalTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        t_specs = CausalCategoricalTreatmentSpecModel(
            kind="categorical",
            column=t.column,
            levels=list(t.levels),
            baseline=None,
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



def _validate_plan_against_constraints(
    *,
    plan: TransformPlan,
    eligible_cols: set[str],
    treatment_col: Optional[str],
    outcome_col: Optional[str],
) -> None:
    cols = [c.column for c in plan.columns]
    if len(cols) != len(set(cols)):
        raise ValueError("Encoding plan has duplicate column entries (not allowed).")

    forbidden = {c for c in (treatment_col, outcome_col) if c}
    illegal = sorted(set(cols) & forbidden)
    if illegal:
        raise ValueError(f"Encoding plan must not include treatment/outcome columns: {illegal}")

    plan_set = set(cols)
    missing = sorted(eligible_cols - plan_set)
    extra = sorted(plan_set - eligible_cols)
    if missing:
        raise ValueError(f"Encoding plan is missing eligible columns: {missing}")
    if extra:
        raise ValueError(f"Encoding plan contains non-eligible columns: {extra}")

    has_x = any(c.role == "X" for c in plan.columns)
    has_w = any(c.role == "W" for c in plan.columns)
    if not has_x or not has_w:
        raise ValueError("Encoding plan must include at least one X and at least one W column.")


def _generate_encoding_plan(
    *,
    llm: LLMService,
    llm_config: LLMConfig,
    protocol: ProtocolSpec,
    selected_model: Any,
    dataset_summary: DatasetSummaryModel,
    prev_training_error: Optional[str] = None,
    documentation: Optional[str] = None,
    history: Optional[Sequence[ChatMessage]],
) -> EncodingPlanLLMOutput:    
    # Eligible columns = X+W minus treatment/outcome
    X_cols = list(protocol.covariates or [])
    W_cols = list(protocol.effect_modifiers or [])
    treatment_col = protocol.treatment_spec.column
    outcome_col = protocol.outcome_spec.column
    eligible = set(X_cols) | set(W_cols)
    eligible.discard(treatment_col)
    eligible.discard(outcome_col)

    if not eligible:
        raise ValueError("No eligible columns for encoding plan (no covariates/effect modifiers besides treatment/outcome).")
    
  

    user_prompt = ENCODING_PLAN_USER_PROMPT_TEMPLATE.format(
        model_selection_json=_dumps(_safe_model_dump(selected_model)),
        selected_model_json=_dumps(_safe_model_dump(selected_model)),
        protocol_json=_dumps(_safe_model_dump(protocol)),
        dataset_summary_json=_dumps(_safe_model_dump(dataset_summary)),
        prev_training_errors_string=prev_training_error,
        documentation_string=documentation,

    )
    
    last_3_messages = list(history[-3:]) if history else None

    out = llm.generate_json(
        schema=EncodingPlanLLMOutput,
        system_prompt=ENCODING_PLAN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        config=llm_config,
        history=last_3_messages,
        max_attempts=3,
    )

    if out.needs_user_input:
        return out

    assert out.plan is not None
    _validate_plan_against_constraints(
        plan=out.plan,
        eligible_cols=eligible,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )
    return out


@dataclass(frozen=True, slots=True)
class ModelTrainNode(Node):
    llm: LLMService
    model_name: str

    @property
    def name(self) -> str:
        return ModelTrainState.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_model_train_node_info()

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
        if not isinstance(state, ModelTrainState):
            raise ValueError(f"{self.name}: invalid state (got {type(state).__name__})")
        
        deps = ModelTrainDeps.from_loaded(previous_state_dependencies)
        
        protocol = deps.compile_protocol.payload.protocol
        assert protocol is not None, "Compiled protocol must be available for model training."

        selected = deps.model_selection.payload.confirmed_model_selection
        assert selected is not None, "Confirmed model selection must be available for model training."

        clean_dataset_id = getattr(deps.clean_protocol.payload, "clean_dataset_id", None)
        assert clean_dataset_id is not None, "Clean dataset ID must be available for model training."
        
        dataset_summary = deps.clean_protocol.payload.summary
        assert dataset_summary is not None, "Cleaned dataset summary must be available for encoding plan generation."
                # Resolve selected causal model
        mf_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        model_factory = cast(CausalModelFactoryTool, mf_raw)

        estimator_fqcn = selected.selected_model
        assert estimator_fqcn is not None, "Selected model must include the fully qualified class name."
        model = model_factory.resolve(estimator_fqcn)
        if model is None:
            raise ValueError(f"Selected model '{estimator_fqcn}' is not supported by the CausalModelFactoryTool.")
        
        if len(protocol.covariates or []) == 0 and len(protocol.effect_modifiers or []) == 0:
                return ModelTrainState(
                    payload= ModelTrainPayload(
                        trained_model_id=None, 
                        column_transformation_plan=None,
                        col_tranformation_not_needed=True,
                        training_warnings=None,
                        user_message="No covariates or effect modifiers detected, so no column transformation needed. Proceeding to the training",
                        needs_user_input=False,
                        error=None,
                    )
                )

        if state.payload.column_transformation_plan is None and (state.payload.col_tranformation_not_needed is None or not state.payload.col_tranformation_not_needed):
            if len(protocol.covariates or []) == 0 and len(protocol.effect_modifiers or []) == 0:
                return ModelTrainState( 
                     payload= ModelTrainPayload(
                        trained_model_id=None, 
                        column_transformation_plan=None,
                        col_tranformation_not_needed=True,
                        training_warnings=None,
                        user_message="No covariates or effect modifiers detected, so no column transformation needed. Proceeding to the training",
                        needs_user_input=False,
                        error=None,
                    )
                )
            
            plan = _generate_encoding_plan(
                llm=self.llm,
                llm_config=LLMConfig(temperature=0.2, model=self.model_name),
                protocol=protocol,
                selected_model=selected,
                dataset_summary=dataset_summary,
                history=messages_history,
                prev_training_error=state.payload.prev_training_errors,
                documentation=model.get_command_info("FIT"),
            )
            
            if plan.needs_user_input:
                logging.warning("ModelTrainNode: LLM indicated user input needed for encoding plan clarification.")
                payload = state.payload.model_copy(
                    update={
                        "needs_user_input": True,
                        "error": None,
                        "user_message": plan.message,
                        "column_transformation_plan": None,
                        "col_tranformation_not_needed": None,
                    }
                )
                return ModelTrainState(payload=payload)   
            
            if plan.plan is None:
                raise ValueError("LLM indicated no user input needed but did not return a plan.")
            
            return ModelTrainState(
                payload=state.payload.model_copy(
                    update={
                        "column_transformation_plan": plan.plan,
                        "col_tranformation_not_needed": False,
                        "needs_user_input": False,
                        "error": None,
                        "user_message": plan.message + "\n\nProceeding to the training.",
                    }
                )
            )
            
        order_X = None
        order_W = None 
        if not state.payload.col_tranformation_not_needed:
            assert state.payload.column_transformation_plan is not None, "Column transformation plan must be available if transformation is needed."
            order_X = [c.column for c in state.payload.column_transformation_plan.columns if c.role == "X"]
            order_W = [c.column for c in state.payload.column_transformation_plan.columns if c.role == "W"]
        run_id = uuid4()
        causal_spec = _protocol_to_causal_spec(protocol)        
        
        cmd = FitCommand(
            model_name=estimator_fqcn,
            dataset_id=clean_dataset_id,
            run_id=run_id,
            protocol_specs=causal_spec,
            data_summary=dataset_summary,
            order_X=order_X,
            order_W=order_W,
            transformation_plan=state.payload.column_transformation_plan if not state.payload.col_tranformation_not_needed else None,
            inputs=FitInputs(),
        )

        res = model.execute(user_id=user_id, conversation_id=conversation_id, command=cmd)
        logging.warning(f"Model training command executed with result: {res}")
        if not isinstance(res, FitResult):
            raise ValueError(f"Expected FitResult from model execution, got {type(res).__name__}")

        match res:
            case FitSuccess():
                message = self.llm.generate(
                    config=LLMConfig(temperature=0.2, model=self.model_name),
                    system_prompt=FIT_SUCCESS_FAILURE_SYSTEM_PROMPT,
                    user_prompt=f"Model training succeeded with warnings: {res.warnings}. Explain to the user in a clinician-friendly way.",
                    history=messages_history,
                ).content
                    
                fitted_model_id = res.fitted_model_id
                warnings_list = res.warnings or []
                warnings_str = "\n".join([str(w) for w in warnings_list]) if warnings_list else None
                payload = state.payload.model_copy(
                   update={
                    "trained_model_id": fitted_model_id,
                    "training_warnings": warnings_str,
                    "order_X": order_X,
                    "order_W": order_W,
                    "needs_user_input": False,
                    "no_of_times_trained": (state.payload.no_of_times_trained or 0) + 1,
                    "error": None,
                    "user_message": message,
                 }
                )
                return ModelTrainState(payload=payload)
            
            case CommandFailure():
                
                err_obj = res.error
                err_msg = getattr(err_obj, "message", None) or str(err_obj) or "Training failed for an unknown reason."
                message = self.llm.generate(
                    config=LLMConfig(temperature=0.2, model=self.model_name),
                    system_prompt=FIT_SUCCESS_FAILURE_SYSTEM_PROMPT,
                    user_prompt=f"Model training failed with error: {err_msg}. Explain to the user in a clinician-friendly way and suggest next steps.",
                    history=messages_history,
                ).content
                if state.payload.no_of_times_trained is not None and state.payload.no_of_times_trained >= state.MaxNoOfInterationTrain:
                    return ModelTrainState(
                        payload=state.payload.model_copy(
                            update={
                                "trained_model_id": None,
                                "training_warnings": None,
                                "order_X": None,
                                "column_transformation_plan": None,
                                "order_W": None,
                                "needs_user_input": False,
                                "no_of_times_trained": state.payload.no_of_times_trained,
                                "error": err_msg,
                                "user_message": message,
                            }
                        )
                    )
                    
                payload = state.payload.model_copy(
                    update={
                        "trained_model_id": None,
                        "needs_user_input": False,
                        "error": None,
                        "column_transformation_plan": None,
                        "prev_training_errors": err_msg,
                        "user_message": message,
                        "no_of_times_trained": (state.payload.no_of_times_trained or 0) + 1,
                    }
                )
                return ModelTrainState(payload=payload)
            
            case _:
                 raise ValueError(f"Unexpected FitResult status: {getattr(res, 'status', None)}")
