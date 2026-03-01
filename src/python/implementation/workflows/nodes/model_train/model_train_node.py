from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.nodes.model_train.model_train_prompts import (
    ENCODING_PLAN_SYSTEM_PROMPT,
    ENCODING_PLAN_USER_PROMPT_TEMPLATE,
    get_model_train_node_info,
)
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainPayload, ModelTrainState

from python.implementation.workflows.tools.causal.causal_command import FitCommand, FitInputs
from python.implementation.workflows.tools.causal.causal_model_factory_tool import CausalModelFactoryTool
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.encoding.encoding_tool import EncodingTool


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
    deps: ModelTrainDeps,
    history: Optional[Sequence[ChatMessage]],
) -> EncodingPlanLLMOutput:
    protocol = deps.compile_protocol.payload.protocol
    assert protocol is not None, "Protocol must be available for encoding plan generation."

    selected_model = deps.model_selection.payload.confirmed_model_selection
    assert selected_model is not None, "Selected model must be available for encoding plan generation."

    dataset_summary = deps.clean_protocol.payload.summary
    assert dataset_summary is not None, "Cleaned dataset summary must be available for encoding plan generation."
    
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

    )
    
    last_6 = list(history[-6:]) if history else None

    out = llm.generate_json(
        schema=EncodingPlanLLMOutput,
        system_prompt=ENCODING_PLAN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        config=llm_config,
        history=last_6,
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
    llm_config: LLMConfig = LLMConfig(temperature=0.2)

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

        if state.payload.column_transformation_plan is None and state.payload.col_tranformation_not_needed is None:
            if len(protocol.covariates or []) == 0 and len(protocol.effect_modifiers or []) == 0:
                return ModelTrainState(
                     payload= ModelTrainPayload(
                        trained_model_id=None, 
                        column_transformation_plan=None,
                        col_tranformation_not_needed=True,
                        training_warnings=None,
                        warnings=None,
                        user_message="No covariates or effect modifiers detected, so no column transformation needed. Proceeding to the training",
                        needs_user_input=False,
                        error=None,
                    )
                )
            
            plan = _generate_encoding_plan(
                llm=self.llm,
                llm_config=self.llm_config,
                deps=deps,
                history=messages_history,
            )
            
            if plan.needs_user_input:
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
        

        if not state.payload.col_tranformation_not_needed:
           assert state.payload.column_transformation_plan is not None, "Column transformation plan must be available if transformation is needed."
            encoding_tool = cast(EncodingTool, tool_factory.get_tool(EncodingTool.NAME)) 
            encoding_tool.compile(
            plan=state.payload.column_transformation_plan,
            X_order=[c.column for c in state.payload.column_transformation_plan.columns if c.role == "X"],
            W_order=[c.column for c in state.payload.column_transformation_plan.columns if c.role == "W"],
            dense_output=True,
        ) 
               


        
        
            
      

        

            # If plan not stored yet and not marked as not needed, generate plan
            plan = state.payload.column_transformation_plan
            plan_message = "Using existing encoding plan."

            if plan is None and not col_transformation_not_needed:
                out = _generate_encoding_plan(
                    llm=self.llm,
                    llm_config=self.llm_config,
                    deps=deps,
                    history=messages_history,
                )
                if out.needs_user_input:
                    payload = state.payload.model_copy(
                        update={
                            "needs_user_input": True,
                            "error": None,
                            "user_message": out.message,
                            "column_transformation_plan": None,
                            "col_tranformation_not_needed": None,
                        }
                    )
                    return ModelTrainState(payload=payload)

                plan = out.plan
                plan_message = out.message

        # Compile transformers if we have a plan
        pre_X = None
        pre_XW = None
        if plan is not None:
            # Derive X/W orders from the plan itself (so compile input order matches plan roles)
            X_order = [c.column for c in plan.columns if c.role == "X"]
            W_order = [c.column for c in plan.columns if c.role == "W"]

            enc_tool_raw = tool_factory.get_tool(EncodingTool.NAME)
            enc_tool = cast(EncodingTool, enc_tool_raw)
            compiled = enc_tool.compile(plan=plan, X_order=X_order, W_order=W_order, dense_output=True)
            pre_X = compiled.pre_X
            pre_XW = compiled.pre_XW

        # Resolve selected causal model
        mf_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        model_factory = cast(CausalModelFactoryTool, mf_raw)

        estimator_fqcn = selected.selected_model.strip()
        model = model_factory.resolve(estimator_fqcn)
        if model is None:
            payload = state.payload.model_copy(
                update={
                    "needs_user_input": True,
                    "error": None,
                    "user_message": "The selected model is not available. Please choose another model.",
                }
            )
            return ModelTrainState(payload=payload)

        # Build command + execute
        run_id = uuid4()
        causal_spec = _build_causal_spec(protocol)

        cmd = FitCommand(
            model_name=estimator_fqcn,
            dataset_id=clean_dataset_id,
            run_id=run_id,
            protocol_specs=causal_spec,
            inputs=FitInputs(pre_X=pre_X, pre_XW=pre_XW),
        )

        res = model.execute(user_id=user_id, conversation_id=conversation_id, command=cmd)

        # Success/failure handling (duck-typed to your FitResult union)
        if getattr(res, "status", None) == "SUCCEEDED":
            fitted_model_id = getattr(res, "fitted_model_id", None)
            warnings_list = getattr(res, "warnings", []) or []
            warnings_str = "\n".join([str(w) for w in warnings_list]) if warnings_list else None

            payload = state.payload.model_copy(
                update={
                    "trained_model_id": fitted_model_id,
                    "column_transformation_plan": plan,
                    "col_tranformation_not_needed": bool(col_transformation_not_needed or plan is None),
                    "training_warnings": warnings_str,
                    "warnings": None,
                    "needs_user_input": False,
                    "error": None,
                    "user_message": plan_message + "\n\nTraining completed successfully.",
                }
            )
            return ModelTrainState(payload=payload)

        # FAILED
        err_obj = getattr(res, "error", None)
        err_msg = None
        if err_obj is not None:
            err_msg = getattr(err_obj, "message", None) or str(err_obj)
        err_msg = err_msg or "Training failed for an unknown reason."

        payload = state.payload.model_copy(
            update={
                "trained_model_id": None,
                "needs_user_input": True,
                "error": err_msg,
                "user_message": f"Training failed: {err_msg}",
            }
        )
        return ModelTrainState(payload=payload)