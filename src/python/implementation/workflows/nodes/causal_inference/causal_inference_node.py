from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Optional, Sequence, cast
from uuid import UUID, uuid4

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import CausalInferenceDeps
from python.implementation.workflows.nodes.causal_inference.causal_inference_prompts import (
    CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT,
    CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferenceState,
)

from python.implementation.workflows.tools.causal.causal_spec import (
    CausalSpec,
    BinaryTreatmentSpecModel as CausalBinaryTreatmentSpecModel,
    CategoricalTreatmentSpecModel as CausalCategoricalTreatmentSpecModel,
    BinaryOutcomeSpecModel as CausalBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as CausalContinuousOutcomeSpecModel,
)

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.tools.causal.causal_command import ATECommand, ATEInputsModel, ATEResult, ATESuccess, CommandFailure  # adjust if needed
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.causal_model_factory_tool import CausalModelFactoryTool

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    ProtocolSpec,
    BinaryTreatmentSpecModel as ProtocolBinaryTreatmentSpecModel,
    CategoricalTreatmentSpecModel as ProtocolCategoricalTreatmentSpecModel,
    BinaryOutcomeSpecModel as ProtocolBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as ProtocolContinuousOutcomeSpecModel,
)


def _dumps(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, default=str)



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

        # Resolve model tool
        mf_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        model_factory = cast(CausalModelFactoryTool, mf_raw)

        model = model_factory.resolve(selected_model_fqcn)
        assert model is not None, f"Model factory could not resolve model for fqcn: {selected_model_fqcn}"

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
                    protocol_specs=spec,
                    fitted_model_id=trained_model_id,
                    inputs=ATEInputsModel(),
                    options={},
                )

                res = model.execute(user_id=user_id, conversation_id=conversation_id, command=cmd)
                
                if not isinstance(res, ATEResult):
                    raise TypeError(f"Expected ATEResult from model.execute, got {type(res).__name__}")
                
                
                if isinstance(res, ATESuccess):
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
                                "ate_result_summary": summary_out,
                                "ate_inference_error": None,
                                "should_abort": False,
                                "abort_error_message": None,
                                "user_message": summary_out.strip(),
                            }
                        )
                    )
                                        

                if isinstance(res, CommandFailure): # pyright: ignore[reportUnnecessaryIsInstance]
                    error_message = f"ATE computation failed: {res.error.message}"
                    return CausalInferenceState(
                        payload=state.payload.model_copy(
                            update={
                                "ate_result_raw_json_str": None,
                                "ate_result_summary": None,
                                "user_message": error_message,
                            }
                        )
                    ) 
                
                raise TypeError(f"Unhandled ATEResult type: {type(res).__name__}")    
            
        message = self.llm.generate(
            system_prompt= "Talk to user",
            user_prompt= state.payload.ate_result_summary if state.payload.ate_result_summary else "The ATE result is available. How can I assist you with it?",
            config=LLMConfig(temperature=0.2, model=self.model_name),
            history=messages_history[-8:] if messages_history else None,
        ).content.strip()
        
        return CausalInferenceState(
            payload=state.payload.model_copy(
                update={
                    "user_message": message,
                    "needs_user_input": True,
                }
            )
        )