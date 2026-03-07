from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast, ClassVar, Mapping
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    ProtocolSpec,
    BinaryTreatmentSpecModel as ProtocolBinaryTreatmentSpecModel,
    BinaryOutcomeSpecModel as ProtocolBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as ProtocolContinuousOutcomeSpecModel,
)
from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.nodes.model_train.model_train_prompts import (
    ENCODING_PLAN_PLAN_USER_PROMPT_TEMPLATE,
    ENCODING_PLAN_TRIAGE_USER_PROMPT_TEMPLATE,
    FIT_SUCCESS_FAILURE_SYSTEM_PROMPT,
    get_model_train_node_info,
)
from python.implementation.workflows.nodes.model_train.model_train_state import (
    ModelTrainState,
)

from python.implementation.workflows.tools.causal.causal_command import (
    CommandFailure,
    FitCommand,
    FitInputs,
    FitResult,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.causal_model_factory_tool import CausalModelFactoryTool
from python.implementation.workflows.tools.causal.causal_spec import (
    CausalSpec,
    BinaryTreatmentSpecModel as CausalBinaryTreatmentSpecModel,
    BinaryOutcomeSpecModel as CausalBinaryOutcomeSpecModel,
    ContinuousOutcomeSpecModel as CausalContinuousOutcomeSpecModel,
)
from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetSummaryModel


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------
class UserPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(..., min_length=1)
    needs_user_input: bool


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _safe_model_dump(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    return x


# ---------------------------------------------------------------------
# Dataset-summary helpers
# ---------------------------------------------------------------------
def _build_profile_index(dataset_summary: DatasetSummaryModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in dataset_summary.profiles:
        n = getattr(p, "name", None)
        if isinstance(n, str) and n.strip():
            out[n.strip()] = p
    return out


def _kind_of(profile: Any) -> str:
    return str(getattr(profile, "inferred_kind", "OTHER"))


def _parse_bool_token(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().casefold()
        if s in {"true", "1", "yes", "t"}:
            return True
        if s in {"false", "0", "no", "f"}:
            return False
    return None


def _parse_numeric_token(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _coerce_literal_against_summary(
    *,
    column: str,
    value: str,
    dataset_summary: DatasetSummaryModel,
) -> Any:
    """
    Convert protocol literals like "1"/"0"/"true" into a type compatible with the
    cleaned dataset summary when possible. This fixes strict downstream mismatches
    such as string "1" vs numeric 1.
    """
    by_name = _build_profile_index(dataset_summary)
    profile = by_name.get(column)
    if profile is None:
        return value

    kind = _kind_of(profile)

    if kind == "BOOLEAN":
        parsed_bool = _parse_bool_token(value)
        return parsed_bool if parsed_bool is not None else value

    if kind == "NUMERIC":
        parsed_num = _parse_numeric_token(value)
        return parsed_num if parsed_num is not None else value

    # CATEGORICAL / OTHER: keep string literal as-is
    return value


def _protocol_to_causal_spec(
    protocol: ProtocolSpec,
    dataset_summary: DatasetSummaryModel,
) -> CausalSpec:
    # -------- Treatment (T) --------
    t = protocol.treatment_spec
    if isinstance(t, ProtocolBinaryTreatmentSpecModel):  # pyright: ignore[reportUnnecessaryIsInstance]
        treated_value = _coerce_literal_against_summary(
            column=str(t.column),
            value=str(t.treated),
            dataset_summary=dataset_summary,
        )
        control_value = _coerce_literal_against_summary(
            column=str(t.column),
            value=str(t.control),
            dataset_summary=dataset_summary,
        )

        t_specs = CausalBinaryTreatmentSpecModel(
            kind="binary",
            column=t.column,
            treated_values=[treated_value],
            control_values=[control_value],
        )
    else:
        raise TypeError(f"Unsupported treatment_spec type: {type(t).__name__}")

    # -------- Outcome (Y) --------
    y = protocol.outcome_spec
    if isinstance(y, ProtocolBinaryOutcomeSpecModel):
        event_value = _coerce_literal_against_summary(
            column=str(y.column),
            value=str(y.event),
            dataset_summary=dataset_summary,
        )
        non_event_value = _coerce_literal_against_summary(
            column=str(y.column),
            value=str(y.non_event),
            dataset_summary=dataset_summary,
        )

        y_specs = CausalBinaryOutcomeSpecModel(
            kind="binary",
            column=y.column,
            event_values=[event_value],
            non_event_values=[non_event_value],
        )
    elif isinstance(y, ProtocolContinuousOutcomeSpecModel):  # pyright: ignore[reportUnnecessaryIsInstance]
        y_specs = CausalContinuousOutcomeSpecModel(
            kind="continuous",
            column=y.column,
            unit=y.unit,
            clip_min=y.clip_min,
            clip_max=y.clip_max,
        )
    else:
        raise TypeError(f"Unsupported outcome_spec type: {type(y).__name__}")

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
    expected_w_cols: set[str],
    expected_x_cols: set[str],
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

    role_by_col = {c.column: c.role for c in plan.columns}

    wrong_w = sorted(c for c in expected_w_cols if role_by_col.get(c) != "W")
    wrong_x = sorted(c for c in expected_x_cols if role_by_col.get(c) != "X")

    if wrong_w:
        raise ValueError(f"Encoding plan assigned wrong role for W columns: {wrong_w}")
    if wrong_x:
        raise ValueError(f"Encoding plan assigned wrong role for X columns: {wrong_x}")


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
) -> tuple[UserPlanInput, Optional[TransformPlan]]:
    covariate_cols = set(protocol.covariates or [])
    effect_modifier_cols = set(protocol.effect_modifiers or [])

    treatment_col = str(protocol.treatment_spec.column)
    outcome_col = str(protocol.outcome_spec.column)

    eligible = (covariate_cols | effect_modifier_cols) - {treatment_col, outcome_col}

    if not eligible:
        raise ValueError(
            "No eligible columns for encoding plan "
            "(no covariates/effect modifiers besides treatment/outcome)."
        )

    prev_training_error_str = prev_training_error or ""
    documentation_str = documentation or ""

    user_prompt_discussion = ENCODING_PLAN_TRIAGE_USER_PROMPT_TEMPLATE.format(
        selected_model_json=_dumps(_safe_model_dump(selected_model)),
        protocol_json=_dumps(_safe_model_dump(protocol)),
        dataset_summary_json=_dumps(_safe_model_dump(dataset_summary)),
        prev_training_errors_string=prev_training_error_str,
        documentation_string=documentation_str,
    )

    last_8_messages = list(history[-8:]) if history else None

    out = llm.generate_json(
        schema=UserPlanInput,
        system_prompt=None,
        user_prompt=user_prompt_discussion,
        config=llm_config,
        history=last_8_messages,
        max_attempts=2,
    )

    if out.needs_user_input:
        return out, None

    user_prompt_plan = ENCODING_PLAN_PLAN_USER_PROMPT_TEMPLATE.format(
        selected_model_json=_dumps(_safe_model_dump(selected_model)),
        protocol_json=_dumps(_safe_model_dump(protocol)),
        dataset_summary_json=_dumps(_safe_model_dump(dataset_summary)),
        prev_training_errors_string=prev_training_error_str,
        documentation_string=documentation_str,
    )

    plan = llm.generate_json(
        schema=TransformPlan,
        system_prompt=None,
        user_prompt=user_prompt_plan,
        config=llm_config,
        history=last_8_messages,
        max_attempts=3,
    )

    _validate_plan_against_constraints(
        plan=plan,
        eligible_cols=eligible,
        expected_w_cols=covariate_cols - {treatment_col, outcome_col},
        expected_x_cols=effect_modifier_cols - {treatment_col, outcome_col},
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )

    return out, plan


@dataclass(frozen=True, slots=True)
class ModelTrainNode(Node):
    llm: LLMService

    NAME: ClassVar[str] = ModelTrainState.NAME

    @property
    def name(self) -> str:
        return self.NAME

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
        previous_state_dependencies: Mapping[str, State],
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
        assert dataset_summary is not None, (
            "Cleaned dataset summary must be available for encoding plan generation."
        )

        mf_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        model_factory = cast(CausalModelFactoryTool, mf_raw)

        estimator_fqcn = selected.selected_model
        assert estimator_fqcn is not None, (
            "Selected model must include the fully qualified class name."
        )

        model = model_factory.resolve(estimator_fqcn)
        if model is None:
            raise ValueError(
                f"Selected model '{estimator_fqcn}' is not supported by the "
                f"CausalModelFactoryTool."
            )

        has_any_adjustment_cols = bool(protocol.covariates or []) or bool(protocol.effect_modifiers or [])

        # -----------------------------------------------------------------
        # Phase 1: prepare encoding plan if needed
        # -----------------------------------------------------------------
        if (
            state.payload.column_transformation_plan is None and has_any_adjustment_cols
        ):

            user_discussion, plan = _generate_encoding_plan(
                llm=self.llm,
                llm_config=LLMConfig(temperature=0.4, model="pro"),
                protocol=protocol,
                selected_model=selected,
                dataset_summary=dataset_summary,
                history=messages_history,
                prev_training_error=state.payload.prev_training_errors,
                documentation=model.get_command_info("FIT"),
            )

            if user_discussion.needs_user_input:
                log.warning(
                    "ModelTrainNode: LLM indicated user input needed for encoding plan clarification."
                )
                payload = state.payload.model_copy(
                    update={
                        "needs_user_input": True,
                        "error": None,
                        "user_message": user_discussion.message,
                        "column_transformation_plan": None,
                        "col_tranformation_not_needed": None,
                    }
                )
                return ModelTrainState(payload=payload)

            if plan is None:
                raise ValueError("LLM indicated no user input needed but did not return a plan.")

            return ModelTrainState(
                payload=state.payload.model_copy(
                    update={
                        "column_transformation_plan": plan,
                        "col_tranformation_not_needed": False,
                        "needs_user_input": False,
                        "error": None,
                        "user_message": user_discussion.message + "\n\nProceeding to training.",
                    }
                )
            )

        # -----------------------------------------------------------------
        # Phase 2: training
        # -----------------------------------------------------------------
        order_X: Optional[list[str]] = None
        order_W: Optional[list[str]] = None

        if state.payload.column_transformation_plan is not None:
            order_X = [
                c.column
                for c in state.payload.column_transformation_plan.columns
                if c.role == "X"
            ]
            order_W = [
                c.column
                for c in state.payload.column_transformation_plan.columns
                if c.role == "W"
            ]

        run_id = uuid4()
        causal_spec = _protocol_to_causal_spec(protocol, dataset_summary)

        cmd = FitCommand(
            model_name=estimator_fqcn,
            dataset_id=clean_dataset_id,
            run_id=run_id,
            protocol_specs=causal_spec,
            data_summary=dataset_summary,
            order_X=order_X,
            order_W=order_W,
            transformation_plan=(
                state.payload.column_transformation_plan
                if state.payload.column_transformation_plan is not None
                else None
            ),
            inputs=FitInputs(),
        )

        res = model.execute(user_id=user_id, conversation_id=conversation_id, command=cmd)
        log.warning("Model training command executed with result: %s", res)

        if not isinstance(res, FitResult):
            raise ValueError(f"Expected FitResult from model execution, got {type(res).__name__}")

        match res:
            case FitSuccess():
                message = self.llm.generate(
                    config=LLMConfig(temperature=0.2, model="basic"),
                    system_prompt=FIT_SUCCESS_FAILURE_SYSTEM_PROMPT,
                    user_prompt=(
                        f"Model training succeeded with warnings: {res.warnings}. "
                        f"Explain to the user in a clinician-friendly way."
                    ),
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
                err_msg = (
                    getattr(err_obj, "message", None)
                    or str(err_obj)
                    or "Training failed for an unknown reason."
                )

                message = self.llm.generate(
                    config=LLMConfig(temperature=0.2, model="basic"),
                    system_prompt=FIT_SUCCESS_FAILURE_SYSTEM_PROMPT,
                    user_prompt=(
                        f"Model training failed with error: {err_msg}. "
                        f"Explain to the user in a clinician-friendly way and suggest next steps."
                    ),
                    history=messages_history,
                ).content

                if (
                    state.payload.no_of_times_trained is not None
                    and state.payload.no_of_times_trained >= state.MaxNoOfInterationTrain
                ):
                    return ModelTrainState(
                        payload=state.payload.model_copy(
                            update={
                                "trained_model_id": None,
                                "training_warnings": None,
                                "order_X": None,
                                "order_W": None,
                                "column_transformation_plan": None,
                                "col_tranformation_not_needed": None,
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
                        "training_warnings": None,
                        "order_X": None,
                        "order_W": None,
                        "needs_user_input": False,
                        "error": None,
                        "column_transformation_plan": None,
                        "col_tranformation_not_needed": None,
                        "prev_training_errors": err_msg,
                        "user_message": message,
                        "no_of_times_trained": (state.payload.no_of_times_trained or 0) + 1,
                    }
                )
                return ModelTrainState(payload=payload)

            case _:
                raise ValueError(f"Unexpected FitResult status: {getattr(res, 'status', None)}")