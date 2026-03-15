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
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetSummaryModel
from python.implementation.workflows.utils.validation import ValidationIssueModel


log = logging.getLogger(__name__)

_ROLE_COVARIATE = "covariate"
_ROLE_EFFECT_MODIFIER = "effect_modifier"


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
# Role-assignment helpers
# ---------------------------------------------------------------------
def _build_role_map_from_causal_specs(causal_specs: CausalSpec) -> dict[str, str]:
    """
    Deterministically assign roles from causal specs only.
    LLM is NOT allowed to decide whether a column is covariate or effect_modifier.
    """
    treatment_col = str(causal_specs.treatment_spec.column)
    outcome_col = str(causal_specs.outcome_spec.column)
    forbidden = {treatment_col, outcome_col}

    role_map: dict[str, str] = {}

    for col in (causal_specs.covariates or []):
        c = str(col)
        if c not in forbidden:
            role_map[c] = _ROLE_COVARIATE

    for col in (causal_specs.effect_modifiers or []):
        c = str(col)
        if c not in forbidden:
            role_map[c] = _ROLE_EFFECT_MODIFIER

    return role_map


def _force_plan_roles_from_causal_specs (
    *,
    plan: TransformPlan,
    causal_specs: CausalSpec,
) -> TransformPlan:
    """
    Keep the LLM's encoding choices, but overwrite each column role using the
    deterministic role from the causal specs.
    """
    role_map = _build_role_map_from_causal_specs(causal_specs)

    fixed_columns = []
    for col_plan in plan.columns:
        expected_role = role_map.get(str(col_plan.column))
        if expected_role is None:
            fixed_columns.append(col_plan) # pyright: ignore[reportUnknownMemberType]
            continue

        if col_plan.role != expected_role:
            log.warning(
                "Overriding LLM-assigned role for column '%s': got=%s expected=%s",
                col_plan.column,
                col_plan.role,
                expected_role,
            )

        fixed_columns.append(col_plan.model_copy(update={"role": expected_role})) # pyright: ignore[reportUnknownMemberType]

    return plan.model_copy(update={"columns": fixed_columns})

def _validate_plan_against_constraints(
    *,
    plan: TransformPlan,
    eligible_cols: set[str],
    expected_covariate_cols: set[str],
    expected_effect_modifier_cols: set[str],
    treatment_col: Optional[str],
    outcome_col: Optional[str],
) -> list[ValidationIssueModel]:

    logging.warning(
        "Validating encoding plan against constraints. Eligible cols: %s, expected covariates: %s, expected effect modifiers: %s, treatment_col: %s, outcome_col: %s. Plan columns: %s",
        eligible_cols,
        expected_covariate_cols,
        expected_effect_modifier_cols,
        treatment_col,
        outcome_col,
        [c.column for c in plan.columns],
    )
    logging.warning("Plan details: %s", plan.model_dump_json(indent=2))

    validation_issues: list[ValidationIssueModel] = []
    cols = [c.column for c in plan.columns]
    if len(cols) != len(set(cols)):
        validation_issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Encoding plan contains duplicate columns.",
                evidence={"duplicate_columns": [c for c in set(cols) if cols.count(c) > 1]},
                fix_hint="Ensure each column appears at most once in the encoding plan."
            )
        )

    forbidden = {c for c in (treatment_col, outcome_col) if c}
    illegal = sorted(set(cols) & forbidden)
    if illegal:
        validation_issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Encoding plan must not include treatment/outcome columns.",
                evidence={"illegal_columns": illegal},
                fix_hint="Remove treatment/outcome columns from the encoding plan."
            )
        )

    plan_set = set(cols)
    missing = sorted(eligible_cols - plan_set)
    extra = sorted(plan_set - eligible_cols)
    if missing:
        validation_issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Encoding plan is missing eligible columns.",
                evidence={"missing_columns": missing},
                fix_hint="Ensure all eligible columns are included in the encoding plan."
            )
        )
    if extra:
        validation_issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Encoding plan contains non-eligible columns.",
                evidence={"extra_columns": extra},
                fix_hint="Remove non-eligible columns from the encoding plan."
            )
        )

    role_by_col = {c.column: c.role for c in plan.columns}

    wrong_covariate = sorted(
        c for c in expected_covariate_cols if role_by_col.get(c) != _ROLE_COVARIATE
    )
    wrong_effect_modifier = sorted(
        c
        for c in expected_effect_modifier_cols
        if role_by_col.get(c) != _ROLE_EFFECT_MODIFIER
    )

    if wrong_covariate:
        validation_issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Encoding plan assigned wrong role for covariate columns.",
                evidence={"wrong_covariate_columns": wrong_covariate},
                fix_hint="Ensure all covariate columns are assigned role 'covariate'."
            )
        )
    if wrong_effect_modifier:
        validation_issues.append(
            ValidationIssueModel(
                severity="FAIL",
                message="Encoding plan assigned wrong role for effect_modifier columns.",
                evidence={"wrong_effect_modifier_columns": wrong_effect_modifier},
                fix_hint="Ensure all effect_modifier columns are assigned role 'effect_modifier'."
            )
        )

    return validation_issues


def _generate_encoding_plan(
    *,
    llm: LLMService,
    causal_specs: CausalSpec,
    selected_model: Any,
    dataset_summary: DatasetSummaryModel,
    prev_training_error: Optional[str] = None,
    documentation: Optional[str] = None,
    history: Optional[Sequence[ChatMessage]],
) -> tuple[UserPlanInput, Optional[TransformPlan]]:

    for _, _ in enumerate(range(2)):
        covariate_cols = set(causal_specs.covariates or [])
        effect_modifier_cols = set(causal_specs.effect_modifiers or [])

        treatment_col = str(causal_specs.treatment_spec.column)
        outcome_col = str(causal_specs.outcome_spec.column)

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
            causal_specs_json=_dumps(_safe_model_dump(causal_specs)),
            dataset_summary_json=_dumps(_safe_model_dump(dataset_summary)),
            prev_training_errors_string=prev_training_error_str,
            documentation_string=documentation_str,
        )

        last_4_messages = list(history[-4:]) if history else None

        out = llm.generate_json(
            schema=UserPlanInput,
            system_prompt=None,
            user_prompt=user_prompt_discussion,
            config=LLMConfig(model="basic", temperature=0.5),
            history=last_4_messages,
            max_attempts=2,
        )

        if out.needs_user_input:
            return out, None

        logging.warning(
            "causal specs and dataset summary for plan generation: causal_specs=%s dataset_summary=%s",
            causal_specs.model_dump_json(),
            dataset_summary.model_dump_json() if hasattr(dataset_summary, "model_dump_json") else _dumps(_safe_model_dump(dataset_summary)),
        )

        user_prompt_plan = ENCODING_PLAN_PLAN_USER_PROMPT_TEMPLATE.format(
            selected_model_json=_dumps(_safe_model_dump(selected_model)),
            causal_specs_json=_dumps(_safe_model_dump(causal_specs)),
            dataset_summary_json=_dumps(_safe_model_dump(dataset_summary)),
            prev_training_errors_string=prev_training_error_str,
            documentation_string=documentation_str,
        )

        plan = llm.generate_json(
            schema=TransformPlan,
            system_prompt=None,
            user_prompt=user_prompt_plan,
            config=LLMConfig(model="basic", temperature=0.1),
            history=last_4_messages,
            max_attempts=3,
        )

        # -------------------------------------------------------------
        # SURGICAL FIX:
        # Keep LLM-generated encodings, but force roles from causal specs.
        # -------------------------------------------------------------
        plan = _force_plan_roles_from_causal_specs(
            plan=plan,
            causal_specs=causal_specs,
        )

        validation_issues = _validate_plan_against_constraints(
            plan=plan,
            eligible_cols=eligible,
            expected_covariate_cols=covariate_cols - {treatment_col, outcome_col},
            expected_effect_modifier_cols=effect_modifier_cols - {treatment_col, outcome_col},
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )

        if validation_issues:
            log.warning(
                "Encoding plan validation issues found: %s",
                [i.model_dump_json() for i in validation_issues],
            )
            prev_training_error = (
                "The encoding plan generated by the model had the following issues:\n"
                + "\n".join(
                    f"- {i.severity}: {i.message} (evidence: {i.evidence})"
                    for i in validation_issues
                )
                + "\nPlease review the issues and adjust the encoding plan accordingly."
            )
            continue

        return out, plan

    raise ValueError(
        "Failed to generate encoding plan after multiple attempts. "
        "LLM output did not meet expected format or constraints."
    )


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
        input_extras: Optional[Mapping[str, Any]] = None,
    ) -> State:
        if not isinstance(state, ModelTrainState):
            raise ValueError(f"{self.name}: invalid state (got {type(state).__name__})")

        deps = ModelTrainDeps.from_loaded(previous_state_dependencies)

        causal_specs = deps.clean_protocol.payload.compiled_causal_spec
        assert causal_specs is not None, "Compiled causal specs must be available for model training."

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

        has_any_adjustment_cols = bool(causal_specs.covariates or []) or bool(causal_specs.effect_modifiers or [])

        # -----------------------------------------------------------------
        # Phase 1: prepare encoding plan if needed
        # -----------------------------------------------------------------
        if (
            state.payload.column_transformation_plan is None and has_any_adjustment_cols
        ):

            user_discussion, plan = _generate_encoding_plan(
                llm=self.llm,
                causal_specs=causal_specs,
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
                    }
                )
                return ModelTrainState(payload=payload)

            if plan is None:
                raise ValueError("LLM indicated no user input needed but did not return a plan.")

            return ModelTrainState(
                payload=state.payload.model_copy(
                    update={
                        "column_transformation_plan": plan,
                        "needs_user_input": False,
                        "error": None,
                        "user_message": user_discussion.message + "\n\nProceeding to training.",
                    }
                )
            )

        # -----------------------------------------------------------------
        # Phase 2: training
        # -----------------------------------------------------------------
        order_effect_modifiers: Optional[list[str]] = None
        order_covariates: Optional[list[str]] = None

        if state.payload.column_transformation_plan is not None:
            order_effect_modifiers = [
                c.column
                for c in state.payload.column_transformation_plan.columns
                if c.role == "effect_modifier"
            ]
            order_covariates = [
                c.column
                for c in state.payload.column_transformation_plan.columns
                if c.role == "covariate"
            ]

        run_id = uuid4()
        cmd = FitCommand(
            model_name=estimator_fqcn,
            dataset_id=clean_dataset_id,
            run_id=run_id,
            causal_specs=causal_specs,
            data_summary=dataset_summary,
            order_effect_modifiers=order_effect_modifiers,
            order_covariates=order_covariates,
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
                        "order_effect_modifiers": order_effect_modifiers,
                        "order_covariates": order_covariates,
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
                                "order_effect_modifiers": None,
                                "order_covariates": None,
                                "column_transformation_plan": None,
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
                        "order_effect_modifiers": None,
                        "order_covariates": None,
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
