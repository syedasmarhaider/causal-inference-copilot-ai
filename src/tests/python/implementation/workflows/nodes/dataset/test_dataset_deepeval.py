from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from python.domain.service.llm_service import AvailableModelsKey, LLMConfig, LLMService
from python.implementation.service.llms.llm_service_factory import (
    LLMServiceSettings,
    make_llm_service,
)
from python.implementation.workflows.nodes.dataset.dataset_node import (
    DatasetIntentModel,
    DatasetNode,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


if not _env_truthy(os.getenv("RUN_DEEPEVAL_TESTS")):
    pytest.skip(
        "Set RUN_DEEPEVAL_TESTS=1 to run DeepEval dataset prompt tests.",
        allow_module_level=True,
    )

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
os.environ.setdefault("DEEPEVAL_CACHE_FOLDER", "/tmp/.deepeval")

pytest.importorskip("deepeval", reason="DeepEval tests require the deepeval package.")

from deepeval import assert_test
from deepeval.metrics import GEval, JsonCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

pytestmark = [pytest.mark.integration, pytest.mark.deepeval]

_JUDGE_MAX_TOKENS = 2_048


class _UnusedDataRepo:
    pass


class _UnusedDataManipulationTool:
    NAME = DataManipulationTool.NAME

    def manipulate(self, **_: object) -> pd.DataFrame:
        raise AssertionError("data manipulation should not be used in prompt evals")


class _UnusedPlotTool:
    NAME = PlotTool.NAME

    def generate_specs(self, **_: object) -> list[dict[str, object]]:
        raise AssertionError("plot generation should not be used in prompt evals")


class _PromptEvalToolFactory:
    def __init__(self) -> None:
        self._profiling_tool = DatasetProfilingTool()
        self._data_manipulation_tool = _UnusedDataManipulationTool()
        self._plot_tool = _UnusedPlotTool()

    def get_tool(self, name: str) -> object:
        if name == DataManipulationTool.NAME:
            return self._data_manipulation_tool
        if name == PlotTool.NAME:
            return self._plot_tool
        if name == DatasetProfilingTool.NAME:
            return self._profiling_tool
        raise KeyError(name)


class _ProjectDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        *,
        service: LLMService,
        model_alias: AvailableModelsKey = "basic",
    ) -> None:
        self._service = service
        self._model_alias = model_alias
        super().__init__(model=f"project/{model_alias}")

    def load_model(self) -> LLMService:
        return self._service

    def generate(
        self,
        prompt: object,
        *,
        schema: type[BaseModel] | None = None,
        **_: object,
    ) -> str | BaseModel:
        normalized_prompt = _normalize_prompt(prompt)
        config = LLMConfig(
            model=self._model_alias,
            temperature=0.0,
            top_p=1.0,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        if schema is not None:
            return self._service.generate_json(
                schema=schema,
                system_prompt=None,
                user_prompt=normalized_prompt,
                config=config,
                history=None,
                max_attempts=2,
            )
        return self._service.generate(
            system_prompt=None,
            user_prompt=normalized_prompt,
            config=config,
            history=None,
        ).content

    async def a_generate(
        self,
        prompt: object,
        *,
        schema: type[BaseModel] | None = None,
        **kwargs: object,
    ) -> str | BaseModel:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return f"project/{self._model_alias}"

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_json_mode(self) -> bool:
        return True


def _normalize_prompt(prompt: object) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, BaseModel):
        return prompt.model_dump_json()
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str)


def _make_geval(
    *,
    name: str,
    model: DeepEvalBaseLLM,
    evaluation_params: Sequence[LLMTestCaseParams],
    evaluation_steps: list[str],
    threshold: float = 0.75,
) -> GEval:
    return GEval(
        name=name,
        model=model,
        evaluation_params=list(evaluation_params),
        evaluation_steps=evaluation_steps,
        threshold=threshold,
        async_mode=False,
    )


def _build_summary_json(dataframe: pd.DataFrame) -> str:
    profiling_tool = DatasetProfilingTool()
    summary = profiling_tool.extract_dataset_summary(
        dataframe,
        max_categories=200,
        sample_distinct=200,
        compute_quantiles=False,
        strict=True,
    )
    return profiling_tool.dataset_summary_to_json(summary)


@pytest.fixture(scope="module")
def llm_service() -> Iterator[LLMService]:
    try:
        service = make_llm_service(
            LLMServiceSettings(
                provider="auto",
                timeout_s=90.0,
                hard_deadline_s=90.0,
                max_retries=1,
                executor_workers=2,
            )
        )
    except ValueError as exc:
        pytest.skip(f"DeepEval tests require configured LLM credentials: {exc}")

    yield service

    close = getattr(service, "close", None)
    if callable(close):
        close()


@pytest.fixture(scope="module")
def judge_model(llm_service: LLMService) -> _ProjectDeepEvalLLM:
    return _ProjectDeepEvalLLM(service=llm_service, model_alias="basic")


@pytest.fixture(scope="module")
def dataset_node(llm_service: LLMService) -> DatasetNode:
    return DatasetNode(
        data_repo=_UnusedDataRepo(),
        llm=llm_service,
        tools_factory=_PromptEvalToolFactory(),
    )


@pytest.fixture(scope="module")
def dataset_summary_json() -> str:
    return _build_summary_json(
        pd.DataFrame(
            [
                {
                    "btransf": 1,
                    "outcome_status": 1,
                    "age": 65,
                    "sex": "female",
                    "site": "A",
                    "extra_flag": "keep?",
                },
                {
                    "btransf": 0,
                    "outcome_status": 0,
                    "age": 72,
                    "sex": "male",
                    "site": "B",
                    "extra_flag": "drop",
                },
                {
                    "btransf": 1,
                    "outcome_status": 1,
                    "age": 58,
                    "sex": "female",
                    "site": "A",
                    "extra_flag": "drop",
                },
            ]
        )
    )


def test_dataset_intent_classifier_rejects_downstream_only_requests(
    dataset_node: DatasetNode,
    dataset_summary_json: str,
    judge_model: DeepEvalBaseLLM,
) -> None:
    latest_user_message = (
        "Train a causal model for treatment effect estimation and pick the best learner."
    )
    intent = dataset_node._classify_intent(
        latest_user_message=latest_user_message,
        chat_history=None,
        dataset_summary=dataset_summary_json,
    )

    expected = DatasetIntentModel(
        intent_data_question=False,
        intent_data_question_brief="",
        intent_manupulation_question=False,
        intent_manupulation_question_brief="",
        intent_manupulation_is_analytical_query=False,
        intent_chart=False,
        intent_chart_brief="",
    )
    test_case = LLMTestCase(
        name="dataset_intent_downstream_only",
        input=json.dumps(
            {
                "latest_user_message": latest_user_message,
                "chat_history": None,
                "dataset_summary": dataset_summary_json,
            },
            ensure_ascii=False,
        ),
        actual_output=intent.model_dump_json(),
        expected_output=expected.model_dump_json(),
        context=[dataset_summary_json],
    )

    assert_test(
        test_case,
        [
            JsonCorrectnessMetric(
                expected_schema=DatasetIntentModel,
                model=judge_model,
                include_reason=False,
                async_mode=False,
            ),
            _make_geval(
                name="Dataset Intent Off Topic",
                model=judge_model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                evaluation_steps=[
                    "Read the user request in the input and determine whether it belongs to "
                    "dataset inspection, dataset transformation, or charting.",
                    "Compare the actual JSON against the expected JSON and confirm all intent "
                    "booleans match.",
                    "Confirm every brief is empty when its corresponding intent is false.",
                    "Fail the evaluation if a downstream-only modeling request is classified as "
                    "a dataset-stage intent.",
                ],
                threshold=0.8,
            ),
        ],
        run_async=False,
    )


def test_dataset_intent_classifier_detects_analytical_chart_combo(
    dataset_node: DatasetNode,
    dataset_summary_json: str,
    judge_model: DeepEvalBaseLLM,
) -> None:
    latest_user_message = "Show outcome counts by sex and plot that as a bar chart."
    intent = dataset_node._classify_intent(
        latest_user_message=latest_user_message,
        chat_history=None,
        dataset_summary=dataset_summary_json,
    )

    expected = DatasetIntentModel(
        intent_data_question=False,
        intent_data_question_brief="",
        intent_manupulation_question=True,
        intent_manupulation_question_brief="show outcome counts by sex",
        intent_manupulation_is_analytical_query=True,
        intent_chart=True,
        intent_chart_brief="plot outcome counts by sex as a bar chart",
    )
    test_case = LLMTestCase(
        name="dataset_intent_analytics_plus_chart",
        input=json.dumps(
            {
                "latest_user_message": latest_user_message,
                "chat_history": None,
                "dataset_summary": dataset_summary_json,
            },
            ensure_ascii=False,
        ),
        actual_output=intent.model_dump_json(),
        expected_output=expected.model_dump_json(),
        context=[dataset_summary_json],
    )

    assert_test(
        test_case,
        [
            JsonCorrectnessMetric(
                expected_schema=DatasetIntentModel,
                model=judge_model,
                include_reason=False,
                async_mode=False,
            ),
            _make_geval(
                name="Dataset Intent Analytical Chart",
                model=judge_model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                evaluation_steps=[
                    "Check whether the request requires grouped aggregation or SQL-style "
                    "querying rather than summary-only answering.",
                    "Confirm the actual JSON marks manipulation true, "
                    "intent_manupulation_is_analytical_query true, and chart true.",
                    "Confirm the data-question intent remains false because the request needs "
                    "analytical reshaping rather than a summary-only answer.",
                    "Confirm the manipulation and chart briefs are non-empty and cover the "
                    "requested grouped counts and bar chart.",
                ],
                threshold=0.8,
            ),
        ],
        run_async=False,
    )


def test_dataset_summary_answers_stay_grounded_when_summary_is_insufficient(
    dataset_node: DatasetNode,
    dataset_summary_json: str,
    judge_model: DeepEvalBaseLLM,
) -> None:
    question = "Which patient had the highest age, and what row number were they on?"
    answer = dataset_node._answer_summary_question(
        intent_brief=question,
        dataset_summary=dataset_summary_json,
        chat_history=None,
    )

    test_case = LLMTestCase(
        name="dataset_summary_grounded_insufficient",
        input=question,
        actual_output=answer,
        context=[dataset_summary_json],
    )

    assert_test(
        test_case,
        [
            _make_geval(
                name="Dataset Summary Groundedness",
                model=judge_model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.CONTEXT,
                ],
                evaluation_steps=[
                    "Use only the dataset summary provided in the context when judging the answer.",
                    "Fail the answer if it invents patient identity, row number, or other row-level "
                    "facts not present in the summary.",
                    "Reward answers that explicitly say the summary is insufficient for row-level "
                    "detail while still being direct and helpful.",
                    "Fail the answer if it overstates certainty or pretends to inspect records.",
                ],
                threshold=0.8,
            )
        ],
        run_async=False,
    )


def test_dataset_protocol_cleaning_instructions_stay_sql_scoped(
    dataset_node: DatasetNode,
    dataset_summary_json: str,
    judge_model: DeepEvalBaseLLM,
) -> None:
    protocol_discussion = (
        "Confirmed protocol: treatment is btransf, outcome is outcome_status, and covariates "
        "are age and sex. Keep only those columns in the final cleaned dataset. Normalize "
        "outcome_status to binary 0/1 values. Drop rows where btransf or outcome_status is "
        "missing. Do not generate charts or do any modeling yet."
    )
    actual_output = dataset_node._build_protocol_cleaning_instructions(
        protocol_discussion=protocol_discussion,
        dataset_summary=dataset_summary_json,
        recent_chat_history=(
            "user: keep the cleaning concrete\nassistant: will translate protocol to SQL work"
        ),
    )

    test_case = LLMTestCase(
        name="dataset_protocol_cleaning_sql_scope",
        input=json.dumps(
            {
                "protocol_discussion": protocol_discussion,
                "dataset_summary": dataset_summary_json,
                "recent_chat_history": (
                    "user: keep the cleaning concrete\n"
                    "assistant: will translate protocol to SQL work"
                ),
            },
            ensure_ascii=False,
        ),
        actual_output=actual_output,
        expected_output=(
            "Return executable data-cleaning instructions only. Keep only btransf, "
            "outcome_status, age, and sex. Normalize outcome_status to 0/1. Drop rows with "
            "missing btransf or outcome_status. Exclude charts, plotting, and modeling."
        ),
        context=[dataset_summary_json],
    )

    assert_test(
        test_case,
        [
            _make_geval(
                name="Dataset Protocol Cleaning Instructions",
                model=judge_model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                evaluation_steps=[
                    "Confirm the output is plain user-intent text for data cleaning or "
                    "transformation, not JSON or markdown.",
                    "Confirm every instruction is grounded in the confirmed protocol discussion "
                    "and stays within SQL-like cleaning or filtering work.",
                    "Confirm the output keeps only the protocol-scope columns and includes the "
                    "requested normalization and missing-data rules.",
                    "Fail the output if it introduces charts, modeling, causal estimation, or "
                    "other tasks outside dataset cleaning.",
                ],
                threshold=0.8,
            )
        ],
        run_async=False,
    )


def test_dataset_final_response_defers_downstream_modeling_requests(
    dataset_node: DatasetNode,
    judge_model: DeepEvalBaseLLM,
) -> None:
    payload = {
        "summary_answer": None,
        "manipulation_result": {
            "status": "dataset_updated",
            "instruction": "Drop duplicate rows and standardize outcome_status.",
            "new_dataset_id": "11111111-1111-1111-1111-111111111111",
            "result": {"rows": 120, "columns": ["btransf", "outcome_status", "age", "sex"]},
        },
        "chart_result": None,
        "dataset_context": {
            "dataset_loaded_this_turn": False,
            "original_user_message": (
                "Drop duplicate rows, standardize outcome_status, and then train a model."
            ),
            "handled_intents": {
                "data_question": False,
                "manipulation_question": True,
                "chart": False,
            },
            "active_dataset_rows": 120,
            "active_dataset_columns": ["btransf", "outcome_status", "age", "sex"],
        },
    }
    actual_output = dataset_node._build_final_message(**payload)

    test_case = LLMTestCase(
        name="dataset_final_response_defers_modeling",
        input=json.dumps(payload, ensure_ascii=False),
        actual_output=actual_output,
        expected_output=(
            "Explain that the dataset transformation was completed and a new working dataset was "
            "saved, while clearly noting that downstream model training should happen later."
        ),
    )

    assert_test(
        test_case,
        [
            _make_geval(
                name="Dataset Final Response Scope",
                model=judge_model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                evaluation_steps=[
                    "Confirm the response is concise, user-facing, and merges the handled dataset "
                    "work into one answer.",
                    "Confirm it mentions the saved dataset update because the manipulation result "
                    "status is dataset_updated.",
                    "Confirm it does not pretend to have trained a model and instead says the "
                    "downstream modeling part should happen later.",
                    "Fail the response if it mentions internal JSON fields or implementation details.",
                ],
                threshold=0.8,
            )
        ],
        run_async=False,
    )
