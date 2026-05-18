from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from python.domain.service.llm_service import AvailableModelsKey, LLMConfig, LLMService
from python.domain.workflows.tool import Tool
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_models import (
    AnalyticsPlanModel,
    AnalyticsResultModel,
)
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_prompts import (
    ANALYTICS_PLAN_SYSTEM_PROMPT,
    ANALYTICS_PLAN_USER_PROMPT_TEMPLATE,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

log = get_app_logger(__name__, component="advanced_analytics_tool", log_type="tool")

_KIND_TO_GUIDE: dict[str, str] = {
    "NUMERIC": "quantitative",
    "DATETIME": "temporal",
    "CATEGORICAL": "nominal",
    "BOOLEAN": "nominal",
    "OTHER": "nominal",
}

_BINARY_FALSE_TOKENS = frozenset(
    {"0", "false", "f", "no", "n", "control", "untreated", "unexposed", "absent"}
)
_BINARY_TRUE_TOKENS = frozenset(
    {"1", "true", "t", "yes", "y", "treated", "exposed", "present", "case"}
)


def _build_field_guide(summary: DatasetSummaryModel) -> str:
    lines: list[str] = []
    for p in summary.profiles:
        name = str(p.name).strip()
        if name:
            field_kind = _KIND_TO_GUIDE.get(str(p.inferred_kind), "nominal")
            distinct = f", distinct={int(p.distinct_count)}" if p.distinct_count is not None else ""
            lines.append(f'- "{name}": {field_kind}{distinct}')
    return "\n".join(lines) if lines else "(no fields)"


def _safe(v: Any) -> Any:
    """Make a value JSON-safe (NaN/Inf → None)."""
    if isinstance(v, np.generic):
        return _safe(v.item())
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _safe_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _safe(v) for k, v in d.items()}


def _format_stat(v: Any, *, digits: int = 4) -> str:
    safe_value = _safe(v)
    if safe_value is None:
        return "n/a"
    if isinstance(safe_value, bool):
        return str(safe_value)
    if isinstance(safe_value, (int, float)):
        return f"{float(safe_value):.{digits}f}"
    return str(safe_value)


def _format_column_list(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "(none)"


def _binary_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _binary_order_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))

    numeric_value: float | None = None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None

    if numeric_value is not None:
        return (1, numeric_value)

    token = _binary_label(value).casefold()
    if token in _BINARY_FALSE_TOKENS:
        return (2, 0)
    if token in _BINARY_TRUE_TOKENS:
        return (2, 1)
    return (3, token)


def _encode_binary_series(
    series: pd.Series,
    *,
    column_name: str,
    role: str,
) -> tuple[pd.Series, dict[str, str]]:
    non_null = series.dropna()
    unique_values = list(non_null.unique())
    if len(unique_values) != 2:
        raise ValueError(
            f"{role.capitalize()} column '{column_name}' must be binary after dropping missing "
            f"values, but found {len(unique_values)} distinct values."
        )

    ordered_values = sorted(unique_values, key=_binary_order_key)
    mapping = {ordered_values[0]: 0, ordered_values[1]: 1}
    encoded = series.map(mapping)
    if encoded.isna().any():
        raise ValueError(
            f"Could not consistently encode binary values for {role} column '{column_name}'."
        )

    return (
        encoded.astype(int),
        {
            "0": _binary_label(ordered_values[0]),
            "1": _binary_label(ordered_values[1]),
        },
    )


# ---------------------------------------------------------------------------
# Individual analysis runners — thin wrappers around library APIs
# ---------------------------------------------------------------------------
def _run_linear_regression(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    import statsmodels.api as sm

    target = str(plan.target)
    predictors = [p for p in plan.predictors if p in df.columns]
    if not predictors or target not in df.columns:
        raise ValueError("Target or predictors not found in dataframe")

    work = df[[target, *predictors]].dropna()
    if work.empty:
        raise ValueError(
            "No complete cases remain for linear regression after dropping missing values."
        )
    y = work[target].astype(float)
    X = pd.get_dummies(work[predictors], drop_first=True, dtype=float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    params = _safe_dict(model.params.to_dict())
    pvalues = _safe_dict(model.pvalues.to_dict())
    conf = {str(k): [_safe(lo), _safe(hi)] for k, (lo, hi) in model.conf_int().iterrows()}

    return AnalyticsResultModel(
        analysis_type="linear_regression",
        summary=(
            f"OLS regression: {target} ~ {' + '.join(predictors)}. "
            f"R²={_format_stat(model.rsquared)}, Adj-R²={_format_stat(model.rsquared_adj)}, "
            f"F={_format_stat(model.fvalue)}, p(F)={_format_stat(model.f_pvalue)}."
        ),
        tables={"coefficients": params, "p_values": pvalues, "conf_int_95": conf},
        metrics={
            "r_squared": _safe(model.rsquared),
            "adj_r_squared": _safe(model.rsquared_adj),
            "f_statistic": _safe(model.fvalue),
            "f_pvalue": _safe(model.f_pvalue),
            "n_obs": int(model.nobs),
        },
    )


def _run_logistic_regression(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    import statsmodels.api as sm

    target = str(plan.target)
    predictors = [p for p in plan.predictors if p in df.columns]
    if not predictors or target not in df.columns:
        raise ValueError("Target or predictors not found in dataframe")

    work = df[[target, *predictors]].dropna()
    if work.empty:
        raise ValueError(
            "No complete cases remain for logistic regression after dropping missing values."
        )
    y, target_levels = _encode_binary_series(work[target], column_name=target, role="target")
    X = pd.get_dummies(work[predictors], drop_first=True, dtype=float)
    if X.shape[1] == 0:
        raise ValueError(
            "The selected predictors do not produce any usable encoded features for logistic regression."
        )
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False, maxiter=100)
    params = _safe_dict(model.params.to_dict())
    pvalues = _safe_dict(model.pvalues.to_dict())
    odds_ratios = _safe_dict(np.exp(model.params).to_dict())

    return AnalyticsResultModel(
        analysis_type="logistic_regression",
        summary=(
            f"Logistic regression: {target} ~ {' + '.join(predictors)}. "
            f"Positive class='{target_levels['1']}'. "
            f"Pseudo-R²={_format_stat(model.prsquared)}, AIC={_format_stat(model.aic)}."
        ),
        tables={
            "coefficients": params,
            "p_values": pvalues,
            "odds_ratios": odds_ratios,
            "target_levels": target_levels,
        },
        metrics={
            "pseudo_r_squared": _safe(model.prsquared),
            "aic": _safe(model.aic),
            "bic": _safe(model.bic),
            "n_obs": int(model.nobs),
            "target_levels": target_levels,
        },
    )


def _run_propensity_score(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    treatment = str(plan.treatment)
    requested_covariates = list(dict.fromkeys(str(c) for c in plan.covariates))
    if treatment in requested_covariates:
        raise ValueError(
            f"Treatment column '{treatment}' cannot also be used as a propensity score covariate."
        )

    covariates = [c for c in requested_covariates if c in df.columns]
    if not covariates or treatment not in df.columns:
        raise ValueError("Treatment or covariates not found in dataframe")

    work = df[[treatment, *covariates]].dropna()
    if work.empty:
        raise ValueError(
            "No complete cases remain for propensity score estimation after dropping missing values."
        )
    y, treatment_levels = _encode_binary_series(
        work[treatment], column_name=treatment, role="treatment"
    )
    X = pd.get_dummies(work[covariates], drop_first=True, dtype=float)
    if X.shape[1] == 0:
        raise ValueError(
            "The selected covariates do not produce any usable encoded features for propensity "
            "score estimation."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_scaled, y)
    p_scores = clf.predict_proba(X_scaled)[:, 1]

    p_series = pd.Series(p_scores, index=work.index)
    treated = p_series[y == 1]
    control = p_series[y == 0]

    return AnalyticsResultModel(
        analysis_type="propensity_score",
        summary=(
            f"Propensity scores estimated for treatment='{treatment}' "
            f"(positive class='{treatment_levels['1']}') using covariates: "
            f"{_format_column_list(covariates)}. "
            f"Treated mean={_format_stat(treated.mean())}, Control mean={_format_stat(control.mean())}."
        ),
        tables={
            "distribution": {
                "treated": _safe_dict(treated.describe().to_dict()),
                "control": _safe_dict(control.describe().to_dict()),
            },
            "treatment_levels": treatment_levels,
        },
        metrics={
            "auc": _safe(float(roc_auc_score(y, p_scores))),
            "n_treated": int(treated.shape[0]),
            "n_control": int(control.shape[0]),
            "n_obs": int(work.shape[0]),
            "treatment": treatment,
            "covariates": covariates,
            "treatment_levels": treatment_levels,
        },
    )


def _run_chi_squared(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    col_a, col_b = str(plan.column_a), str(plan.column_b)
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError(f"Columns '{col_a}' or '{col_b}' not found")

    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, expected = sp_stats.chi2_contingency(ct)

    return AnalyticsResultModel(
        analysis_type="chi_squared",
        summary=(
            f"Chi-squared test: {col_a} × {col_b}. "
            f"χ²={_format_stat(chi2)}, df={dof}, p={_format_stat(p, digits=6)}."
        ),
        tables={
            "contingency": {str(k): _safe_dict(v) for k, v in ct.to_dict().items()},
        },
        metrics={"chi2": _safe(chi2), "p_value": _safe(p), "dof": int(dof)},
    )


def _run_ttest(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    num_col = str(plan.numeric_column)
    grp_col = str(plan.group_column)
    if num_col not in df.columns or grp_col not in df.columns:
        raise ValueError(f"Columns '{num_col}' or '{grp_col}' not found")

    work = df[[num_col, grp_col]].dropna()
    groups = sorted(work[grp_col].unique())
    if len(groups) != 2:
        raise ValueError(f"t-test requires exactly 2 groups, found {len(groups)}")

    a = work.loc[work[grp_col] == groups[0], num_col].astype(float)
    b = work.loc[work[grp_col] == groups[1], num_col].astype(float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            "t-test requires at least 2 observations in each group after dropping missing values."
        )
    t_stat, p_val = sp_stats.ttest_ind(a, b, equal_var=False)

    return AnalyticsResultModel(
        analysis_type="ttest",
        summary=(
            f"Welch's t-test: {num_col} by {grp_col} ({groups[0]} vs {groups[1]}). "
            f"t={_format_stat(t_stat)}, p={_format_stat(p_val, digits=6)}. "
            f"Means: {_format_stat(a.mean())} vs {_format_stat(b.mean())}."
        ),
        tables={
            "group_stats": {
                str(groups[0]): _safe_dict(a.describe().to_dict()),
                str(groups[1]): _safe_dict(b.describe().to_dict()),
            }
        },
        metrics={
            "t_statistic": _safe(t_stat),
            "p_value": _safe(p_val),
            "mean_diff": _safe(a.mean() - b.mean()),
        },
    )


_RUNNERS = {
    "linear_regression": _run_linear_regression,
    "logistic_regression": _run_logistic_regression,
    "propensity_score": _run_propensity_score,
    "chi_squared": _run_chi_squared,
    "ttest": _run_ttest,
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdvancedAnalyticsTool(Tool):
    NAME: ClassVar[str] = "ADVANCED_ANALYTICS"

    llm: LLMService
    model: AvailableModelsKey = "basic"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Advanced analytics tool for tabular data. Supports OLS / logistic regression, "
            "propensity score estimation, chi-squared tests, and independent t-tests. Uses an LLM to plan the analysis "
            "from a natural-language request, then executes it with statsmodels / scipy / sklearn."
        )

    def analyze(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: DatasetSummaryModel,
        user_request: str,
        max_attempts: int = 3,
    ) -> AnalyticsResultModel:
        request = user_request.strip()
        if not request:
            raise ValueError("user_request must be non-empty")

        field_guide = _build_field_guide(data_summary)
        user_prompt = ANALYTICS_PLAN_USER_PROMPT_TEMPLATE.format(
            user_request=request,
            n_rows=data_summary.n_rows,
            field_guide=field_guide,
        )

        log.info("planning advanced analytics", user_request=request[:120])
        plan = self.llm.generate_json(
            schema=AnalyticsPlanModel,
            system_prompt=ANALYTICS_PLAN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            config=LLMConfig(model=self.model, temperature=0.0, top_p=1.0),
            history=None,
            max_attempts=max_attempts,
        )

        runner = _RUNNERS.get(plan.analysis_type)
        if runner is None:
            raise ValueError(f"Unknown analysis type: {plan.analysis_type}")

        log.info("executing analysis", analysis_type=plan.analysis_type)
        result = runner(dataframe, plan)
        log.info(
            "analysis complete", analysis_type=plan.analysis_type, summary=result.summary[:200]
        )
        return result


__all__ = ["AdvancedAnalyticsTool", "AnalyticsPlanModel", "AnalyticsResultModel"]
