from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
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


def _build_field_guide(summary: DatasetSummaryModel) -> str:
    lines: list[str] = []
    for p in summary.profiles:
        name = str(p.name).strip()
        if name:
            lines.append(f'- "{name}": {_KIND_TO_GUIDE.get(str(p.inferred_kind), "nominal")}')
    return "\n".join(lines) if lines else "(no fields)"


def _safe(v: Any) -> Any:
    """Make a value JSON-safe (NaN/Inf → None)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def _safe_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _safe(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Individual analysis runners — thin wrappers around library APIs
# ---------------------------------------------------------------------------


def _run_descriptive(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    cols = [c for c in plan.columns if c in df.columns]
    if not cols:
        raise ValueError("No valid numeric columns found in dataframe")
    subset = df[cols]
    if plan.group_by and plan.group_by in df.columns:
        desc = subset.groupby(df[plan.group_by]).describe()
        table = {str(k): _safe_dict(v) for k, v in desc.to_dict().items()}
    else:
        desc = subset.describe()
        table = {str(k): _safe_dict(v) for k, v in desc.to_dict().items()}

    return AnalyticsResultModel(
        analysis_type="descriptive",
        summary=f"Descriptive statistics for {len(cols)} column(s): {', '.join(cols)}.",
        tables={"describe": table},
    )


def _run_correlation(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    cols = [c for c in plan.columns if c in df.columns]
    if len(cols) < 2:
        raise ValueError("Need at least 2 valid numeric columns for correlation")
    corr = df[cols].corr(method="pearson")
    table = {str(k): _safe_dict(v) for k, v in corr.to_dict().items()}
    return AnalyticsResultModel(
        analysis_type="correlation",
        summary=f"Pearson correlation matrix for: {', '.join(cols)}.",
        tables={"correlation_matrix": table},
    )


def _run_linear_regression(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    import statsmodels.api as sm

    target = str(plan.target)
    predictors = [p for p in plan.predictors if p in df.columns]
    if not predictors or target not in df.columns:
        raise ValueError("Target or predictors not found in dataframe")

    work = df[[target, *predictors]].dropna()
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
            f"R²={_safe(model.rsquared)}, Adj-R²={_safe(model.rsquared_adj)}, "
            f"F={_safe(model.fvalue)}, p(F)={_safe(model.f_pvalue)}."
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
    y = work[target].astype(float)
    X = pd.get_dummies(work[predictors], drop_first=True, dtype=float)
    X = sm.add_constant(X)

    model = sm.Logit(y, X).fit(disp=False, maxiter=100)
    params = _safe_dict(model.params.to_dict())
    pvalues = _safe_dict(model.pvalues.to_dict())
    odds_ratios = _safe_dict(np.exp(model.params).to_dict())

    return AnalyticsResultModel(
        analysis_type="logistic_regression",
        summary=(
            f"Logistic regression: {target} ~ {' + '.join(predictors)}. "
            f"Pseudo-R²={_safe(model.prsquared)}, AIC={_safe(model.aic)}."
        ),
        tables={
            "coefficients": params,
            "p_values": pvalues,
            "odds_ratios": odds_ratios,
        },
        metrics={
            "pseudo_r_squared": _safe(model.prsquared),
            "aic": _safe(model.aic),
            "bic": _safe(model.bic),
            "n_obs": int(model.nobs),
        },
    )


def _run_propensity_score(df: pd.DataFrame, plan: AnalyticsPlanModel) -> AnalyticsResultModel:
    treatment = str(plan.treatment)
    covariates = [c for c in plan.covariates if c in df.columns]
    if not covariates or treatment not in df.columns:
        raise ValueError("Treatment or covariates not found in dataframe")

    work = df[[treatment, *covariates]].dropna()
    y = work[treatment].astype(int)
    X = pd.get_dummies(work[covariates], drop_first=True, dtype=float)

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
            f"Propensity scores estimated for treatment='{treatment}' using "
            f"{len(covariates)} covariate(s). "
            f"Treated mean={_safe(treated.mean()):.4f}, Control mean={_safe(control.mean()):.4f}."
        ),
        tables={
            "distribution": {
                "treated": _safe_dict(treated.describe().to_dict()),
                "control": _safe_dict(control.describe().to_dict()),
            },
        },
        metrics={
            "auc": _safe(float(clf.score(X_scaled, y))),
            "n_treated": int(treated.shape[0]),
            "n_control": int(control.shape[0]),
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
            f"χ²={_safe(chi2):.4f}, df={dof}, p={_safe(p):.6f}."
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
    t_stat, p_val = sp_stats.ttest_ind(a, b, equal_var=False)

    return AnalyticsResultModel(
        analysis_type="ttest",
        summary=(
            f"Welch's t-test: {num_col} by {grp_col} ({groups[0]} vs {groups[1]}). "
            f"t={_safe(t_stat):.4f}, p={_safe(p_val):.6f}. "
            f"Means: {_safe(a.mean()):.4f} vs {_safe(b.mean()):.4f}."
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
    "descriptive": _run_descriptive,
    "correlation": _run_correlation,
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
            "Advanced analytics tool for tabular data. Supports descriptive statistics, "
            "correlation matrices, OLS / logistic regression, propensity score estimation, "
            "chi-squared tests, and independent t-tests. Uses an LLM to plan the analysis "
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
        log.info("analysis complete", analysis_type=plan.analysis_type, summary=result.summary[:200])
        return result


__all__ = ["AdvancedAnalyticsTool", "AnalyticsPlanModel", "AnalyticsResultModel"]
