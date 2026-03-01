from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Tuple


# --------------------------------------------------------------------------------------
# Canonical estimator identifiers (EXACT EconML fully-qualified class names)
# --------------------------------------------------------------------------------------




def get_econml_allowed_estimators() -> Tuple[str, ...]:
    """
    Return the exact fully-qualified estimator class names that the LLM is allowed to output.
    """
    return ECONML_ALLOWED_ESTIMATORS


# --------------------------------------------------------------------------------------
# EconML method notes 
# - Intentionally written as "authoritative notes" to be embedded in prompts.
# --------------------------------------------------------------------------------------

def get_econml_method_notes_broad() -> str:

    return (
        "ECONML METHOD NOTES (authoritative; derived ONLY from the provided EconML doc text;)\n"
        "\n"
        "Also use validation notes if provided (e.g., treatment type, CI requirement, dimensionality hints) to rule out candidates that are incompatible with stated facts or that have limitations that are explicitly relevant.\n"
        "A) ORTHOGONAL / DOUBLE MACHINE LEARNING (DML)\n"
        "- Setting described: observational (or experimental/A-B) data with recorded controls/confounders; "
        "controls may be high-dimensional and/or relationships may be non-parametric.\n"
        "- Core idea: reduce causal estimation to two first-stage predictive tasks, then a final-stage effect model:\n"
        "  1) Outcome nuisance: predict outcome Y from controls (and, in general, conditional expectations).\n"
        "  2) Treatment nuisance: predict treatment T from controls.\n"
        "  Then compute residuals (residual-on-residual construction) and regress residualized outcome on residualized "
        "treatment (often with an effect model indexed by features X) to learn heterogeneous effects.\n"
        "- Key property emphasized: with cross-fitting (see _OrthoLearner in the docs), certain final-stage estimators "
        "retain favorable statistical properties (e.g., MSE behavior, asymptotic normality, and confidence intervals) "
        "even when nuisance models are estimated with flexible ML, as long as nuisance estimation reaches stated rates.\n"
        "- Robustness mechanism highlighted: Neyman orthogonality of the moment equations corresponding to the final "
        "least-squares objective with respect to nuisance parameters; cross-fitting is required for the stated theorem.\n"
        "- What you estimate: the docs discuss learning a (constant marginal) CATE as a function of covariates/features "
        "X, with additional controls W.\n"
        "\n"
        "DML-related estimator classes mentioned:\n"
        "1) econml.dml.DML\n"
        "   - Effect model is linear in a featurization; allows arbitrary scikit-learn linear estimator as model_final "
        "(e.g., LassoCV, ElasticNetCV, LinearRegression variants).\n"
        "   - Final-stage fits on features derived from Kronecker-product style construction involving T and features of X.\n"
        "   - Targets settings discussed in the excerpt: linear/sparse/regularized final model families analyzed in the cited works.\n"
        "2) econml.dml.LinearDML\n"
        "   - Uses an unregularized low-dimensional final linear model.\n"
        "   - Positioned for valid confidence intervals via asymptotic normality when feature dimension is small relative to samples.\n"
        "3) econml.dml.SparseLinearDML\n"
        "   - Uses an L1-regularized/debiased-lasso-style final model (DebiasedLasso in EconML).\n"
        "   - Positioned for confidence intervals in high-dimensional sparse linear settings.\n"
        "4) econml.dml.KernelDML\n"
        "   - Variant of RKHS approach (Nie 2017 as referenced); approximates RKHS via random Fourier features.\n"
        "   - Uses ElasticNet-style regularization in final stage; assumes an RBF kernel per excerpt.\n"
        "5) econml.dml.NonParamDML\n"
        "   - Makes no parametric assumption on the effect model; applies only when treatment is binary or single-dimensional continuous.\n"
        "   - Rewrites square loss into a weighted regression formulation; model_final can be any regressor supporting sample_weight.\n"
        "   - Example families in the excerpt include forests/GBMs/SVMs, and wrappers to add sample-weight support.\n"
        "6) econml.dml.CausalForestDML\n"
        "   - Child of _RLearner in the excerpt; uses a causal forest as the final model (CausalForest implementation).\n"
        "   - Positioned as flexible non-linear CATE, with confidence intervals via Bootstrap-of-Little-Bags as described in the excerpt.\n"
        "7) econml.dml._RLearner (not allowed as output in our selection list; referenced in the excerpt)\n"
        "   - Parent pattern enabling custom final models that consume residuals and X; described as more cumbersome because of non-standard inputs.\n"
        "\n"
        "DML usage guidance in the excerpt:\n"
        "- If you want confidence intervals: LinearDML when low-dimensional X; SparseLinearDML when #features comparable to #samples.\n"
        "- If treatment is single-dimensional continuous or binary and you want non-linear models + CIs: CausalForestDML is highlighted.\n"
        "- If you have no idea how heterogeneity looks: use flexible featurizers + SparseLinearDML, or use CausalForestDML.\n"
        "- Cross-fitting parameter cv: default minimal is 2; larger (e.g., 5/6) suggested for stability in small samples (with compute cost).\n"
        "\n"
        "B) DOUBLY ROBUST LEARNING (DR)\n"
        "- Setting described: categorical (finite) treatments; all potential confounders/controls recorded.\n"
        "- Core two nuisance tasks:\n"
        "  1) Regression model: predict outcome Y from (treatment T + controls).\n"
        "  2) Propensity model: predict treatment T from controls (multi-class classification; requires predict_proba).\n"
        "- Then construct a doubly-robust pseudo-outcome / debiased target and regress it on X to learn CATE.\n"
        "- Key robustness guarantee emphasized: final error depends on the product of nuisance errors; if either nuisance is accurate "
        "enough, the final estimator is valid (per excerpt).\n"
        "- Noted downside emphasized: can have higher variance, especially when some treatments have small assignment probability "
        "(small overlap / positivity issues).\n"
        "\n"
        "DR-related estimator classes mentioned:\n"
        "1) econml.dr.DRLearner\n"
        "   - No parametric assumption on effect model; arbitrary scikit-learn models can be used for regression/propensity/final.\n"
        "   - Cross-fitting via cv.\n"
        "2) econml.dr.LinearDRLearner\n"
        "   - Unregularized low-dimensional final linear model; positioned for confidence intervals via asymptotic normality.\n"
        "3) econml.dr.SparseLinearDRLearner\n"
        "   - L1-regularized/debiased-lasso-style final model; positioned for confidence intervals in high-dimensional settings.\n"
        "4) econml.dr.ForestDRLearner\n"
        "   - Uses subsampled honest forest regressor as final model; positioned for confidence intervals via BLB per excerpt.\n"
        "\n"
        "C) FOREST-BASED ESTIMATORS (ORF vs forest-DML vs forest-DR)\n"
        "- Grouped in the excerpt as:\n"
        "  * Orthogonal Random Forests: econml.orf.DMLOrthoForest and econml.orf.DROrthoForest\n"
        "  * Forest Double Machine Learning: econml.dml.CausalForestDML\n"
        "  * Forest Doubly Robust: econml.dr.ForestDRLearner\n"
        "- Shared theme: very flexible non-linear effect heterogeneity with valid confidence intervals.\n"
        "- Key distinction emphasized:\n"
        "  * ORF methods (DMLOrthoForest / DROrthoForest): local nuisance estimation around each target X point using forest-derived "
        "similarity weights; can improve performance but adds compute cost (nuisances per target point).\n"
        "  * CausalForestDML / ForestDRLearner: global nuisance estimation; forest similarity used for effect estimation but is not "
        "coupled to nuisance estimation.\n"
        "- Treatment-type split emphasized:\n"
        "  * DMLOrthoForest supports continuous or discrete treatments (per excerpt wording).\n"
        "  * DROrthoForest applies to discrete/categorical treatments and uses DR-style moments.\n"
        "\n"
        "D) META-LEARNERS\n"
        "- In the excerpt: SLearner, TLearner, XLearner, DomainAdaptationLearner.\n"
        "- Described as black-box combinations of ML stages; used for flexibility and cross-validated model selection.\n"
        "- Noted caveat: typically do not offer valid confidence intervals because bias/variance tradeoffs of arbitrary ML are unclear.\n"
    )