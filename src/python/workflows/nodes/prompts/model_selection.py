# src/python/prompts/model_selection.py
from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Tuple


# --------------------------------------------------------------------------------------
# Canonical estimator identifiers (EXACT EconML fully-qualified class names)
# --------------------------------------------------------------------------------------

ECONML_ALLOWED_ESTIMATORS: Tuple[str, ...] = (
    # DML-family
    "econml.dml.DML",
    "econml.dml.LinearDML",
    "econml.dml.SparseLinearDML",
    "econml.dml.KernelDML",
    "econml.dml.NonParamDML",
    "econml.dml.CausalForestDML",
    # DR-family
    "econml.dr.DRLearner",
    "econml.dr.LinearDRLearner",
    "econml.dr.SparseLinearDRLearner",
    "econml.dr.ForestDRLearner",
    # ORF-family
    "econml.orf.DMLOrthoForest",
    "econml.orf.DROrthoForest",
    # Meta-learners
    "econml.metalearners.SLearner",
    "econml.metalearners.TLearner",
    "econml.metalearners.XLearner",
    "econml.metalearners.DomainAdaptationLearner",
)


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

# --------------------------------------------------------------------------------------
# Prompt templates (as Templates with $PLACEHOLDERS)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptInputs:
    inference_ready_state_summary: str
    dataset_summary: str
    protocol_state: str
    # Optional: if you want to include draft output from prompt 1 into prompt 2
    paste_from_previous_step: str = ""
    final_selection_json: str = ""


def get_model_selection_prompt_1_template() -> str:
    """
    Prompt 1: strict selection proposal (NO outside knowledge).
    Includes BROADER EconML notes (from provided excerpt).
    """
    econml_notes = get_econml_method_notes_broad()

    return Template(
        "You are selecting an EconML CATE/HTE estimator class based ONLY on:\n"
        "1) INFERENCE_READY_STATE_SUMMARY (authoritative)\n"
        "2) DATASET_SUMMARY (authoritative)\n"
        "3) PROTOCOL_STATE (authoritative)\n"
        "4) ECONML METHOD NOTES BELOW (authoritative; derived from provided library text;)\n"
        "\n"
        "STRICT RULES\n"
        "- Do NOT use any outside knowledge. Do NOT invent dataset facts. Do NOT invent requirements.\n"
        "- Do NOT propose any estimator not in the Allowed Estimator Classes list below.\n"
        "- If key info is missing (e.g., treatment type), explicitly mark it as UNKNOWN and keep multiple branches.\n"
        "\n"
        "ALLOWED ESTIMATOR CLASSES (use EXACT names/paths)\n"
        + "\n".join(f"- {fqcn}" for fqcn in ECONML_ALLOWED_ESTIMATORS)
        + "\n\n"
        + econml_notes
        + "\n"
        "INPUTS\n"
        "INFERENCE_READY_STATE_SUMMARY:\n"
        "$INFERENCE_READY_STATE_SUMMARY\n\n"
        "DATASET_SUMMARY:\n"
        "$DATASET_SUMMARY\n\n"
        "PROTOCOL_STATE:\n"
        "$PROTOCOL_STATE\n\n"
        "TASK\n"
        "1) Extract ONLY the decision-relevant fields from the three inputs:\n"
        "   - treatment variable type (binary / continuous / categorical / multi-treatment) if stated\n"
        "   - outcome type/dimension if stated\n"
        "   - which features are X (heterogeneity) vs W (controls/confounders) if stated\n"
        "   - sample size and feature dimensionality hints if stated\n"
        "   - whether confidence intervals / inference is required (explicitly stated or not)\n"
        "   - any constraints stated in protocol (estimand, interpretability, etc.)\n"
        "2) Propose a DRAFT “top 3” candidate estimators from the allowed list.\n"
        "3) For each candidate, justify selection ONLY with:\n"
        "   - which notes above it matches (DML vs DR vs ORF vs META)\n"
        "   - which constraints from the three inputs it fits\n"
        "   - what limitations are explicitly relevant from the notes (e.g., categorical treatment for DR; "
        "binary/single-dim for NonParamDML; low-dim X for LinearDML; high-dim X for SparseLinearDML; CI notes)\n"
        "\n"
        "OUTPUT FORMAT (structured; do not include anything else)\n"
        "A) EXTRACTED_FACTS:\n"
        "- bullet list of extracted fields, each tagged with its source: [INFERENCE] / [DATASET] / [PROTOCOL]\n"
        "B) CANDIDATES_DRAFT (ranked 1..3):\n"
        "- rank:\n"
        "  estimator_fqcn: \"econml....ClassName\"\n"
        "  family: \"DML\"|\"DR\"|\"ORF\"|\"META\"\n"
        "  fit_rationale: 3-6 bullets grounded in EXTRACTED_FACTS + METHOD NOTES\n"
        "  potential_mismatch: 1-3 bullets (only from METHOD NOTES + missing/unknown facts)\n"
        "C) MISSING_INFO (only if needed):\n"
        "- minimal list of missing fields that block a confident choice\n"
    ).template


def get_model_selection_prompt_1(inputs: PromptInputs) -> str:
    return Template(get_model_selection_prompt_1_template()).substitute(
        INFERENCE_READY_STATE_SUMMARY=inputs.inference_ready_state_summary,
        DATASET_SUMMARY=inputs.dataset_summary,
        PROTOCOL_STATE=inputs.protocol_state,
    )


def get_model_selection_prompt_2_template() -> str:
    """
    Prompt 2: refutation/finalization. You asked to allow outside knowledge here if desired.
    We keep EconML notes OUT of this prompt to avoid duplication/noise; instead we hard-restrict
    the allowed estimator list and require strict grounding + explicit [OUTSIDE] tagging if used.
    """
    outside_rule = (
        "- You MAY use outside knowledge/research for critique only. If you do, tag each such claim with [OUTSIDE].\n"
        "- You MUST still prioritize contradictions/compatibility checks that follow from the provided facts.\n"
    )

    return Template(
        "You are performing a strict second-pass critique of a draft EconML estimator shortlist.\n" 
        "\n"
        "STRICT RULES\n"
        "- Use ONLY the content provided in:\n"
        "  (1) PASTE_FROM_PREVIOUS_STEP (EXTRACTED_FACTS + CANDIDATES_DRAFT)\n"
        "  (2) The Allowed Estimator Classes list below\n"
        + outside_rule +  
        "- Do NOT output anything except the FINAL JSON object.\n"
        "\n"
        "ALLOWED ESTIMATOR CLASSES (exact; do not deviate)\n"
        + "\n".join(f"- {fqcn}" for fqcn in ECONML_ALLOWED_ESTIMATORS)
        + "\n\n"
        "INPUT\n"
        "PASTE_FROM_PREVIOUS_STEP:\n"
        "$PASTE_FROM_PREVIOUS_STEP\n\n"
        "TASK\n"
        "1) For each draft candidate, attempt to refute it using:\n"
        "   - explicit treatment-type compatibility (categorical vs continuous vs binary vs single-dimensional)\n"
        "   - explicit dimensionality constraints if present in facts (e.g., low-dim X vs high-dim X)\n"
        "   - overlap/positivity risk if explicitly stated in facts (variance concerns)\n"
        "   - inference/CI requirement if explicitly stated in facts\n"
        "2) If key facts are UNKNOWN, keep candidates that cover plausible branches; do NOT over-commit.\n"
        "3) Output FINAL TOP-3 with EXACT fully qualified class names.\n"
        "\n"
        "OUTPUT (JSON ONLY; no markdown; no commentary)\n"
        "{\n"
        "  \"selected_top3\": [\n"
        "    \"econml....ClassName\",\n"
        "    \"econml....ClassName\",\n"
        "    \"econml....ClassName\"\n"
        "  ],\n"
        "  \"selection_notes\": [\n"
        "    \"short grounded note 1\",\n"
        "    \"short grounded note 2\",\n"
        "    \"short grounded note 3\"\n"
        "  ],\n"
        "  \"rejected\": [\n"
        "    {\"estimator_fqcn\": \"econml....ClassName\", \"reason\": \"grounded reason (+[OUTSIDE] tags if used)\"}\n"
        "  ],\n"
        "  \"unknowns\": [\n"
        "    \"blocking unknown that materially affects ranking (if any)\"\n"
        "  ]\n"
        "}\n"
    ).template


def get_model_selection_prompt_2(inputs: PromptInputs) -> str:
    return Template(get_model_selection_prompt_2_template()).substitute(
        PASTE_FROM_PREVIOUS_STEP=inputs.paste_from_previous_step
    )


def get_model_selection_prompt_3_template() -> str:
    """
    Prompt 3: rationale summary (paper-ready). Keep it grounded; no new facts.
    """
    return Template(
        "You are writing a concise justification for why the final top-3 EconML estimators were selected.\n"
        "INPUTS\n"
        "INFERENCE_READY_STATE_SUMMARY:\n"
        "$INFERENCE_READY_STATE_SUMMARY\n\n"
        "DATASET_SUMMARY:\n"
        "$DATASET_SUMMARY\n\n"
        "PROTOCOL_STATE:\n"
        "$PROTOCOL_STATE\n\n"
        "FINAL_SELECTION_JSON:\n"
        "$FINAL_SELECTION_JSON\n\n"
        "TASK\n"
        "1) Briefly restate the selection context using only explicit facts from the three summaries "
        "(treatment type, outcome, X vs W roles, sample/feature scale if provided, CI requirement if provided).\n"
        "2) For each of the three selected estimators:\n"
        "   - explain briefly why it matches the stated setting\n"
        "   - state one limitation/caveat that is explicitly relevant (do NOT speculate)\n"
        "3) Do not mention any estimator not in FINAL_SELECTION_JSON.\n"
        "\n"
    ).template


def get_model_selection_prompt_3(inputs: PromptInputs) -> str:
    return Template(get_model_selection_prompt_3_template()).substitute(
        INFERENCE_READY_STATE_SUMMARY=inputs.inference_ready_state_summary,
        DATASET_SUMMARY=inputs.dataset_summary,
        PROTOCOL_STATE=inputs.protocol_state,
        FINAL_SELECTION_JSON=inputs.final_selection_json,
    )
