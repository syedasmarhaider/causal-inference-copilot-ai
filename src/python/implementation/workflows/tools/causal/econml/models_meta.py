from __future__ import annotations

SupportedModelsLiteral = [
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
] 