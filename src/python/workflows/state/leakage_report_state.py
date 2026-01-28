from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

LeakageRisk = Literal["LOW", "MED", "HIGH"]


class LeakageFinding(TypedDict, total=False):
    """
    A per-variable leakage assessment.
    """
    z: str

    # Risk from deterministic heuristics
    risk_static: LeakageRisk

    # Risk suggested by the LLM
    risk_llm: LeakageRisk

    # Final risk used by the pipeline (conservative merge)
    risk_final: LeakageRisk

    # Human-readable explanation
    reason: str

    # Concrete remediation guidance
    action: str

    # Optional debugging / evidence for audits
    evidence: Dict[str, Any]


class LeakageReportState(TypedDict, total=False):
    """
    Workflow artifact produced by the leakage scan node.
    """
    z_candidates: List[str]
    findings: List[LeakageFinding]
    danger_list: List[str]  # typically all z where risk_final == HIGH

    n_low: int
    n_med: int
    n_high: int

    model_name: str
    created_at: str  # ISO-8601 UTC
    notes: str

    # Keep raw text for debugging / reproducibility (optional)
    raw_llm_output: Optional[str]
