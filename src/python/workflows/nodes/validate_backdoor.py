# src/python/workflows/nodes/validate_backdoor.py
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast
from uuid import UUID

from langchain_core.messages import BaseMessage

from python.domain.service.mcp_client import McpClient, McpError
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ACTION, NEED_STAGE, ControlState, Stage, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState
from python.workflows.utils.types import JSONDict

log = logging.getLogger(__name__)


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


def _as_metadata(state: ConversationState) -> MetadataState:
    return cast(MetadataState, state.get("metadata", {}))  # type: ignore


def _uniq_preserve(xs: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for x in xs:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _extract_issues(report: Any) -> List[Dict[str, Any]]:
    """
    Accepts common shapes:
      - report is dict with key "issues"
      - report nested under {"content": {...}} or {"data": {...}}
    """
    if isinstance(report, dict):
        if isinstance(report.get("issues"), list):
            return cast(List[Dict[str, Any]], report["issues"])
        for k in ("data", "content", "result"):
            v = report.get(k)
            if isinstance(v, dict) and isinstance(v.get("issues"), list):
                return cast(List[Dict[str, Any]], v["issues"])
    return []


def _has_hard_failure(issues: Sequence[Dict[str, Any]]) -> bool:
    for it in issues:
        sev = str(it.get("severity", "")).upper()
        if sev == "HARD":
            return True
    return False


def _summarize_report(report: Any) -> str:
    issues = _extract_issues(report)
    n = len(issues)
    hard = sum(1 for it in issues if str(it.get("severity", "")).upper() == "HARD")
    soft = sum(1 for it in issues if str(it.get("severity", "")).upper() == "SOFT")
    warn = sum(1 for it in issues if str(it.get("severity", "")).upper() in ("WARN", "WARNING"))

    lines: List[str] = []
    lines.append("🧪 Backdoor validation (MCP)")
    lines.append(f"- issues: total={n}, hard={hard}, soft={soft}, warn={warn}")

    # show up to 6 most important issues (hard first)
    def score(it: Dict[str, Any]) -> Tuple[int, int]:
        sev = str(it.get("severity", "")).upper()
        sev_rank = {"HARD": 0, "SOFT": 1, "WARN": 2, "WARNING": 2}.get(sev, 9)
        return (sev_rank, 0)

    top = sorted(list(issues), key=score)[:6]
    if top:
        lines.append("\nTop issues:")
        for it in top:
            sev = str(it.get("severity", "")).upper() or "?"
            code = str(it.get("code", "")).strip() or "UNKNOWN"
            msg = str(it.get("message", "")).strip()
            msg = msg if msg else str(it.get("detail", "")).strip()
            if len(msg) > 180:
                msg = msg[:180] + "…"
            lines.append(f"- [{sev}] {code}: {msg}")

    return "\n".join(lines)


def make_validate_backdoor_node(
    mcp: McpClient,
    *,
    tool_name: str = "backdoor_validate",
    spec_overrides: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    exclude_checks: Optional[List[str]] = None,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    hard_failure_rewind_stage: NEED_STAGE = "GET_FILE",
) -> Callable[[ConversationState], ConversationState]:
    """
    VALIDATE_BACKDOOR stage.

    - Requires: dataset.id (UUID) AND metadata.final_design accepted.
    - Calls MCP tool backdoor_validate(dataset_id, treat_col, outcome_col, feature_cols, ...)
    - HARD issues => ABORTED + suggest hard_failure_rewind_stage (default GET_FILE, as you requested)
    - Otherwise => DONE (and PRESENT summary)
    """

    def node(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)
        metadata_in = _as_metadata(state)

        conversation_id = control_in["conversation_id"]
        stage: Stage = control_in["stage"]

        def mk_control(
            *,
            status: Status,
            post_action: ACTION,
            post_failure_suggested_stage: NEED_STAGE | None,
            node_message: str,
            last_error: JSONDict | None,
            pending_stage: Stage | None = None,
        ) -> ControlState:
            out: ControlState = cast(
                ControlState,
                {
                    **control_in,
                    "conversation_id": conversation_id,
                    "stage": stage,
                    "status": status,
                    "post_action": post_action,
                    "post_failure_suggested_stage": post_failure_suggested_stage,
                    "last_error": last_error,
                    "node_message": node_message,
                    "pending_stage": pending_stage,
                },
            )
            return out

        # ---- guards: dataset id must exist
        ds_id = dataset_in.get("id")
        if not isinstance(ds_id, UUID):
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="LOAD_DATASET",
                    last_error={"code": "MISSING_DATASET_ID", "detail": "dataset.id missing; run LOAD_DATASET first."},
                    node_message="Fatal: dataset not loaded/registered. Suggested recovery: LOAD_DATASET.",
                ),
            }

        # ---- guards: must have accepted final metadata
        final_design = metadata_in.get("final_design")
        if not isinstance(final_design, dict) or final_design.get("accepted") is not True:
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="CONFIRM_METADATA",
                    last_error={"code": "MISSING_FINAL_METADATA", "detail": "metadata.final_design not accepted yet."},
                    node_message="Metadata not confirmed yet. Please finish CONFIRM_METADATA first.",
                ),
            }

        treat = final_design.get("treatment")
        outcome = final_design.get("outcome")
        covs = final_design.get("covariates") or []
        mods = final_design.get("effect_modifiers") or []

        if not isinstance(treat, str) or not isinstance(outcome, str):
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="CONFIRM_METADATA",
                    last_error={"code": "INVALID_METADATA", "detail": "treatment/outcome missing in final_design."},
                    node_message="Metadata is incomplete (missing treatment/outcome). Go back to CONFIRM_METADATA.",
                ),
            }

        feature_cols = _uniq_preserve([*(cast(List[str], covs)), *(cast(List[str], mods))])

        # ---- call MCP
        args: Dict[str, Any] = {
            "dataset_id": str(ds_id),
            "treat_col": treat,
            "outcome_col": outcome,
            "feature_cols": feature_cols,
            "exclude_checks": exclude_checks or [],
            "params": params or {},
            "spec_overrides": spec_overrides or None,
            "config_overrides": config_overrides or None,
        }

        try:
            res = mcp.call_tool(tool_name, args)
            report = res.data
        except McpError as e:
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="VALIDATE_BACKDOOR",
                    last_error={"code": "MCP_CALL_FAILED", "detail": str(e)},
                    node_message=(
                        "Validation failed because MCP call failed.\n"
                        "Make sure the MCP server is running and reachable, then try again."
                    ),
                ),
            }
        except Exception as e:
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="VALIDATE_BACKDOOR",
                    last_error={"code": "VALIDATE_EXCEPTION", "detail": str(e)},
                    node_message="Validation failed due to an unexpected error.",
                ),
            }

        # Persist raw report (best-effort; keeps typing flexible)
        dataset_out: DatasetState = cast(DatasetState, {**dataset_in, "validation_report": report})

        issues = _extract_issues(report)
        hard = _has_hard_failure(issues)

        summary = _summarize_report(report)

        if hard:
            # As requested: HARD => go back to GET_FILE
            return {
                **state,
                "dataset": dataset_out,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage=hard_failure_rewind_stage,
                    last_error={"code": "VALIDATION_HARD_FAIL", "detail": "One or more HARD issues in report."},
                    node_message=summary + "\n\nHARD failure => restarting at GET_FILE (validation is mandatory).",
                ),
            }

        return {
            **state,
            "dataset": dataset_out,
            "control": mk_control(
                status="DONE",
                post_action="PRESENT",
                post_failure_suggested_stage=None,
                last_error=None,
                node_message=summary,
            ),
        }

    return node
