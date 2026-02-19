from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from pydantic import ValidationError

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol import compile_protocol_prompt
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_deps import CompileProtocolDeps
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.load_dataset.load_dataset_utils import DatasetSummary
from python.implementation.workflows.utils.utils import json_sanitize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileProtocolNode(Node):
    NAME: ClassVar[str] = CompileProtocolState.NAME

    llm: LLMService
    model_name: str

    # Compile-level attempts: attempt 1 (base prompt) -> attempt 2 (repair prompt)
    max_attempts: int = 2

    # LLM internal attempts per prompt (handled inside llm.generate_json)
    json_attempts: int = 2

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return compile_protocol_prompt.compile_protocol_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: Optional[ToolFactory],
        previous_state_dependencies: Mapping[str, State],
        user_message: Optional[str],
        router_message: Optional[str],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        deps = CompileProtocolDeps.from_loaded(previous_state_dependencies)
        ld = deps.load_dataset
        ds_summary: DatasetSummary | None = ld.summary
        assert ds_summary is not None  
        protocol_text = _extract_protocol_text(deps)
        if protocol_text is None:
            raise AssertionError("Protocol discussion text is required for protocol compilation.")

        dataset_cols = _extract_columns_from_profiles(ds_summary)

        last_json: str = ""
        last_errors: List[str] = []
        last_issues: List[Dict[str, Any]] = []

        for attempt in range(1, max(1, self.max_attempts) + 1):
            prompt = _build_prompt(
                attempt=attempt,
                protocol_text=protocol_text,
                dataset_summary=ds_summary,
                previous_json=last_json,
                validation_errors=last_errors,
                router_message=router_message,
            )
            
            attempt_history = messages_history if attempt == 1 else None

            try:
                protocol_model = _llm_protocol_model(
                    llm=self.llm,
                    model_name=self.model_name,
                    prompt=prompt,
                    history=attempt_history,
                    json_attempts=max(1, self.json_attempts),
                )

                protocol_dict = protocol_model.model_dump(mode="json")
                last_json = json.dumps(protocol_dict, ensure_ascii=False)

                sem_issues = _semantic_validate_against_dataset_columns(protocol_dict, dataset_cols)
                if sem_issues:
                    last_issues = sem_issues
                    last_errors = [_issue_to_str(x) for x in sem_issues]
                    continue

                # Store protocol as JSON dict for deterministic state serialization
                return CompileProtocolState(
                    protocol=protocol_model,
                    compile_error=None,
                    compile_issues=None,
                    user_message="Protocol compiled successfully.",
                )

            except ValidationError as ve:
                # Safety: if the parser returns something but Pydantic validation fails
                issues = _pydantic_error_to_issues(ve)
                last_issues = issues
                last_errors = [_issue_to_str(x) for x in issues]
                continue

            except Exception as e:
                msg = f"Attempt {attempt} failed: {e!r}"
                log.exception(msg)
                last_errors = [msg]
                last_issues = [{"path": "", "message": msg, "type": "exception", "input": None}]
                continue

        err_text = _format_errors(last_errors, last_json)
        return CompileProtocolState(
            protocol=None,
            compile_error=err_text,
            compile_issues=last_issues or None,
            user_message="Failed to compile a valid protocol. Lets discuss the specs again.",
        )


# =============================================================================
# LLM call
# =============================================================================

def _llm_protocol_model(
    *,
    llm: LLMService,
    model_name: str,
    prompt: str,
    history: Optional[Sequence[ChatMessage]],
    json_attempts: int,
) -> ProtocolSpec:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    return llm.generate_json(
        schema=ProtocolSpec,
        system_prompt="Return JSON only. No extra text.",
        user_prompt=prompt,
        config=cfg,
        history=history,
        max_attempts=max(1, json_attempts),
    )


# =============================================================================
# protocol text extraction
# =============================================================================

def _extract_protocol_text(deps: CompileProtocolDeps) -> Optional[str]:
    """
    This is the only part that depends on how your ProtocolDiscussionState is shaped.
    It tries common fields; if none exist/valid, it returns None (caller asserts).
    """
    pd = deps.protocol_discussion
    for attr in ("discussion", "protocol_text", "summary", "text"):
        v = getattr(pd, attr, None)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
    return None


# =============================================================================
# dataset columns extraction (STRICT to your DatasetSummary format)
# =============================================================================

def _extract_columns_from_profiles(ds_summary: Mapping[str, Any]) -> Optional[List[str]]:
    """
    STRICT: DatasetSummary has:
      - profiles: List[ColumnProfile]
      - each ColumnProfile has name: str
    """
    profs = ds_summary.get("profiles")
    if not isinstance(profs, list):
        return None

    out: List[str] = []
    for p in profs:
        if isinstance(p, dict):
            name = p.get("name")
            if isinstance(name, str):
                n = name.strip()
                if n:
                    out.append(n)

    return out or None


# =============================================================================
# prompt builder
# =============================================================================

def _build_prompt(
    *,
    attempt: int,
    protocol_text: str,
    dataset_summary: Mapping[str, Any],
    previous_json: str,
    validation_errors: List[str],
    router_message: Optional[str],
) -> str:
    ds_json = json.dumps(json_sanitize(dict(dataset_summary)), ensure_ascii=False)

    appendix = ""
    if router_message:
        appendix = "\n\nROUTER_MESSAGE:\n" + router_message.strip()

    if attempt == 1:
        return (
            compile_protocol_prompt.compile_protocol_prompt()
            .replace("{{PROTOCOL_TEXT}}", protocol_text)
            .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
            + appendix
        )

    return (
        compile_protocol_prompt.compile_protocol_repair_prompt()
        .replace("{{PROTOCOL_TEXT}}", protocol_text)
        .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
        .replace("{{PREVIOUS_JSON}}", previous_json or "{}")
        .replace(
            "{{VALIDATION_ERRORS}}",
            json.dumps(validation_errors or ["Unknown compiler error"], ensure_ascii=False),
        )
        + appendix
    )


# =============================================================================
# semantic validation (columns)
# =============================================================================

def _semantic_validate_against_dataset_columns(
    protocol: Mapping[str, Any],
    dataset_columns: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not dataset_columns:
        return []

    cols = set(dataset_columns)
    issues: List[Dict[str, Any]] = []

    def bad(path: str, val: Any) -> None:
        issues.append(
            {
                "path": path,
                "message": f"Column not found in dataset: {val!r}",
                "type": "column_not_in_dataset",
                "input": val,
            }
        )

    excls = protocol.get("exclusions")
    if isinstance(excls, list):
        for i, ex in enumerate(excls):
            if isinstance(ex, dict):
                c = ex.get("column")
                if isinstance(c, str) and c not in cols:
                    bad(f"exclusions.{i}.column", c)

    ts = protocol.get("treatment_spec")
    if isinstance(ts, dict):
        c = ts.get("column")
        if isinstance(c, str) and c not in cols:
            bad("treatment_spec.column", c)

    ys = protocol.get("outcome_spec")
    if isinstance(ys, dict):
        kind = ys.get("kind")
        if kind == "duration":
            dc = ys.get("duration_column")
            ec = ys.get("event_column")
            if isinstance(dc, str) and dc not in cols:
                bad("outcome_spec.duration_column", dc)
            if isinstance(ec, str) and ec not in cols:
                bad("outcome_spec.event_column", ec)
        else:
            c2 = ys.get("column")
            if isinstance(c2, str) and c2 not in cols:
                bad("outcome_spec.column", c2)

    for key in ("covariates", "effect_modifiers"):
        v = protocol.get(key)
        if isinstance(v, list):
            for j, item in enumerate(v):
                if isinstance(item, str) and item not in cols:
                    bad(f"{key}.{j}", item)

    return issues


# =============================================================================
# error formatting
# =============================================================================

def _pydantic_error_to_issues(err: ValidationError) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for e in err.errors():
        loc = e.get("loc", ())
        path = ".".join(str(x) for x in loc)
        issues.append(
            {
                "path": path,
                "message": str(e.get("msg", "Invalid value")),
                "type": str(e.get("type", "validation_error")),
                "input": e.get("input"),
            }
        )
    return issues


def _issue_to_str(issue: Mapping[str, Any]) -> str:
    path = str(issue.get("path", "")).strip()
    msg = str(issue.get("message", "Invalid value")).strip()
    return f"{path}: {msg}" if path else msg


def _format_errors(errors: List[str], raw_json: str) -> str:
    e = "\n".join([f"- {x}" for x in (errors or ["Unknown error"])])
    snippet = (raw_json or "").strip()
    if len(snippet) > 1000:
        snippet = snippet[:1000] + "…"
    return f"Validation/compile errors:\n{e}\n\nLast JSON snippet:\n{snippet}"
