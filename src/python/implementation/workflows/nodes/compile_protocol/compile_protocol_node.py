from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, cast
from uuid import UUID

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol import compile_protocol_prompt
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_deps import CompileProtocolDeps
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    ProtocolSpec,
    validate_protocol_payload_structured,
)
from python.implementation.workflows.utils.utils import json_sanitize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileProtocolNode(Node):
    NAME: ClassVar[str] = CompileProtocolState.NAME
    llm: LLMService
    model_name: str
    max_attempts: int = 2

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
        mesages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        deps = CompileProtocolDeps.from_loaded(previous_state_dependencies)
        ds_summary = _extract_dataset_summary(deps)
        assert ds_summary is not None, "Dataset summary is required for protocol compilation."
        
    
        protocol_text = _extract_protocol_text(deps)
        assert protocol_text is not None, "Protocol discussion text is required for protocol compilation."
        
        dataset_cols = _extract_columns(ds_summary)

        last_raw: str = ""
        last_errors: List[str] = []
        last_issues: List[Dict[str, Any]] = []

        for attempt in range(1, max(1, self.max_attempts) + 1):
            prompt = _build_prompt(
                attempt=attempt,
                protocol_text=protocol_text,
                dataset_summary=ds_summary,
                previous_json=last_raw,
                validation_errors=last_errors,
                router_message=router_message,
            )

            try:
                raw = _llm_json_only(
                    llm=self.llm,
                    model_name=self.model_name,
                    prompt=prompt,
                    history=mesages_history,
                )
                last_raw = raw

                obj = _parse_json_object(raw)

                # 1) Schema validation (Pydantic)
                model_dict, issues = validate_protocol_payload_structured(obj)
                if model_dict is None:
                    last_issues = issues
                    last_errors = [_issue_to_str(x) for x in issues]
                    continue

                # 2) Semantic validation (dataset columns)
                sem_issues = _semantic_validate_against_dataset_columns(model_dict, dataset_cols)
                if sem_issues:
                    last_issues = sem_issues
                    last_errors = [_issue_to_str(x) for x in sem_issues]
                    continue

                protocol = cast(ProtocolSpec, model_dict)
                return CompileProtocolState(
                    protocol=protocol,
                    compile_error=None,
                    compile_issues=None,
                    user_message="Protocol compiled successfully.",
                )

            except Exception as e:
                msg = f"Attempt {attempt} failed: {e!r}"
                log.exception(msg)
                last_errors = [msg]
                last_issues = [{"path": "", "message": msg, "type": "exception", "input": None}]
                continue

        err_text = _format_errors(last_errors, last_raw)
        return CompileProtocolState(
            protocol=None,
            compile_error=err_text,
            compile_issues=last_issues or None,
            user_message="Failed to compile a valid protocol. There are errors in the specs. Lets discuss the specs again",
        )


# =============================================================================
# deps + extraction
# =============================================================================

def _extract_dataset_summary(deps: CompileProtocolDeps) -> Optional[Mapping[str, Any]]:
    """
    Be forgiving: LoadDatasetState may store the summary under different field names.
    """
    ld = deps.load_dataset

    # common candidates
    for attr in ("dataset_summary", "summary", "dataset"):
        v = getattr(ld, attr, None)
        if isinstance(v, Mapping):
            # if it's a "dataset" wrapper that contains "summary"
            inner = v.get("summary") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if isinstance(inner, Mapping):
                return cast(Mapping[str, Any], inner)
            return cast(Mapping[str, Any], v)

    # fallback: inspect JSON
    if hasattr(ld, "to_json_dict"):
        jd = cast(Any, ld).to_json_dict()
        if isinstance(jd, dict):
            ds = jd.get("dataset")
            if isinstance(ds, dict) and isinstance(ds.get("summary"), dict):
                return cast(Mapping[str, Any], ds["summary"])
            summ = jd.get("summary")
            if isinstance(summ, dict):
                return cast(Mapping[str, Any], summ)

    return None


def _extract_protocol_text(deps: CompileProtocolDeps) -> Optional[str]:
    pd = deps.protocol_discussion
    for attr in ("discussion", "protocol_text", "summary", "text"):
        v = getattr(pd, attr, None)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
    return None


def _extract_columns(ds_summary: Mapping[str, Any]) -> Optional[List[str]]:
    """
    Supports both:
      - {"column_names":[...]}
      - {"columns":[{"name":"..."}]}
      - {"profiles":[{"name":"..."}]}  (your newer profiler format)
    """
    col_names = ds_summary.get("column_names")
    if isinstance(col_names, list) and all(isinstance(x, str) for x in col_names):
        return list(col_names)

    cols = ds_summary.get("columns")
    if isinstance(cols, list):
        out: List[str] = []
        for c in cols:
            if isinstance(c, dict):
                name = c.get("name")
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
        return out or None

    profs = ds_summary.get("profiles")
    if isinstance(profs, list):
        out2: List[str] = []
        for p in profs:
            if isinstance(p, dict):
                name = p.get("name")
                if isinstance(name, str) and name.strip():
                    out2.append(name.strip())
        return out2 or None

    return None


# =============================================================================
# prompt / LLM / parse
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
        .replace("{{PREVIOUS_JSON}}", previous_json)
        .replace(
            "{{VALIDATION_ERRORS}}",
            json.dumps(validation_errors or ["Unknown compiler error"], ensure_ascii=False),
        )
        + appendix
    )


def _llm_json_only(
    *,
    llm: LLMService,
    model_name: str,
    prompt: str,
    history: Optional[Sequence[ChatMessage]],
) -> str:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    resp = llm.generate(
        config=cfg,
        system_prompt="Return JSON only. No extra text.",
        user_prompt=prompt,
        history=history,
    )
    return str(cast(Any, resp).content or "")


def _parse_json_object(raw: str) -> Dict[str, Any]:
    txt = (raw or "").strip()

    # tolerate leading/trailing junk; extract the outermost JSON object
    if not txt.startswith("{"):
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            txt = txt[i : j + 1]

    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object.")
    return cast(Dict[str, Any], obj)


# =============================================================================
# semantic validation (keep minimal; Pydantic already covers schema)
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

    # exclusions[].column
    excls = protocol.get("exclusions")
    if isinstance(excls, list):
        for i, ex in enumerate(excls):
            if isinstance(ex, dict):
                c = ex.get("column")
                if isinstance(c, str) and c not in cols:
                    bad(f"exclusions.{i}.column", c)

    # treatment_spec.column (always present)
    ts = protocol.get("treatment_spec")
    if isinstance(ts, dict):
        c = ts.get("column")
        if isinstance(c, str) and c not in cols:
            bad("treatment_spec.column", c)

    # outcome_spec: duration uses duration_column/event_column; others use column
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
            c = ys.get("column")
            if isinstance(c, str) and c not in cols:
                bad("outcome_spec.column", c)

    # covariates/effect_modifiers
    for key in ("covariates", "effect_modifiers"):
        v = protocol.get(key)
        if isinstance(v, list):
            for j, item in enumerate(v):
                if isinstance(item, str) and item not in cols:
                    bad(f"{key}.{j}", item)

    return issues


# =============================================================================
# errors formatting
# =============================================================================

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
