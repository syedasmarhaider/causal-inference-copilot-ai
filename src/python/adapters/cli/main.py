from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from python.domain.models.workflow_response import WorkflowResponse
from python.domain.repo.conversation_repo import ConversationRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.domain.service.mcp_client import McpClient

from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState

from python.implementation.repo.file_data_repo import FileDataRepo
from python.implementation.service.gemini_llm_service import GeminiLLMService
from python.implementation.service.http_mcp_client import HttpMcpClient

from python.workflows.graph.simple_flow_entry import SimpleWorkflow, new_conversation_state
from python.workflows.graph.simple_flow_router import build_simple_copilot_app

log = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(
    "./data/486f4975-6cd9-4261-a122-e6b0fc46462d/data.csv"
).resolve()


# =============================================================================
# In-memory repo for CLI
# =============================================================================
class InMemoryConversationRepo(ConversationRepo):
    def __init__(self) -> None:
        self._store: dict[tuple[UUID, UUID], ConversationState] = {}

    def load(self, *, user_id: UUID, conversation_id: UUID) -> ConversationState | None:
        return self._store.get((user_id, conversation_id))

    def save(self, *, user_id: UUID, conversation_id: UUID, state: ConversationState) -> None:
        self._store[(user_id, conversation_id)] = state


# =============================================================================
# Drain adapter (keeps calling invoke(None) until stop)
# =============================================================================
@dataclass(frozen=True)
class DrainResult:
    outputs: list[str]
    needs_input: bool
    stage: str
    status: str
    post_action: str


class ConsoleAdapter:
    def __init__(self, *, workflow: SimpleWorkflow, repo: ConversationRepo) -> None:
        self._workflow = workflow
        self._repo = repo

    def snapshot(self, *, user_id: UUID, conversation_id: UUID) -> tuple[str, str, str]:
        s = self._repo.load(user_id=user_id, conversation_id=conversation_id)
        if s is None:
            return ("?", "?", "?")
        c: ControlState = s["control"]
        return (
            str(c.get("stage", "?")),
            str(c.get("status", "?")),
            str(c.get("post_action", "NONE")),
        )

    def drain(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_text: str | None,
        max_drains: int = 16,
    ) -> DrainResult:
        outputs: list[str] = []
        last: WorkflowResponse | None = None

        for i in range(max_drains):
            before = self.snapshot(user_id=user_id, conversation_id=conversation_id)

            txt = user_text if i == 0 else None
            last = self._workflow.invoke(
                user_id=user_id,
                conversation_id=conversation_id,
                user_text=txt,
            )

            t = (last.text or "").strip()
            if t:
                outputs.append(t)

            if last.needs_input:
                break

            after = self.snapshot(user_id=user_id, conversation_id=conversation_id)

            # Stop draining if nothing new and no control-plane progress.
            if not t and after == before:
                break

        stg, st, pa = self.snapshot(user_id=user_id, conversation_id=conversation_id)
        return DrainResult(
            outputs=outputs,
            needs_input=bool(last.needs_input) if last else False,
            stage=stg,
            status=st,
            post_action=pa,
        )


# =============================================================================
# Dependency wiring
# =============================================================================
def _wire_data_repo() -> DataRepo:
    return FileDataRepo(root_dir=Path(".local_data_repo"))


def _wire_llm() -> LLMService:
    return GeminiLLMService()


def _wire_mcp() -> McpClient:
    return HttpMcpClient(endpoint="http://127.0.0.1:8765/mcp")


# =============================================================================
# Main console loop
# =============================================================================
def run_console(*, data_repo: DataRepo, llm: LLMService, mcp_client: McpClient) -> None:
    logging.basicConfig(level=logging.INFO)

    repo = InMemoryConversationRepo()

    workflow = build_simple_copilot_app(
        repo=repo,
        data_repo=data_repo,
        llm=llm,
        mcp_client=mcp_client,
    )

    adapter = ConsoleAdapter(workflow=workflow, repo=repo)

    user_id = uuid4()
    conversation_id = uuid4()

    # Optional seed: pre-set dataset path so GET_FILE can skip asking
    seed = new_conversation_state(conversation_id)
    seed["dataset"]["path"] = str(DEFAULT_DATASET_PATH)
    repo.save(user_id=user_id, conversation_id=conversation_id, state=seed)

    print("Causal Copilot (console). Type /help. Type /exit to quit.\n")

    # Kick once (no user text) to let workflow emit first prompt if any
    kick = adapter.drain(user_id=user_id, conversation_id=conversation_id, user_text=None)
    for msg in kick.outputs:
        print(msg)
        print()

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if user_text == "/exit":
            print("bye")
            return
        if user_text == "/help":
            print("Commands: /help, /state, /exit\n")
            continue
        if user_text == "/state":
            stg, st, pa = adapter.snapshot(user_id=user_id, conversation_id=conversation_id)
            print(f"(stage={stg}, status={st}, post_action={pa})\n")
            continue
        if not user_text:
            continue

        turn = adapter.drain(user_id=user_id, conversation_id=conversation_id, user_text=user_text)
        print("\n\n".join(turn.outputs) if turn.outputs else "(no output)")
        print()


if __name__ == "__main__":
    run_console(data_repo=_wire_data_repo(), llm=_wire_llm(), mcp_client=_wire_mcp())
