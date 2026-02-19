from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID, uuid4

from python.domain.repo.conversation_repo import ConversationRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import LLMService

from python.implementation.repo.inmemory_conversation_repo import InMemoryConversationRepo
from python.implementation.repo.file_data_repo import FileDataRepo
from python.implementation.repo.models_repo import FileSystemModelsRepo
from python.implementation.service.llms.llm_service_factory import (
    LLMServiceSettings,
    make_llm_service,
)

from python.workflows.graph.simple_flow_entry import SimpleWorkflow, WorkflowConfig


log = logging.getLogger(__name__)


# =============================================================================
# Adapter: calls workflow.invoke() repeatedly with user_text only on first call,
# until we need user input OR nothing changes.
# (invoke itself is still exactly 1 node execution per call)
# =============================================================================
@dataclass(frozen=True)
class DrainResult:
    outputs: list[str]
    needs_input: bool
    stage: str
    status: str


class ConsoleAdapter:
    def __init__(self, *, workflow: SimpleWorkflow, repo: ConversationRepo) -> None:
        self._workflow = workflow
        self._repo = repo

    def _snapshot(self, *, user_id: UUID, conversation_id: UUID) -> tuple[str, str]:
        s = self._repo.load(user_id=user_id, conversation_id=conversation_id)
        if not isinstance(s, dict):
            return ("?", "?")
        c = s.get("control")
        # new control keys
        return (str(c.get("current_stage", "?")), str(c.get("current_stage_status", "?")))

    def drain(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_text: str | None,
        max_steps: int = 16,
    ) -> DrainResult:
        outputs: list[str] = []

        needs_input = False
        last_stage = "?"
        last_status = "?"

        for i in range(max_steps):
            before_stage, before_status = self._snapshot(user_id=user_id, conversation_id=conversation_id)

            txt = user_text if i == 0 else None
            resp = self._workflow.invoke(
                user_id=user_id,
                conversation_id=conversation_id,
                user_text=txt,
            )

            msg = (resp.node_message or "").strip()
            if msg:
                outputs.append(msg)

            needs_input = bool(resp.needs_input)
            last_stage = str(resp.current_stage)
            last_status = str(resp.current_stage_status)

            if needs_input:
                break

            after_stage, after_status = self._snapshot(user_id=user_id, conversation_id=conversation_id)

            # Stop if no message AND no control-plane progress.
            if not msg and (after_stage, after_status) == (before_stage, before_status):
                break

        return DrainResult(
            outputs=outputs,
            needs_input=needs_input,
            stage=last_stage,
            status=last_status,
        )


# =============================================================================
# Wiring (no DB; in-memory conversation repo)
# =============================================================================
def wire_repo() -> ConversationRepo:
    return InMemoryConversationRepo()


def wire_data_repo() -> DataRepo:
    # file-based dataset access; not a DB
    return FileDataRepo()

def wire_models_repo() -> ModelsRepo:
    return FileSystemModelsRepo()

def wire_llm() -> LLMService:
    return make_llm_service(LLMServiceSettings(provider="openai"))



# =============================================================================
# Main console
# =============================================================================
def run_console(*, repo: ConversationRepo, data_repo: DataRepo, llm: LLMService) -> None:
    logging.basicConfig(level=logging.INFO)

    cfg = WorkflowConfig(data_repo=data_repo, llm=llm, models_repo=wire_models_repo())
    workflow = SimpleWorkflow(repo=repo, cfg=cfg)
    adapter = ConsoleAdapter(workflow=workflow, repo=repo)

    user_id = uuid4()
    conversation_id = uuid4()

    print("Causal Copilot (console). Commands: /help, /exit, /new\n")

    # Kick: let the workflow run until it needs user input (or stops progressing)
    kick = adapter.drain(user_id=user_id, conversation_id=conversation_id, user_text=None)
    for out in kick.outputs:
        print(out)
        print()

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not user_text:
            continue

        if user_text in {"/exit", "exit", "quit"}:
            print("bye")
            return

        if user_text == "/help":
            print(
                "Commands:\n"
                "  /help  show this help\n"
                "  /exit  quit\n"
                "  /new   start a new conversation\n"
            )
            continue

        if user_text == "/new":
            conversation_id = uuid4()
            print("(new conversation)\n")
            kick2 = adapter.drain(user_id=user_id, conversation_id=conversation_id, user_text=None)
            for out in kick2.outputs:
                print(out)
                print()
            continue

        turn = adapter.drain(user_id=user_id, conversation_id=conversation_id, user_text=user_text)
        print("\n\n".join(turn.outputs) if turn.outputs else "(no output)")
        print()


if __name__ == "__main__":
    run_console(
        repo=wire_repo(),
        data_repo=wire_data_repo(),
        llm=wire_llm(),
    )
