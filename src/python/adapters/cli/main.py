from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, BaseMessage

from python.implementation.repo.file_data_repo import FileDataRepo
from python.implementation.service.gemini_llm_service import GeminiLLMService
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState
from python.workflows.graph.graph import build_copilot_app

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService


class StateStore(Protocol):
    def load(self, conversation_id: UUID) -> ConversationState: ...
    def save(self, conversation_id: UUID, state: ConversationState) -> None: ...


class InMemoryStateStore:
    def __init__(self) -> None:
        self._by_id: dict[UUID, ConversationState] = {}

    def load(self, conversation_id: UUID) -> ConversationState:
        return self._by_id[conversation_id]

    def save(self, conversation_id: UUID, state: ConversationState) -> None:
        self._by_id[conversation_id] = state


def new_state(conversation_id: UUID) -> ConversationState:
    control: ControlState = {
        "conversation_id": conversation_id,
        "status": "PENDING",
        "stage": "LOAD_DATASET",
        "outcome": "NOT_RUN_YET",
        "need": "DATASET_PATH",
        "interrupt_type": None,
        "last_error": None,
        "node_message": "Paste a path to a .csv file to begin.",
    }
    return {
        "control": control,
        "dataset": {},
        "metadata": {},
        "messages": [],
    }


def _last_ai_text(messages: list[BaseMessage]) -> Optional[str]:
    for m in reversed(messages):
        if getattr(m, "type", None) == "ai":
            txt = str(getattr(m, "content", "")).strip()
            if txt:
                return txt
    return None


def _apply_console_input(state: ConversationState, user_text: str) -> ConversationState:
    # If we're waiting for dataset path, treat raw input as path unless it's a command.
    control = state["control"]
    if control["stage"] == "LOAD_DATASET" and control["need"] == "DATASET_PATH":
        if user_text.lower().startswith("/load "):
            path = user_text.split(" ", 1)[1].strip()
            return {**state, "dataset": {**state.get("dataset", {}), "path": path}}
        # heuristic: plain path
        if user_text.strip().lower().endswith(".csv"):
            return {**state, "dataset": {**state.get("dataset", {}), "path": user_text.strip()}}
    return state


@dataclass(frozen=True)
class ConsoleCopilot:
    app: Any
    store: StateStore

    def step(self, conversation_id: UUID, user_text: str) -> str:
        state = self.store.load(conversation_id)

        # commands
        if user_text.strip() == "/state":
            c = state["control"]
            return f"(stage={c['stage']}, status={c['status']}, outcome={c['outcome']}, need={c['need']})"
        if user_text.strip() == "/help":
            return "Commands: /help, /state, /exit, /load <path.csv>"

        # attach user message + update dataset.path if needed
        state = _apply_console_input(state, user_text)
        state = {**state, "messages": [*state.get("messages", []), HumanMessage(content=user_text)]}

        # run one graph turn (graph itself will auto-advance until it needs input)
        new_state: ConversationState = self.app.invoke(state)

        self.store.save(conversation_id, new_state)

        msg = _last_ai_text(list(new_state.get("messages", [])))
        return msg or str(new_state["control"].get("node_message") or "OK")


def run_console(*, data_repo: DataRepo, llm: LLMService) -> None:
    app = build_copilot_app(data_repo=data_repo, llm=llm)
    store = InMemoryStateStore()

    conversation_id = uuid4()
    store.save(conversation_id, new_state(conversation_id))

    copilot = ConsoleCopilot(app=app, store=store)

    print("Causal Copilot (console). Type /help. Type /exit to quit.\n")
    # kick: run once with empty input to get first assistant message if your graph/presenter does that
    # (optional) otherwise we just wait for user.

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if user_text == "/exit":
            print("bye")
            return

        if not user_text:
            continue

        out = copilot.step(conversation_id, user_text)
        print(out)
        print()


def _wire_data_repo() -> DataRepo:
     return FileDataRepo(root_dir="./data")


def _wire_llm() -> LLMService:

    return GeminiLLMService()


if __name__ == "__main__":
    data_repo = _wire_data_repo()
    llm = _wire_llm()
    run_console(data_repo=data_repo, llm=llm)
