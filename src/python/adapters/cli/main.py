# src/python/adapters/cli/main.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, cast
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, HumanMessage

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.implementation.repo.file_data_repo import FileDataRepo
from python.implementation.service.gemini_llm_service import GeminiLLMService
from python.workflows.graph.graph import build_copilot_app
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState


# -----------------------------
# State storage (swap later)
# -----------------------------

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


# -----------------------------
# Bootstrap state
# -----------------------------

DEFAULT_DATASET_PATH = Path("./data/486f4975-6cd9-4261-a122-e6b0fc46462d/data.csv").resolve()

def new_state(conversation_id: UUID) -> ConversationState:
    control: ControlState = {
        "conversation_id": conversation_id,
        "status": "PENDING",
        "stage": "LOAD_DATASET",
        "need": "NONE",
        "last_error": None,
        "node_message": f"Bootstrapped dataset path: {DEFAULT_DATASET_PATH}",
        # optional field in your TypedDict; safe to omit or include:
        "pending_stage": None,
    }
    return {
        "control": control,
        "dataset": {"path": str(DEFAULT_DATASET_PATH)},
        "metadata": {},
        "messages": [],
    }


# -----------------------------
# Helpers
# -----------------------------

def _ai_texts_since(messages: list[BaseMessage], start_idx: int) -> list[str]:
    out: list[str] = []
    for m in messages[start_idx:]:
        if getattr(m, "type", None) == "ai":
            txt = str(getattr(m, "content", "")).strip()
            if txt:
                out.append(txt)
    return out


@dataclass
class TurnResult:
    outputs: list[str]
    stage: str
    status: str
    need: str


@dataclass
class CopilotEngine:
    """
    Transport-agnostic runner.
    CLI calls it; later REST handler can call the same methods.
    """
    app: Any
    store: StateStore
    max_safety_invokes: int = 3  # with END-on-PRESENT graph, 1 is enough; this is guardrail only.

    def _invoke_once(self, conversation_id: UUID) -> TurnResult:
        state = self.store.load(conversation_id)

        before_msgs = cast(list[BaseMessage], state.get("messages", []))
        before_len = len(before_msgs)

        state2: ConversationState = self.app.invoke(state)
        self.store.save(conversation_id, state2)

        after_msgs = cast(list[BaseMessage], state2.get("messages", []))
        outs = _ai_texts_since(after_msgs, before_len)

        c = cast(ControlState, state2["control"])
        return TurnResult(
            outputs=outs,
            stage=str(c.get("stage")),
            status=str(c.get("status")),
            need=str(c.get("need") or "NONE"),
        )

    def kick(self, conversation_id: UUID) -> TurnResult:
        # No user input; just run until boundary (should stop at PRESENT or NEEDS_INPUT)
        last: TurnResult | None = None
        for _ in range(self.max_safety_invokes):
            last = self._invoke_once(conversation_id)
            # boundary reached if we produced output or we need input
            if last.outputs or last.need == "NEEDS_INPUT":
                return last
        return last or TurnResult(outputs=["(no output)"], stage="?", status="?", need="?")

    def run_turn(self, conversation_id: UUID, user_text: str) -> TurnResult:
        state = self.store.load(conversation_id)

        # append human message
        msgs = [*cast(list[BaseMessage], state.get("messages", [])), HumanMessage(content=user_text)]
        state2: ConversationState = {**state, "messages": msgs}
        self.store.save(conversation_id, state2)

        # run until boundary
        last: TurnResult | None = None
        for _ in range(self.max_safety_invokes):
            last = self._invoke_once(conversation_id)
            if last.outputs or last.need == "NEEDS_INPUT":
                return last
        return last or TurnResult(outputs=["(no output)"], stage="?", status="?", need="?")


# -----------------------------
# Wiring
# -----------------------------

def _wire_data_repo() -> DataRepo:
    return FileDataRepo(root_dir=Path(".local_data_repo"))

def _wire_llm() -> LLMService:
    return GeminiLLMService()


def run_console(*, data_repo: DataRepo, llm: LLMService) -> None:
    app = build_copilot_app(data_repo=data_repo, llm=llm)
    store = InMemoryStateStore()

    conversation_id = uuid4()
    store.save(conversation_id, new_state(conversation_id))
    engine = CopilotEngine(app=app, store=store)

    print("Causal Copilot (console). Type /help. Type /exit to quit.\n")

    res = engine.kick(conversation_id)
    for msg in res.outputs:
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
            s = store.load(conversation_id)
            c = cast(ControlState, s["control"])
            print(f"(stage={c['stage']}, status={c['status']}, need={c['need']})\n")
            continue
        if not user_text:
            continue

        res = engine.run_turn(conversation_id, user_text)
        print("\n\n".join(res.outputs) if res.outputs else "(no output)")
        print()


if __name__ == "__main__":
    run_console(data_repo=_wire_data_repo(), llm=_wire_llm())
