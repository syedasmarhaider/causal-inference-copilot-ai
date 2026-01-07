from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage, HumanMessage

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.domain.service.mcp_client import McpClient
from python.implementation.repo.file_data_repo import FileDataRepo
from python.implementation.service.gemini_llm_service import GeminiLLMService
from python.implementation.service.http_mcp_client import HttpMcpClient

from python.workflows.graph.static_state_router import build_simple_copilot_app
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState


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


DEFAULT_DATASET_PATH = Path("./data/486f4975-6cd9-4261-a122-e6b0fc46462d/data.csv").resolve()


def new_state(conversation_id: UUID) -> ConversationState:
    control: ControlState = {
        "conversation_id": conversation_id,
        "stage": "GET_FILE",
        "status": "PENDING",
        "post_action": "NONE",
        "awaiting_user": False,  # ✅ IMPORTANT
        "post_failure_suggested_stage": None,
        "last_error": None,
        "node_message": "",
        "pending_stage": None,
    }

    return {
        "control": control,
        "dataset": {"path": str(DEFAULT_DATASET_PATH)},
        "metadata": {},
        "messages": [],
    }


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
    post_action: str


@dataclass
class CopilotEngine:
    app: Any
    store: StateStore
    max_safety_invokes: int = 4  # app.invoke already loops internally; keep small

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
            post_action=str(c.get("post_action") or "NONE"),
        )

    def kick(self, conversation_id: UUID) -> TurnResult:
        last: TurnResult | None = None
        for _ in range(self.max_safety_invokes):
            last = self._invoke_once(conversation_id)
            if last.outputs:
                return last
        return last or TurnResult(outputs=["(no output)"], stage="?", status="?", post_action="?")

    def run_turn(self, conversation_id: UUID, user_text: str) -> TurnResult:
        state = self.store.load(conversation_id)
        msgs = [*cast(list[BaseMessage], state.get("messages", [])), HumanMessage(content=user_text)]
        self.store.save(conversation_id, {**state, "messages": msgs})

        last: TurnResult | None = None
        for _ in range(self.max_safety_invokes):
            last = self._invoke_once(conversation_id)
            if last.outputs:
                return last
        return last or TurnResult(outputs=["(no output)"], stage="?", status="?", post_action="?")


def _wire_data_repo() -> DataRepo:
    return FileDataRepo(root_dir=Path(".local_data_repo"))


def _wire_llm() -> LLMService:
    return GeminiLLMService()


def _wire_mcp() -> McpClient:
    return HttpMcpClient(endpoint="http://127.0.0.1:8765/mcp")


def run_console(*, data_repo: DataRepo, llm: LLMService, mcp_client: McpClient) -> None:
    app = build_simple_copilot_app(data_repo=data_repo, llm=llm, mcp_client=mcp_client)

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
            print(f"(stage={c['stage']}, status={c['status']}, post_action={c.get('post_action','NONE')}, awaiting_user={c.get('awaiting_user', False)})\n")
            continue
        if not user_text:
            continue

        res = engine.run_turn(conversation_id, user_text)
        print("\n\n".join(res.outputs) if res.outputs else "(no output)")
        print()


if __name__ == "__main__":
    run_console(data_repo=_wire_data_repo(), llm=_wire_llm(), mcp_client=_wire_mcp())
