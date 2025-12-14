from __future__ import annotations

import os
from uuid import uuid4

from langchain_core.messages import HumanMessage

from python.implementation.repo.file_data_repo import FileDataRepo
from python.domain.service.llm_service import LLMService

from python.workflows.graph.graph import build_copilot_app
from python.workflows.state.conversation_state import ConversationState


def _get_llm() -> LLMService:
    # You already planned get_llm_service() — use that here
    from python.implementation.service.gemini_llm_service import get_llm_service
    return get_llm_service()


def main() -> None:
    conversation_id = str(uuid4())

    data_repo = FileDataRepo(root_dir=os.getenv("COPILOT_DATA_DIR", ".copilot_data"))
    llm = _get_llm()

    app = build_copilot_app(data_repo=data_repo, llm=llm)

    # initial state
    state: ConversationState = {
        "control": {
            "conversation_id": conversation_id,
            "status": "OK",
            "stage": "LOAD_DATASET",
            "analysis_goal": "",
            "clarification_needed": True,
            "interrupt_type": None,
            "last_error": None,
            "node_message": "Provide a CSV path to begin.",
        },
        "dataset": {},
        "metadata": {},
        "messages": [],
    }

    print(f"[conversation_id={conversation_id}]")
    print("Type a CSV path first. Type 'exit' to quit.\n")

    while True:
        user_text = input("> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        # minimal input wiring:
        # - always add user message to history
        # - if we’re at LOAD_DATASET, treat input as dataset path
        update: ConversationState = {
            "messages": [HumanMessage(content=user_text)],
        }

        stage = state.get("control", {}).get("stage", "LOAD_DATASET")  # type: ignore[assignment]
        if stage == "LOAD_DATASET":
            update["dataset"] = {"path": user_text}

        # invoke one step (router -> stage -> present -> END)
        state = app.invoke({**state, **update})

        # print the *last* assistant message only
        msgs = state.get("messages", [])
        if msgs:
            print("\n" + str(msgs[-1].content) + "\n")


if __name__ == "__main__":
    main()
