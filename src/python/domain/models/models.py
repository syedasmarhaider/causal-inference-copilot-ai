from dataclasses import dataclass
import json
from typing import Annotated, Any, Literal, Sequence
from uuid import UUID

from pydantic import StringConstraints
from typing_extensions import TypedDict

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

MessageRole = Literal["user", "assistant", "system"]

ArtifactKind = Literal["graph", "data"]
ArtifactFormat = Literal["json", "csv"]


class ArtifactRef(TypedDict, total=False):
    id: UUID
    kind: ArtifactKind
    format: ArtifactFormat


class ArtifactPayload(TypedDict, total=False):
    id: UUID
    content: Any
    kind: ArtifactKind
    format: ArtifactFormat


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    artifact_refs: Sequence[ArtifactRef] | None = None
    artifacts: Sequence[ArtifactPayload] | None = None
    id: str | None = None

    @property
    def message(self) -> str:
        return self.content


def get_chat_message_role_and_message_json(message: ChatMessage) -> str:
    payload = {
        "role": message.role,
        "message": message.content,
    }
    return json.dumps(payload, ensure_ascii=False)


def get_chat_messages_role_and_message_json(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(get_chat_message_role_and_message_json(msg) for msg in messages)


    
