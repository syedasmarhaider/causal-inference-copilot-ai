from __future__ import annotations

from typing import Callable, TypedDict, List
from langchain_core.messages import BaseMessage

from python.workflows.state.control_state import ControlState
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState

class ConversationState(TypedDict):
    control: ControlState
    dataset: DatasetState
    metadata: MetadataState
    messages: List[BaseMessage]

CallableNodeFunc = Callable[[ConversationState], ConversationState]