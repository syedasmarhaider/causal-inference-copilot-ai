from typing import  Annotated, List, TypedDict
from langgraph.graph.message import add_messages # pyright: ignore[reportMissingTypeStubs]
from langchain_core.messages import BaseMessage

from python.workflows.state.control_state import ControlState
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState

class ConversationState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    # Structured sub-states
    control: ControlState
    dataset: DatasetState
    metadata: MetadataState
    # estimation: EstimationState
    # effects: EffectsState