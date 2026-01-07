from __future__ import annotations

from typing import Optional, TypedDict
from uuid import UUID

from python.workflows.utils.types import JSONDict


class DatasetState(TypedDict, total=False):
    """
    total=False because early stages only have `path`,
    later stages enrich with id/schema/summary.
    """
    path: Optional[str]
    id: Optional[UUID]
    raw_schema: Optional[JSONDict]
    summary: Optional[JSONDict]
    load_error: Optional[str]

    # If you use it in GET_FILE to prevent re-parsing old messages
    get_file_last_user_msg_idx: Optional[int]


def empty_dataset_state(path: Optional[str] = None) -> DatasetState:
    return {
        "path": path,
        "id": None,
        "raw_schema": None,
        "summary": None,
        "load_error": None,
        "get_file_last_user_msg_idx": -1,
    }
