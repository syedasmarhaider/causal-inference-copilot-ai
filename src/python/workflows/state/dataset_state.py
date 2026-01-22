from __future__ import annotations

from typing import Any, List, Optional, TypedDict
from uuid import UUID

from python.workflows.utils.types import JSONDict


class DatasetState(TypedDict, total=False):
    """
    total=False because early stages only have `path`,
    later stages enrich with id/schema/summary.
    """
    id: Optional[UUID]
    raw_schema: Optional[JSONDict]
    summary: Optional[JSONDict]
    load_error: Optional[str]
    get_file_last_user_msg_idx: Optional[int]


class DatasetStateHelpers:
    @staticmethod
    def extract_columns_from_df(df: Any) -> List[str]:
        """
        Best-effort extraction of column names from a DataFrame-like object.
        Works for pandas DataFrame and similar objects that expose `.columns`.
        Returns [] if not available.
        """
        try:
            raw = getattr(df, "columns", None)
            if raw is None:
                return []
            cols = [str(c).strip() for c in list(raw)]
            return [c for c in cols if c]
        except Exception:
            return []
