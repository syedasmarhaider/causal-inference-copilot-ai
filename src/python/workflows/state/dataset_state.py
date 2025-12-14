from __future__ import annotations

from typing import  Optional, TypedDict
from uuid import UUID

from python.workflows.nodes.load_dataset import JSONDict

class DatasetState(TypedDict, total=False):
    path: Optional[str]                    # path to CSV
    id: Optional[UUID]                     # later when you add a repo
    raw_schema: Optional[JSONDict]         # {"columns": [{"name":..., "dtype":...}, ...]}
    summary: Optional[JSONDict]            # {"n_rows": ..., "n_cols": ...}
    load_error: Optional[str]              # "NO_DATASET_PATH", "CSV_READ_ERROR", ...

