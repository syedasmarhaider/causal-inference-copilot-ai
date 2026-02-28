import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from uuid import UUID


# -----------------------------------------------------------------------------
# Robust module importer (you might rename files/paths while refactoring)
# -----------------------------------------------------------------------------
CANDIDATE_LINEAR_DML_MODULES = [
    "python.implementation.workflows.tools.causal.econml.dml.linear_dml_causal_model",
    "python.implementation.workflows.tools.causal.econml.dml.linear_dml",
    "python.implementation.workflows.tools.causal.econml.linear_dml_causal_model",
    "python.implementation.workflows.tools.causal.econml.linear_dml",
]


def import_linear_dml_module():
    last_err: Optional[Exception] = None
    for mod_name in CANDIDATE_LINEAR_DML_MODULES:
        try:
            return importlib.import_module(mod_name)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import LinearDMLCausalModel module. Tried:\n"
        + "\n".join(f"  - {m}" for m in CANDIDATE_LINEAR_DML_MODULES)
        + f"\nLast error: {last_err!r}"
    )


def import_command_module():
    return importlib.import_module("python.implementation.workflows.tools.causal.causal_command")


# -----------------------------------------------------------------------------
# In-memory repos (duck-typed, no dependency on your domain layer)
# -----------------------------------------------------------------------------
@dataclass
class InMemoryModelRecord:
    model: Any
    metadata: Dict[str, Any]


class InMemoryDataRepo:
    def __init__(self):
        self._dfs: Dict[UUID, Any] = {}
        self.calls: list[tuple[UUID, UUID, UUID, Optional[int]]] = []

    def put(self, dataset_id: UUID, df: Any) -> None:
        self._dfs[dataset_id] = df

    def get_csv_data(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID, limit: Optional[int] = None):
        self.calls.append((user_id, conversation_id, dataset_id, limit))
        if dataset_id not in self._dfs:
            raise FileNotFoundError(f"dataset {dataset_id} not found")
        return self._dfs[dataset_id]


class InMemoryModelsRepo:
    def __init__(self):
        self._store: Dict[Tuple[UUID, UUID, UUID], InMemoryModelRecord] = {}
        self.save_calls: list[tuple[UUID, UUID, UUID]] = []
        self.load_calls: list[tuple[UUID, UUID, UUID]] = []

    def save_model(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID, model: Any, metadata: Dict[str, Any]):
        self.save_calls.append((user_id, conversation_id, model_id))
        self._store[(user_id, conversation_id, model_id)] = InMemoryModelRecord(model=model, metadata=metadata)

    def load_model(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID):
        self.load_calls.append((user_id, conversation_id, model_id))
        return self._store.get((user_id, conversation_id, model_id))