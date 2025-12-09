# from __future__ import annotations

# from typing import Any, Dict, Protocol, Callable, Optional
# from uuid import UUID, uuid4
# import os

# import pandas as pd  # local CSV validation

# from langchain_core.messages import AIMessage

# from workflows.state.conversation_state import ConversationState


# JSONDict = Dict[str, Any]



# def make_load_and_validate_dataset_node(
#     user_repo: UserDatasetRepo,
# ) -> Callable[[CausalState], CausalState]:
#     """
#     Factory that wires the UserDatasetRepo dependency into the node.

#     Usage when building the graph:
#         from workflows.nodes.load_and_validate_dataset import (
#             make_load_and_validate_dataset_node,
#         )

#         load_and_validate_dataset = make_load_and_validate_dataset_node(user_repo)
#         graph.add_node("load_and_validate_dataset", load_and_validate_dataset)
#     """

#     def load_and_validate_dataset(state: CausalState) -> CausalState:
#         conversation_id = state.get("conversation_id")

#         # ------------------------------------------------------------------
#         # 1. Decide which dataset reference to use
#         # ------------------------------------------------------------------
#         explicit_path: Optional[str] = state.get("dataset_path")  # new in this turn
#         current_dataset_id: Optional[UUID] = state.get("dataset_id")

#         prev_dataset_id: Optional[UUID] = None
#         prev_dataset_path: Optional[str] = None

#         if conversation_id is not None:
#             prev_dataset_id, prev_dataset_path = user_repo.get_last_dataset(
#                 conversation_id=conversation_id
#             )

#         dataset_path: Optional[str] = explicit_path or prev_dataset_path
#         dataset_id: Optional[UUID] = current_dataset_id or prev_dataset_id

#         messages: list[AIMessage] = []

#         # No dataset anywhere → ask user to provide one
#         if not dataset_path:
#             text = (
#                 "I don’t see any dataset associated with this conversation yet.\n\n"
#                 "Please either:\n"
#                 "  • upload a CSV file, or\n"
#                 "  • provide a local/remote path to a CSV dataset I can load.\n"
#             )
#             messages.append(AIMessage(content=text))
#             return {
#                 "messages": messages,
#                 "last_error": {
#                     "type": "NO_DATASET",
#                     "detail": "No dataset_path provided and no previous dataset stored.",
#                 },
#                 "load_error": "NO_DATASET",
#             }

#         # ------------------------------------------------------------------
#         # 2. Validate basic file properties
#         # ------------------------------------------------------------------
#         if not dataset_path.lower().endswith(".csv"):
#             text = (
#                 f"The file `{dataset_path}` does not look like a CSV "
#                 "(expected a `.csv` extension).\n"
#                 "Please provide a CSV file."
#             )
#             messages.append(AIMessage(content=text))
#             return {
#                 "messages": messages,
#                 "last_error": {
#                     "type": "INVALID_FORMAT",
#                     "detail": f"Path {dataset_path!r} is not a .csv file.",
#                 },
#                 "load_error": "INVALID_FORMAT",
#             }

#         if not os.path.exists(dataset_path):
#             text = (
#                 f"I tried to load `{dataset_path}`, but the file does not exist.\n"
#                 "Please check the path or upload the file again."
#             )
#             messages.append(AIMessage(content=text))
#             return {
#                 "messages": messages,
#                 "last_error": {
#                     "type": "FILE_NOT_FOUND",
#                     "detail": f"File {dataset_path!r} not found on disk.",
#                 },
#                 "load_error": "FILE_NOT_FOUND",
#             }

#         # ------------------------------------------------------------------
#         # 3. Try to read CSV and run simple validation
#         # ------------------------------------------------------------------
#         try:
#             df = pd.read_csv(dataset_path)
#         except Exception as e:  # pragma: no cover  (depends on pandas internals)
#             text = (
#                 f"I tried to read `{dataset_path}` as a CSV, but got an error:\n\n"
#                 f"  {e}\n\n"
#                 "Please make sure the file is a valid CSV (comma or semicolon separated)."
#             )
#             messages.append(AIMessage(content=text))
#             return {
#                 "messages": messages,
#                 "last_error": {
#                     "type": "CSV_READ_ERROR",
#                     "detail": str(e),
#                 },
#                 "load_error": "CSV_READ_ERROR",
#             }

#         problems: list[str] = []
#         n_rows, n_cols = df.shape

#         if n_rows == 0:
#             problems.append("The dataset has 0 rows.")
#         if n_cols < 2:
#             problems.append("The dataset has fewer than 2 columns.")

#         if problems:
#             text = (
#                 "I was able to open your CSV, but there are validation issues:\n\n"
#                 + "\n".join(f"- {p}" for p in problems)
#                 + "\n\nPlease fix these issues or provide a different dataset."
#             )
#             messages.append(AIMessage(content=text))
#             return {
#                 "messages": messages,
#                 "last_error": {
#                     "type": "VALIDATION_ERROR",
#                     "detail": problems,
#                 },
#                 "load_error": "VALIDATION_ERROR",
#             }

#         # ------------------------------------------------------------------
#         # 4. Build schema + summary
#         # ------------------------------------------------------------------
#         raw_schema: JSONDict = {
#             "columns": [
#                 {"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()
#             ]
#         }
#         dataset_summary: JSONDict = {
#             "n_rows": int(n_rows),
#             "n_cols": int(n_cols),
#         }

#         # Assign dataset_id if new
#         if dataset_id is None:
#             dataset_id = uuid4()

#         # Persist for this conversation so next turn can reuse automatically
#         if conversation_id is not None:
#             user_repo.set_last_dataset(
#                 conversation_id=conversation_id,
#                 dataset_id=dataset_id,
#                 dataset_path=dataset_path,
#             )

#         # Friendly confirmation message
#         messages.append(
#             AIMessage(
#                 content=(
#                     "I’ve loaded your dataset successfully.\n\n"
#                     f"- Path: `{dataset_path}`\n"
#                     f"- Rows: {n_rows}\n"
#                     f"- Columns: {n_cols}\n\n"
#                     "Next I’ll infer the causal design (treatment, outcome, roles)."
#                 )
#             )
#         )

#         # Node output: only fields that changed / were computed
#         return {
#             "messages": messages,
#             "dataset_id": dataset_id,
#             "dataset_path": dataset_path,
#             "raw_schema": raw_schema,
#             "dataset_summary": dataset_summary,
#             "load_error": None,
#             "last_error": None,
#         }

#     return load_and_validate_dataset
