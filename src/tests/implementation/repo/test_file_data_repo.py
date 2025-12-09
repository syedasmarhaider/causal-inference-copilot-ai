import json
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.implementation.repo.file_data_repo import FileDataRepo


# ---------- helpers ----------

def _read_json(path: Path) -> dict: # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    if not path.exists():
        return {} # pyright: ignore[reportUnknownVariableType]
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {} # pyright: ignore[reportUnknownVariableType]
    return json.loads(raw)


# ---------- basic initialization / indices ----------

def test_init_creates_index_files(tmp_path: Path) -> None:
    _ = FileDataRepo(tmp_path)  # noqa: F841

    datasets_index = tmp_path / "datasets_index.json"
    conversations_index = tmp_path / "conversation_index.json"

    assert datasets_index.exists()
    assert conversations_index.exists()

    assert _read_json(datasets_index) == {}
    assert _read_json(conversations_index) == {}


# ---------- register_csv_dataset behavior ----------

def test_register_creates_dataset_and_conversation_entry(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_csv(csv_path, index=False)

    conv_id = uuid4()
    dataset_id = repo.register_csv_dataset(
        conversation_id=conv_id,
        dataset_path=str(csv_path),
    )

    # dataset_id should be a UUID
    assert isinstance(dataset_id, UUID)

    datasets_index = _read_json(tmp_path / "datasets_index.json") # pyright: ignore[reportUnknownVariableType]
    conversations_index = _read_json(tmp_path / "conversation_index.json") # pyright: ignore[reportUnknownVariableType]

    ds_key = str(dataset_id)
    conv_key = str(conv_id)

    assert conv_key in conversations_index
    assert conversations_index[conv_key] == ds_key

    assert ds_key in datasets_index
    assert datasets_index[ds_key]["path"] == str(csv_path)


def test_register_same_conversation_reuses_dataset_id_and_updates_path(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    conv_id = uuid4()

    csv_path_1 = tmp_path / "data1.csv"
    csv_path_2 = tmp_path / "data2.csv"
    pd.DataFrame({"x": [1]}).to_csv(csv_path_1, index=False)
    pd.DataFrame({"x": [2]}).to_csv(csv_path_2, index=False)

    dataset_id_1 = repo.register_csv_dataset(
        conversation_id=conv_id,
        dataset_path=str(csv_path_1),
    )
    dataset_id_2 = repo.register_csv_dataset(
        conversation_id=conv_id,
        dataset_path=str(csv_path_2),
    )

    # same conversation -> same dataset_id
    assert dataset_id_1 == dataset_id_2

    datasets_index = _read_json(tmp_path / "datasets_index.json") # pyright: ignore[reportUnknownVariableType]
    conversations_index = _read_json(tmp_path / "conversation_index.json") # pyright: ignore[reportUnknownVariableType]

    ds_key = str(dataset_id_1)
    conv_key = str(conv_id)

    assert conversations_index[conv_key] == ds_key
    # last registration wins for the path
    assert datasets_index[ds_key]["path"] == str(csv_path_2)


def test_register_different_conversations_produce_different_dataset_ids(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

    conv_id_1 = uuid4()
    conv_id_2 = uuid4()

    ds1 = repo.register_csv_dataset(conversation_id=conv_id_1, dataset_path=str(csv_path))
    ds2 = repo.register_csv_dataset(conversation_id=conv_id_2, dataset_path=str(csv_path))

    assert ds1 != ds2

    datasets_index = _read_json(tmp_path / "datasets_index.json") # pyright: ignore[reportUnknownVariableType]
    conversations_index = _read_json(tmp_path / "conversation_index.json") # pyright: ignore[reportUnknownVariableType]

    assert conversations_index[str(conv_id_1)] == str(ds1)
    assert conversations_index[str(conv_id_2)] == str(ds2)
    assert str(ds1) in datasets_index
    assert str(ds2) in datasets_index


# ---------- get_last_dataset ----------

def test_get_last_dataset_existing(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

    conv_id = uuid4()
    registered_ds_id = repo.register_csv_dataset(
        conversation_id=conv_id,
        dataset_path=str(csv_path),
    )

    ds_id, path = repo.get_last_dataset(conversation_id=conv_id)
    assert ds_id == registered_ds_id
    assert path == str(csv_path)


def test_get_last_dataset_missing_conversation_returns_none(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    conv_id = uuid4()
    ds_id, path = repo.get_last_dataset(conversation_id=conv_id)

    assert ds_id is None
    assert path is None


def test_get_last_dataset_missing_dataset_entry_returns_none(tmp_path: Path) -> None:
    """
    conversation_index points to a dataset_id that does not exist
    in datasets_index.json -> should return (None, None)
    """
    repo = FileDataRepo(tmp_path)

    conv_id = uuid4()
    conv_key = str(conv_id)

    # manually corrupt indices
    (tmp_path / "datasets_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "conversation_index.json").write_text(
        json.dumps({conv_key: str(uuid4())}), encoding="utf-8"
    )

    ds_id, path = repo.get_last_dataset(conversation_id=conv_id)
    assert ds_id is None
    assert path is None


def test_get_last_dataset_with_non_dict_dataset_entry_returns_none(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    conv_id = uuid4()
    fake_ds_id = str(uuid4())

    (tmp_path / "datasets_index.json").write_text(
        json.dumps({fake_ds_id: "not-a-dict"}), encoding="utf-8"
    )
    (tmp_path / "conversation_index.json").write_text(
        json.dumps({str(conv_id): fake_ds_id}), encoding="utf-8"
    )

    ds_id, path = repo.get_last_dataset(conversation_id=conv_id)
    assert ds_id is None
    assert path is None


def test_get_last_dataset_with_invalid_uuid_returns_none(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    conv_id = uuid4()
    bad_ds_id = "not-a-uuid"

    (tmp_path / "datasets_index.json").write_text(
        json.dumps({bad_ds_id: {"path": "/tmp/something.csv"}}),
        encoding="utf-8",
    )
    (tmp_path / "conversation_index.json").write_text(
        json.dumps({str(conv_id): bad_ds_id}), encoding="utf-8"
    )

    ds_id, path = repo.get_last_dataset(conversation_id=conv_id)
    assert ds_id is None
    assert path is None


# ---------- _resolve_path (indirectly tests _load_datasets_index) ----------

def test_resolve_path_success(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

    conv_id = uuid4()
    ds_id = repo.register_csv_dataset(conversation_id=conv_id, dataset_path=str(csv_path))

    resolved = repo._resolve_path(ds_id)  # type: ignore[attr-defined]
    assert resolved == csv_path


def test_resolve_path_missing_dataset_raises(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    missing_ds_id = uuid4()
    with pytest.raises(KeyError):
        repo._resolve_path(missing_ds_id)  # type: ignore[attr-defined]


def test_resolve_path_invalid_path_type_raises(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    ds_id = uuid4()
    (tmp_path / "datasets_index.json").write_text(
        json.dumps({str(ds_id): {"path": 123}}), encoding="utf-8"
    )

    with pytest.raises(KeyError):
        repo._resolve_path(ds_id)  # type: ignore[attr-defined]


# ---------- get_csv_data ----------

def test_get_csv_data_reads_full_dataframe(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    csv_path = tmp_path / "data.csv"
    expected = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    expected.to_csv(csv_path, index=False)

    conv_id = uuid4()
    ds_id = repo.register_csv_dataset(conversation_id=conv_id, dataset_path=str(csv_path))

    df = repo.get_csv_data(ds_id)
    # reset index to ensure equality
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_get_csv_data_respects_limit(tmp_path: Path, limit: int) -> None:
    repo = FileDataRepo(tmp_path)

    csv_path = tmp_path / "data.csv"
    base_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    base_df.to_csv(csv_path, index=False)

    conv_id = uuid4()
    ds_id = repo.register_csv_dataset(conversation_id=conv_id, dataset_path=str(csv_path))

    df = repo.get_csv_data(ds_id, limit=limit)

    expected = base_df.head(limit) if limit >= 0 else base_df
    pd.testing.assert_frame_equal(df.reset_index(drop=True), expected.reset_index(drop=True))


# ---------- get_csv_data_iteratively ----------

def test_get_csv_data_iteratively_yields_chunks(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    n_rows = 10
    csv_path = tmp_path / "data.csv"
    base_df = pd.DataFrame({"a": list(range(n_rows))})
    base_df.to_csv(csv_path, index=False)

    conv_id = uuid4()
    ds_id = repo.register_csv_dataset(conversation_id=conv_id, dataset_path=str(csv_path))

    chunk_size = 4
    chunks = list(repo.get_csv_data_iteratively(ds_id, chunk_size=chunk_size))

    # Expect 3 chunks: 4 + 4 + 2
    lengths = [len(c) for c in chunks]
    assert lengths == [4, 4, 2]

    # Concatenate and compare to original
    concat_df = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(
        concat_df.reset_index(drop=True),
        base_df.reset_index(drop=True),
    )


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_get_csv_data_iteratively_invalid_chunk_size_raises(tmp_path: Path, chunk_size: int) -> None:
    repo = FileDataRepo(tmp_path)
    ds_id = uuid4()  # won't be used if validation happens first

    with pytest.raises(ValueError):
        _ = list(repo.get_csv_data_iteratively(ds_id, chunk_size=chunk_size))


# ---------- corrupted / empty index file handling (_load_json) ----------

def test_load_datasets_index_empty_file_is_treated_as_empty_dict(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    # Overwrite datasets_index.json with an empty file
    (tmp_path / "datasets_index.json").write_text("", encoding="utf-8")

    # Should not raise; should behave as empty dict internally
    conv_id = uuid4()
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

    ds_id = repo.register_csv_dataset(conversation_id=conv_id, dataset_path=str(csv_path))
    assert isinstance(ds_id, UUID)


def test_load_datasets_index_invalid_json_is_treated_as_empty(tmp_path: Path) -> None:
    repo = FileDataRepo(tmp_path)

    # Overwrite with invalid JSON
    (tmp_path / "datasets_index.json").write_text("not valid json", encoding="utf-8")

    conv_id = uuid4()
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)

    # Should still work and not crash
    ds_id = repo.register_csv_dataset(conversation_id=conv_id, dataset_path=str(csv_path))
    assert isinstance(ds_id, UUID)
