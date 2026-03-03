from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID

import pandas as pd

ImageMime = Literal["image/png", "image/jpeg", "image/webp"]

@dataclass(frozen=True)
class ArtifactRef:
    user_id: UUID
    conversation_id: UUID
    artifact_id: UUID
    mime: ImageMime
    path: Path
    size_bytes: int


class DataRepo(ABC):
    @abstractmethod
    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve data for a dataset_id as a pandas DataFrame.

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param dataset_id: Dataset UUID.
        :param limit: Optional row limit (head).
        """

    @abstractmethod
    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> Path:
        """
        Persist data for a dataset_id to durable storage.

        Expected storage layout (file-backed impl):
          ./data/<user_id>/<conversation_id>/<dataset_id>/data.csv

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param dataset_id: Dataset UUID.
        :param df: DataFrame to persist.
        :param overwrite: If False, raise if the target already exists.
        :param include_index: If True, write the DataFrame index into the CSV.
        :return: Path/URI to the persisted CSV (Path for file-backed repos).
        """
    @abstractmethod
    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: ImageMime,
        overwrite: bool = True,
    ) -> ArtifactRef:
        """
        Persist an image (bytes) to durable storage.

        Expected storage layout (file-backed impl):
          ./data/<user_id>/<conversation_id>/images/<artifact_id>.<ext>

        :param mime: MUST match content encoding (repo does not transcode).
        :param overwrite: If False, raise if target exists.
        :return: ArtifactRef with path + metadata
        """

    @abstractmethod
    def get_artifact_ref(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: Optional[ImageMime] = None,
    ) -> ArtifactRef:
        """
        Return ArtifactRef (path + mime). If expected_mime is provided and mismatched, raise.
        """    
