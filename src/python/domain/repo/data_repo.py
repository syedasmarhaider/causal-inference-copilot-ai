from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional
from uuid import UUID

import pandas as pd

ImageMime = Literal["image/png", "image/jpeg", "image/webp"]


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
    ) -> None:
        """
        Persist data for a dataset_id to durable storage.

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param dataset_id: Dataset UUID.
        :param df: DataFrame to persist.
        :param overwrite: If False, raise if the target already exists.
        :param include_index: If True, write the DataFrame index into the CSV.
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
    ) -> None:
        """
        Persist an image artifact to durable storage.

        :param mime: MUST match content encoding (repo does not transcode).
        :param overwrite: If False, raise if target exists.
        """

    @abstractmethod
    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: Optional[ImageMime] = None,
    ) -> bytes:
        """
        Return artifact bytes. If expected_mime is provided and mismatched, raise.
        """

    @abstractmethod
    def get_artifact_mime(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> ImageMime:
        """
        Return the artifact MIME type.
        """