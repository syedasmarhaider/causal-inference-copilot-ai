from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal
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
    def get_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
    ) -> str:
        """
        Retrieve data for a dataset_id as a pandas DataFrame.

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param dataset_id: Dataset UUID.
        :param limit: Optional row limit (head).
        """

    @abstractmethod
    def save_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        json_data: str,
        *,
        overwrite: bool = True,
    ) -> None:
        """
        Persist data for a dataset_id to durable storage.

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param dataset_id: Dataset UUID.
        :param json_data: JSON string to persist.
        :param overwrite: If False, raise if the target already exists.
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
        Persist a binary artifact for a conversation.
        """

    @abstractmethod
    def get_artifact_mime(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> ImageMime:
        """
        Resolve the stored mime type for an artifact.
        """

    @abstractmethod
    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: ImageMime | None = None,
    ) -> bytes:
        """
        Retrieve artifact content, optionally validating the expected mime type.
        """
