from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from domain.models.unconfounded_causal_meta_data import UnconfoundedCausalMetaData
from domain.models.unconfounded_validation import ValidationReport


class UnconfoundedDataProcessingService(ABC):
    """
    Interface for the unconfounded causal pipeline.
    All methods are keyword-only to keep call sites explicit.
    """
 
    @abstractmethod
    def getMetaData(
        self,
        *,
        dataset_id: str | None = None,
    ) -> UnconfoundedCausalMetaData:
        """
        Build and return UnconfoundedCausalMetaData for the dataset.

        Parameters
        ----------
        dataset_id : str | None
            Logical dataset identifier (filename or ID). If None, use the service default.
        """

    @abstractmethod
    def validateData(
        self,
        *,
        meta: UnconfoundedCausalMetaData,
    ) -> ValidationReport:
        """
        Validate the current dataset against the provided metadata.

        Parameters
        ----------
        meta : UnconfoundedCausalMetaData
            Canonical metadata to validate against.

        Returns
        -------
        ValidationReport
            Structured issues and overall validity.
        """

    @abstractmethod
    def getData(
        self,
        *,
        dataset_id: str,
        as_stream: bool = False,
        chunk_size: int | None = None,
    ) -> Any:
        """
        Load the dataset by identifier.

        Parameters
        ----------
        dataset_id : str
            Either the original file name (e.g., 'cohort.csv') or a logical dataset id (e.g., 'ds_123').
        as_stream : bool
            If True, return an iterator/stream of chunks/records.
        chunk_size : int | None
            Optional row count per chunk when streaming.

        Returns
        -------
        Any
            - If as_stream == False: a DataFrame-like object (kept as Any to avoid a hard pandas dep).
            - If as_stream == True: an iterator over DataFrame-like chunks or row dicts.
        """
