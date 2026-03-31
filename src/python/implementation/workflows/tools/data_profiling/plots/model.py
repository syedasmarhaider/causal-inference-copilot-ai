from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from python.domain.repo.data_repo import ImageMime


@dataclass(frozen=True)
class GraphImage:
    key: str
    title: str
    mime: ImageMime
    content: bytes


@dataclass (frozen=True)
class CohortCate:
    group_key: str
    cate: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None    