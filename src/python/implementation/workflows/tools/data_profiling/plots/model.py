from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

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
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None    