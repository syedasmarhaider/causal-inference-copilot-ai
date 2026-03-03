
from __future__ import annotations

from pydantic.dataclasses import dataclass

from python.domain.repo.data_repo import ImageMime


@dataclass(frozen=True)
class GraphImage:
    key: str
    title: str
    mime: ImageMime
    content: bytes