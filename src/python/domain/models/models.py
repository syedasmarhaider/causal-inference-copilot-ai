from typing import Annotated, Literal, TypedDict
from uuid import UUID

from pydantic import StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

ArtifactType = Literal["csv", "json"]


class Artifact_Id(TypedDict, total=False):
    id: UUID
    type: ArtifactType
