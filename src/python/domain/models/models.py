from typing import Annotated, Literal
from uuid import UUID

from pydantic import StringConstraints
from typing_extensions import TypedDict

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

ArtifactType = Literal["csv", "json"]


class Artifact_Id(TypedDict, total=False):
    id: UUID
    type: ArtifactType
