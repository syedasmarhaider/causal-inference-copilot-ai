from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DomainMessage:
    """
    What your CLI/UI consumes.

    - command drives UI behavior
      - "PRESENT": render message, then UI may call invoke() again with user_text=None
      - "PRESENT_AND_USER_INPUT": render message, then prompt user for input
      - "NEEDS_INPUT": prompt user (text may be empty or contain a hint)
      - "NONE": no-op (should rarely escape invoke(), but kept for completeness)
    """
    text: str
    wait_for_user: bool
    graph_instructions: Optional[dict] = None # pyright: ignore[reportMissingTypeArgument]