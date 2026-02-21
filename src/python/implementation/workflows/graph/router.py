from __future__ import annotations


from typing import Mapping, Optional, Protocol, Union

from python.domain.workflows.node import Node
from python.domain.workflows.state import State

class Router:
    def get_state(
        self,
        state_name: str,
        *,
        renew_on_already_executed: bool,
    ) -> State: ...

    # 2) Get node (you can pass state name or node name; your choice)
    def get_node(self, name: Union[str, State]) -> Node: ...

    # 3) Decide next (the router’s job)
    def decide_next(
        self,
        *,
        current_state_name: Optional[str] = None,
        user_message_present: bool,
    ) -> NextDecision: ...

    # 4) Optional but useful: introspection/debug
    def snapshot(self) -> Mapping[str, State]: ...