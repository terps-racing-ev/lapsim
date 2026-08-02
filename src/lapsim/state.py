"""Shared interfaces for components that own simulation state."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ResettableComponent(Protocol):
    """Component whose dynamic state can be restored before a simulation."""

    def reset_state(self) -> None:
        """Restore the component's configured initial state."""

