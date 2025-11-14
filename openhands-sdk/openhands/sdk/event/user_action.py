from rich.text import Text

from openhands.sdk.event.base import Event
from openhands.sdk.event.types import SourceType


class PauseEvent(Event):
    """Event indicating that the agent execution was paused by user request."""

    source: SourceType = "user"

    def visualize(self, concise: bool = False) -> Text:
        """Return Rich Text representation of this pause event.

        Args:
            concise: If True, return a minimal 1-2 line summary.
                    If False (default), return detailed verbose representation.
        """
        content = Text()

        if concise:
            # Concise mode: one-line summary
            content.append("Paused", style="bold bright_yellow")
        else:
            # Verbose mode: full detail
            content.append("Conversation Paused", style="bold")

        return content

    def __str__(self) -> str:
        """Plain text string representation for PauseEvent."""
        return f"{self.__class__.__name__} ({self.source}): Agent execution paused"
