import re

from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

from openhands.sdk.conversation.visualizer.base import (
    ConversationVisualizerBase,
)
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    ConversationStateUpdateEvent,
    MessageEvent,
    ObservationEvent,
    PauseEvent,
    SystemPromptEvent,
    UserRejectObservation,
)
from openhands.sdk.event.base import Event
from openhands.sdk.event.condenser import Condensation, CondensationRequest


# These are external inputs
_OBSERVATION_COLOR = "yellow"
_MESSAGE_USER_COLOR = "gold3"
_PAUSE_COLOR = "bright_yellow"
# These are internal system stuff
_SYSTEM_COLOR = "magenta"
_THOUGHT_COLOR = "bright_black"
_ERROR_COLOR = "red"
# These are agent actions
_ACTION_COLOR = "blue"
_MESSAGE_ASSISTANT_COLOR = _ACTION_COLOR

DEFAULT_HIGHLIGHT_REGEX = {
    r"^Reasoning:": f"bold {_THOUGHT_COLOR}",
    r"^Thought:": f"bold {_THOUGHT_COLOR}",
    r"^Action:": f"bold {_ACTION_COLOR}",
    r"^Arguments:": f"bold {_ACTION_COLOR}",
    r"^Tool:": f"bold {_OBSERVATION_COLOR}",
    r"^Result:": f"bold {_OBSERVATION_COLOR}",
    r"^Rejection Reason:": f"bold {_ERROR_COLOR}",
    # Markdown-style
    r"\*\*(.*?)\*\*": "bold",
    r"\*(.*?)\*": "italic",
}


def indent_content(content: Text, spaces: int = 4) -> Text:
    """Indent content for visual hierarchy while preserving all formatting."""
    prefix = " " * spaces
    lines = content.split("\n")

    indented = Text()
    for i, line in enumerate(lines):
        if i > 0:
            indented.append("\n")
        indented.append(prefix)
        indented.append(line)

    return indented


def section_header(title: str, color: str) -> Rule:
    """Create a semantic divider with title."""
    return Rule(
        f"[{color} bold]{title}[/{color} bold]",
        style=color,
        characters="─",
        align="left",
    )


def build_event_block(
    content: Text,
    title: str,
    title_color: str,
    subtitle: str | None = None,
    indent: bool = False,
) -> Group:
    """Build a complete event block with header, content, and optional subtitle."""
    parts = []

    # Header with rule
    parts.append(section_header(title, title_color))
    parts.append(Text())  # Blank line after header

    # Content (optionally indented)
    if indent:
        parts.append(indent_content(content))
    else:
        parts.append(content)

    # Subtitle (metrics) if provided
    if subtitle:
        parts.append(Text())  # Blank line before subtitle
        subtitle_text = Text.from_markup(subtitle)
        subtitle_text.stylize("dim")
        parts.append(subtitle_text)

    parts.append(Text())  # Blank line after block

    return Group(*parts)


class DefaultConversationVisualizer(ConversationVisualizerBase):
    """Handles visualization of conversation events with Rich formatting.

    Provides dual-mode output:
    - Verbose: Full detail with rich formatting, indentation, and metrics
    - Concise: Minimal 1-2 line summaries for quick scanning
    """

    _console: Console
    _skip_user_messages: bool
    _highlight_patterns: dict[str, str]
    _mode: str
    _show_metrics_in_concise: bool

    def __init__(
        self,
        mode: str = "verbose",
        highlight_regex: dict[str, str] | None = DEFAULT_HIGHLIGHT_REGEX,
        skip_user_messages: bool = False,
        show_metrics_in_concise: bool = False,
    ):
        """Initialize the visualizer.

        Args:
            mode: Visualization mode - "verbose" or "concise"
            highlight_regex: Dictionary mapping regex patterns to Rich color styles
                           for highlighting keywords in the visualizer.
                           Only used in verbose mode.
                           For example: {"Reasoning:": "bold blue",
                           "Thought:": "bold green"}
            skip_user_messages: If True, skip displaying user messages. Useful for
                                scenarios where user input is not relevant to show.
            show_metrics_in_concise: If True, show token metrics even in concise mode
        """
        super().__init__()
        self._console = Console()
        self._skip_user_messages = skip_user_messages
        self._highlight_patterns = highlight_regex or {}
        self._mode = mode.lower()
        self._show_metrics_in_concise = show_metrics_in_concise

        if self._mode not in ("verbose", "concise"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'verbose' or 'concise'")

    def on_event(self, event: Event) -> None:
        """Main event handler that displays events with Rich formatting."""
        if self._mode == "verbose":
            output = self._create_verbose_panel(event)
            if output:
                self._console.print(output)
        else:  # concise mode
            output = self._create_concise_line(event)
            if output:
                self._console.print(output)

    def _apply_highlighting(self, text: Text) -> Text:
        """Apply regex-based highlighting to text content.

        Args:
            text: The Rich Text object to highlight

        Returns:
            A new Text object with highlighting applied
        """
        if not self._highlight_patterns:
            return text

        # Create a copy to avoid modifying the original
        highlighted = text.copy()

        # Apply each pattern using Rich's built-in highlight_regex method
        for pattern, style in self._highlight_patterns.items():
            pattern_compiled = re.compile(pattern, re.MULTILINE)
            highlighted.highlight_regex(pattern_compiled, style)

        return highlighted

    def _create_verbose_panel(self, event: Event) -> Group | None:
        """Create a verbose Rich panel for the event with full detail."""
        # Use the event's visualize property for content
        content = event.visualize

        if not content.plain.strip():
            return None

        # Apply highlighting if configured
        if self._highlight_patterns:
            content = self._apply_highlighting(content)

        # Determine panel styling based on event type
        if isinstance(event, SystemPromptEvent):
            return build_event_block(
                content=content,
                title="System Prompt",
                title_color=_SYSTEM_COLOR,
            )
        elif isinstance(event, ActionEvent):
            # Check if action is None (non-executable)
            if event.action is None:
                title = "Agent Action (Not Executed)"
            else:
                title = "Agent Action"
            return build_event_block(
                content=content,
                title=title,
                title_color=_ACTION_COLOR,
                subtitle=self._format_metrics_subtitle(),
            )
        elif isinstance(event, ObservationEvent):
            return build_event_block(
                content=content,
                title="Observation",
                title_color=_OBSERVATION_COLOR,
            )
        elif isinstance(event, UserRejectObservation):
            return build_event_block(
                content=content,
                title="User Rejected Action",
                title_color=_ERROR_COLOR,
            )
        elif isinstance(event, MessageEvent):
            if (
                self._skip_user_messages
                and event.llm_message
                and event.llm_message.role == "user"
            ):
                return
            assert event.llm_message is not None
            # Role-based styling
            role_colors = {
                "user": _MESSAGE_USER_COLOR,
                "assistant": _MESSAGE_ASSISTANT_COLOR,
            }
            role_color = role_colors.get(event.llm_message.role, "white")

            # Simple titles for base visualizer
            if event.llm_message.role == "user":
                title = "Message from User"
            else:
                title = "Message from Agent"

            return build_event_block(
                content=content,
                title=title,
                title_color=role_color,
                subtitle=self._format_metrics_subtitle(),
            )
        elif isinstance(event, AgentErrorEvent):
            return build_event_block(
                content=content,
                title="Agent Error",
                title_color=_ERROR_COLOR,
                subtitle=self._format_metrics_subtitle(),
            )
        elif isinstance(event, PauseEvent):
            return build_event_block(
                content=content,
                title="User Paused",
                title_color=_PAUSE_COLOR,
            )
        elif isinstance(event, Condensation):
            return build_event_block(
                content=content,
                title="Condensation",
                title_color="white",
                subtitle=self._format_metrics_subtitle(),
            )
        elif isinstance(event, CondensationRequest):
            return build_event_block(
                content=content,
                title="Condensation Request",
                title_color=_SYSTEM_COLOR,
            )
        elif isinstance(event, ConversationStateUpdateEvent):
            # Skip visualizing conversation state updates - these are internal events
            return None
        else:
            # Fallback panel for unknown event types
            title = f"UNKNOWN Event: {event.__class__.__name__}"
            subtitle = f"({event.source})"
            return build_event_block(
                content=content,
                title=title,
                title_color=_ERROR_COLOR,
                subtitle=subtitle,
            )

    def _create_concise_line(self, event: Event) -> Text | None:
        """Create a concise one-line summary of the event."""
        line = Text()

        if isinstance(event, SystemPromptEvent):
            line.append("System", style=f"bold {_SYSTEM_COLOR}")
            return line

        elif isinstance(event, ActionEvent):
            # Format: "Run: <tool_name> <brief_args>" or "Thinking" for reasoning
            content = event.visualize.plain

            # Check if this is a thinking/reasoning action
            if "Reasoning:" in content or "Thought:" in content:
                line.append("Thinking", style=f"{_THOUGHT_COLOR}")
                return line

            # Extract action name if available
            if event.action:
                action_name = event.action.__class__.__name__
                line.append("Run: ", style=f"bold {_ACTION_COLOR}")
                line.append(action_name, style=_ACTION_COLOR)

                # Try to extract key arguments for context
                args_preview = self._extract_action_preview(content)
                if args_preview:
                    line.append(f" {args_preview}")
            else:
                line.append("Action (Not Executed)", style=f"dim {_ACTION_COLOR}")

            return line

        elif isinstance(event, ObservationEvent):
            # Format: "Read: <tool> returned X lines" or "Read: <tool> <result_summary>"
            content = event.visualize.plain
            line.append("Read: ", style=f"bold {_OBSERVATION_COLOR}")

            # Extract tool name
            tool_match = re.search(r"Tool:\s*(\w+)", content)
            if tool_match:
                tool_name = tool_match.group(1)
                line.append(tool_name, style=_OBSERVATION_COLOR)

                # Count lines in result
                result_lines = content.count("\n")
                if result_lines > 5:
                    line.append(f" returned {result_lines} lines")
                else:
                    # Show brief result summary
                    result_preview = self._extract_result_preview(content)
                    if result_preview:
                        line.append(f" → {result_preview}")
            else:
                line.append("observation", style=_OBSERVATION_COLOR)

            return line

        elif isinstance(event, UserRejectObservation):
            line.append("Rejected: ", style=f"bold {_ERROR_COLOR}")
            # Try to extract rejection reason
            content = event.visualize.plain
            reason_match = re.search(r"Rejection Reason:\s*(.+?)(?:\n|$)", content)
            if reason_match:
                reason = reason_match.group(1).strip()[:60]
                line.append(reason, style=_ERROR_COLOR)
            else:
                line.append("User rejected action", style=_ERROR_COLOR)
            return line

        elif isinstance(event, MessageEvent):
            if (
                self._skip_user_messages
                and event.llm_message
                and event.llm_message.role == "user"
            ):
                return None

            assert event.llm_message is not None
            role = event.llm_message.role

            if role == "user":
                line.append("User: ", style=f"bold {_MESSAGE_USER_COLOR}")
                # Show first line or truncated content
                content = event.visualize.plain.strip()
                preview = content.split("\n")[0][:80]
                if len(content) > 80:
                    preview += "..."
                line.append(f'"{preview}"', style=_MESSAGE_USER_COLOR)
            else:
                line.append("Assistant: ", style=f"bold {_MESSAGE_ASSISTANT_COLOR}")
                content = event.visualize.plain.strip()

                # Count tokens or show character count
                preview = content.split("\n")[0][:80]
                if len(content) > 80:
                    line.append(f"response ({len(content)} chars)")
                else:
                    line.append(f'"{preview}"')

            return line

        elif isinstance(event, AgentErrorEvent):
            line.append("Error: ", style=f"bold {_ERROR_COLOR}")
            content = event.visualize.plain.strip()
            error_preview = content.split("\n")[0][:80]
            line.append(error_preview, style=_ERROR_COLOR)
            return line

        elif isinstance(event, PauseEvent):
            line.append("Paused", style=f"bold {_PAUSE_COLOR}")
            return line

        elif isinstance(event, Condensation):
            line.append("Condensed", style="dim")
            # Optionally show how many events were condensed
            content = event.visualize.plain
            line.append(f" ({len(content)} chars)")
            return line

        elif isinstance(event, CondensationRequest):
            line.append("Condensation Request", style=f"dim {_SYSTEM_COLOR}")
            return line

        elif isinstance(event, ConversationStateUpdateEvent):
            # Skip internal events in concise mode
            return None

        else:
            # Unknown event type
            line.append(
                f"Unknown: {event.__class__.__name__}", style=f"dim {_ERROR_COLOR}"
            )
            return line

    def _extract_action_preview(self, content: str) -> str:
        """Extract a brief preview of action arguments for concise mode."""
        # Try to extract key information like file paths, commands, etc.
        lines = content.split("\n")

        # Look for common patterns
        for line in lines:
            if "path:" in line.lower():
                match = re.search(r"path:\s*(.+?)(?:\n|$)", line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            if "command:" in line.lower():
                match = re.search(r"command:\s*(\w+)", line, re.IGNORECASE)
                if match:
                    cmd = match.group(1).strip()
                    return f"({cmd})"

        return ""

    def _extract_result_preview(self, content: str) -> str:
        """Extract a brief preview of observation result for concise mode."""
        # Find the Result: section and grab first meaningful line
        result_match = re.search(r"Result:\s*(.+?)(?:\n\n|\Z)", content, re.DOTALL)
        if result_match:
            result_text = result_match.group(1).strip()
            # Get first non-empty line
            for line in result_text.split("\n"):
                line = line.strip()
                if line and not line.startswith("Here's"):
                    return line[:60] + ("..." if len(line) > 60 else "")
        return ""

    def _format_metrics_subtitle(self) -> str | None:
        """Format LLM metrics as a visually appealing subtitle string with icons,
        colors, and k/m abbreviations using conversation stats."""
        stats = self.conversation_stats
        if not stats:
            return None

        combined_metrics = stats.get_combined_metrics()
        if not combined_metrics or not combined_metrics.accumulated_token_usage:
            return None

        usage = combined_metrics.accumulated_token_usage
        cost = combined_metrics.accumulated_cost or 0.0

        # helper: 1234 -> "1.2K", 1200000 -> "1.2M"
        def abbr(n: int | float) -> str:
            n = int(n or 0)
            if n >= 1_000_000_000:
                val, suffix = n / 1_000_000_000, "B"
            elif n >= 1_000_000:
                val, suffix = n / 1_000_000, "M"
            elif n >= 1_000:
                val, suffix = n / 1_000, "K"
            else:
                return str(n)
            return f"{val:.2f}".rstrip("0").rstrip(".") + suffix

        input_tokens = abbr(usage.prompt_tokens or 0)
        output_tokens = abbr(usage.completion_tokens or 0)

        # Cache hit rate (prompt + cache)
        prompt = usage.prompt_tokens or 0
        cache_read = usage.cache_read_tokens or 0
        cache_rate = f"{(cache_read / prompt * 100):.2f}%" if prompt > 0 else "N/A"
        reasoning_tokens = usage.reasoning_tokens or 0

        # Cost
        cost_str = f"{cost:.4f}" if cost > 0 else "0.00"

        # Build with fixed color scheme
        parts: list[str] = []
        parts.append(f"[cyan]↑ input {input_tokens}[/cyan]")
        parts.append(f"[magenta]cache hit {cache_rate}[/magenta]")
        if reasoning_tokens > 0:
            parts.append(f"[yellow] reasoning {abbr(reasoning_tokens)}[/yellow]")
        parts.append(f"[blue]↓ output {output_tokens}[/blue]")
        parts.append(f"[green]$ {cost_str}[/green]")

        return "Tokens: " + " • ".join(parts)
