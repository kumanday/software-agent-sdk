"""Mixin for normalizing LLM responses with model-specific quirks.

This handles models that support native function calling but return malformed
tool call arguments (e.g., arrays/objects encoded as JSON strings).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol

from litellm.types.utils import Choices


if TYPE_CHECKING:
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.utils import ModelResponse


class _HostSupports(Protocol):
    """Protocol defining what the host class must provide."""

    model: str

    def _normalize_arguments_str(self, arguments_str: str) -> str:
        """Normalize tool call arguments."""
        ...


class ResponseNormalizerMixin:
    """Mixin for normalizing tool call arguments in LLM responses.

    Some LLMs return tool call arguments with nested data structures
    (arrays, objects) encoded as JSON strings instead of proper structures.
    For example: {"view_range": "[1, 100]"} instead of {"view_range": [1, 100]}

    This mixin provides methods to detect and fix these quirks based on model features.

    Host requirements:
    - self.model: str - The model identifier
    """

    def normalize_chat_completion_response(
        self: _HostSupports, response: ModelResponse
    ) -> ModelResponse:
        """Normalize tool call arguments in a Chat Completions API response.

        Modifies the response in-place if the model has known quirks.

        Args:
            response: The ModelResponse from litellm.completion()

        Returns:
            The same response object (modified in-place)
        """
        # Normalize tool call arguments in each choice's message
        for choice in response.choices:
            # Type guard: only handle non-streaming Choices
            if not isinstance(choice, Choices):
                continue
            if choice.message and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    if tool_call.function and tool_call.function.arguments:
                        tool_call.function.arguments = self._normalize_arguments_str(
                            tool_call.function.arguments
                        )

        return response

    def normalize_responses_api_response(
        self: _HostSupports, response: ResponsesAPIResponse
    ) -> ResponsesAPIResponse:
        """Normalize tool call arguments in a Responses API response.

        Modifies the response in-place if the model has known quirks.

        Args:
            response: The ResponsesAPIResponse from litellm.responses()

        Returns:
            The same response object (modified in-place)
        """
        # Normalize tool call arguments in output items
        if response.output:
            for item in response.output:
                # Check if it's a function_call item with arguments
                if (
                    hasattr(item, "type")
                    and getattr(item, "type", None) == "function_call"
                    and hasattr(item, "arguments")
                ):
                    args = getattr(item, "arguments", None)
                    if args:
                        setattr(item, "arguments", self._normalize_arguments_str(args))

        return response

    def _normalize_arguments_str(self: _HostSupports, arguments_str: str) -> str:
        """Recursively parse nested JSON strings in tool call arguments.

        This fixes arguments where nested structures are encoded as JSON strings.

        Args:
            arguments_str: The raw arguments JSON string from the LLM

        Returns:
            Normalized arguments JSON string with proper data structures
        """
        from openhands.sdk.llm.utils.model_features import get_features

        # Only normalize if model has this quirk
        if not get_features(self.model).args_as_json_strings:
            return arguments_str

        try:
            # Parse the JSON string
            args_dict = json.loads(arguments_str)

            # Recursively parse nested JSON strings
            def _recursively_parse(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: _recursively_parse(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_recursively_parse(item) for item in obj]
                elif isinstance(obj, str):
                    # Try to parse strings as JSON
                    try:
                        parsed = json.loads(obj)
                        return _recursively_parse(parsed)
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, return as-is
                        return obj
                else:
                    # Numbers, booleans, None - return as-is
                    return obj

            normalized_dict = _recursively_parse(args_dict)

            # Re-encode to JSON string
            return json.dumps(normalized_dict)
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, return original
            return arguments_str
