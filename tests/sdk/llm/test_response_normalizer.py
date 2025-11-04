"""Tests for ResponseNormalizerMixin."""

import json
from unittest.mock import MagicMock

from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import (
    Choices,
    Function,
    Message as LiteLLMMessage,
    ModelResponse,
)

from openhands.sdk.llm.llm import LLM
from openhands.sdk.llm.mixins.response_normalizer import ResponseNormalizerMixin


class TestResponseNormalizerMixin:
    """Test ResponseNormalizerMixin functionality."""

    def test_llm_inherits_mixin(self):
        """Verify LLM inherits from ResponseNormalizerMixin."""
        assert issubclass(LLM, ResponseNormalizerMixin)

    def test_mixin_methods_available(self):
        """Verify mixin methods are available on LLM instances."""
        llm = LLM(model="litellm_proxy/openrouter/z-ai/glm-4.6")
        assert hasattr(llm, "normalize_chat_completion_response")
        assert hasattr(llm, "normalize_responses_api_response")
        assert hasattr(llm, "_normalize_arguments_str")

    def test_normalize_arguments_str_glm(self):
        """Test normalization for GLM models."""
        llm = LLM(model="litellm_proxy/openrouter/z-ai/glm-4.6")

        # Test array normalization
        args_str = '{"view_range": "[1, 100]"}'
        normalized = llm._normalize_arguments_str(args_str)
        parsed = json.loads(normalized)
        assert isinstance(parsed["view_range"], list)
        assert parsed["view_range"] == [1, 100]

        # Test nested object normalization
        args_str = '{"nested": "{\\"key\\": \\"value\\"}"}'
        normalized = llm._normalize_arguments_str(args_str)
        parsed = json.loads(normalized)
        assert isinstance(parsed["nested"], dict)
        assert parsed["nested"]["key"] == "value"

        # Test complex nested structure
        args_str = '{"data": "{\\"items\\": \\"[1, 2, 3]\\"}"}'
        normalized = llm._normalize_arguments_str(args_str)
        parsed = json.loads(normalized)
        assert isinstance(parsed["data"], dict)
        assert isinstance(parsed["data"]["items"], list)
        assert parsed["data"]["items"] == [1, 2, 3]

    def test_normalize_arguments_str_non_glm(self):
        """Test that non-GLM models pass through unchanged."""
        llm = LLM(model="gpt-4")

        args_str = '{"view_range": "[1, 100]"}'
        normalized = llm._normalize_arguments_str(args_str)
        # Should not change for non-GLM models
        assert normalized == args_str

    def test_normalize_arguments_str_invalid_json(self):
        """Test handling of invalid JSON."""
        llm = LLM(model="litellm_proxy/openrouter/z-ai/glm-4.6")

        # Invalid JSON should return original string
        args_str = "not json"
        normalized = llm._normalize_arguments_str(args_str)
        assert normalized == args_str

    def test_normalize_chat_completion_response_glm(self):
        """Test normalizing Chat Completions API response for GLM."""
        llm = LLM(model="litellm_proxy/openrouter/z-ai/glm-4.6")

        # Create a mock response with malformed tool call arguments
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="str_replace_editor",
                arguments='{"path": "/test.py", "view_range": "[1, 100]"}',
            ),
        )

        message = LiteLLMMessage(role="assistant", content=None, tool_calls=[tool_call])

        choice = Choices(finish_reason="tool_calls", index=0, message=message)

        response = ModelResponse(
            id="resp_123",
            model="glm-4",
            object="chat.completion",
            created=1234567890,
            choices=[choice],
        )

        # Normalize the response
        normalized = llm.normalize_chat_completion_response(response)

        # Check that tool call arguments are normalized
        first_choice = normalized.choices[0]
        assert isinstance(first_choice, Choices)
        assert first_choice.message.tool_calls is not None
        normalized_args = json.loads(
            first_choice.message.tool_calls[0].function.arguments
        )
        assert isinstance(normalized_args["view_range"], list)
        assert normalized_args["view_range"] == [1, 100]

    def test_normalize_chat_completion_response_non_glm(self):
        """Test that non-GLM models are not normalized."""
        llm = LLM(model="gpt-4")

        # Create a mock response
        tool_call = ChatCompletionMessageToolCall(
            id="call_123",
            type="function",
            function=Function(
                name="test_function",
                arguments='{"view_range": "[1, 100]"}',
            ),
        )

        message = LiteLLMMessage(role="assistant", content=None, tool_calls=[tool_call])

        choice = Choices(finish_reason="tool_calls", index=0, message=message)

        response = ModelResponse(
            id="resp_123",
            model="gpt-4",
            object="chat.completion",
            created=1234567890,
            choices=[choice],
        )

        # Normalize the response
        normalized = llm.normalize_chat_completion_response(response)

        # Arguments should remain unchanged for non-GLM models
        first_choice = normalized.choices[0]
        assert isinstance(first_choice, Choices)
        assert first_choice.message.tool_calls is not None
        assert (
            first_choice.message.tool_calls[0].function.arguments
            == '{"view_range": "[1, 100]"}'
        )

    def test_normalize_responses_api_response_glm(self):
        """Test normalizing Responses API response for GLM."""
        llm = LLM(model="litellm_proxy/openrouter/z-ai/glm-4.6")

        # Create a mock Responses API response
        function_call_item = MagicMock()
        function_call_item.type = "function_call"
        function_call_item.arguments = '{"path": "/test.py", "view_range": "[1, 100]"}'

        response = MagicMock()
        response.output = [function_call_item]

        # Normalize the response
        normalized = llm.normalize_responses_api_response(response)

        # Check that arguments are normalized
        assert normalized.output is not None
        assert len(normalized.output) > 0
        first_item = normalized.output[0]
        assert hasattr(first_item, "arguments")
        arguments_str = getattr(first_item, "arguments")
        assert arguments_str is not None
        normalized_args = json.loads(arguments_str)
        assert isinstance(normalized_args["view_range"], list)
        assert normalized_args["view_range"] == [1, 100]

    def test_mixin_is_reusable(self):
        """Test that the mixin can be used by other classes."""

        class CustomClass(ResponseNormalizerMixin):
            def __init__(self, model: str):
                self.model = model

        # Create instance with GLM model
        obj = CustomClass(model="litellm_proxy/openrouter/z-ai/glm-4.6")
        args_str = '{"arr": "[1, 2, 3]"}'
        normalized = obj._normalize_arguments_str(args_str)
        parsed = json.loads(normalized)
        assert isinstance(parsed["arr"], list)
        assert parsed["arr"] == [1, 2, 3]

        # Create instance with non-GLM model
        obj_gpt = CustomClass(model="gpt-4")
        normalized_gpt = obj_gpt._normalize_arguments_str(args_str)
        assert normalized_gpt == args_str

    def test_normalize_preserves_non_json_strings(self):
        """Test that regular strings that aren't JSON are preserved."""
        llm = LLM(model="litellm_proxy/openrouter/z-ai/glm-4.6")

        args_str = '{"message": "Hello, world!", "count": 42}'
        normalized = llm._normalize_arguments_str(args_str)
        parsed = json.loads(normalized)
        assert parsed["message"] == "Hello, world!"
        assert parsed["count"] == 42

    def test_normalize_empty_tool_calls(self):
        """Test normalizing response with no tool calls."""
        llm = LLM(model="litellm_proxy/openrouter/z-ai/glm-4.6")

        message = LiteLLMMessage(role="assistant", content="Hello", tool_calls=None)

        choice = Choices(finish_reason="stop", index=0, message=message)

        response = ModelResponse(
            id="resp_123",
            model="glm-4",
            object="chat.completion",
            created=1234567890,
            choices=[choice],
        )

        # Should not raise any errors
        normalized = llm.normalize_chat_completion_response(response)
        first_choice = normalized.choices[0]
        assert isinstance(first_choice, Choices)
        assert first_choice.message.content == "Hello"
