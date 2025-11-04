#!/usr/bin/env python3
"""
Example demonstrating how to configure the LLM class for enterprise API gateways
that require custom headers and SSL certificate handling.

This example shows configuration patterns commonly used at large enterprises
with corporate proxies or API management systems.
"""

import os
import uuid
from datetime import datetime

from pydantic import SecretStr

from openhands.sdk.llm import LLM, Message, TextContent, content_to_str


def create_enterprise_llm():
    """
    Create an LLM instance configured for enterprise gateway access.

    This example shows how to:
    1. Add custom headers required by the gateway (auth tokens, correlation IDs)
    2. Set a custom base URL for the enterprise proxy
    3. Disable SSL verification when corporate proxies break cert chains
    4. Specify the underlying provider explicitly
    """

    # Generate dynamic headers that may be required by the gateway
    now = datetime.now()
    correlation_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    # Configure the LLM with enterprise gateway settings
    llm = LLM(
        model="gemini-2.5-flash",  # Model name as exposed by gateway
        api_key=SecretStr("placeholder"),  # Often required even if not used
        # Enterprise proxy endpoint
        base_url="https://your-corporate-proxy.company.com/api/llm",
        # Custom headers required by the gateway
        extra_headers={
            "Authorization": "Bearer YOUR_ENTERPRISE_TOKEN",
            "Content-Type": "application/json",
            "x-correlation-id": correlation_id,
            "x-request-id": request_id,
            "x-request-date": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "X-USECASE-ID": "YOUR_USECASE_ID",
            "x-client-id": "YOUR_CLIENT_ID",
            "x-api-key": "YOUR_API_KEY",
        },
        # Disable SSL verification if corporate proxy breaks certificate chain
        ssl_verify=False,  # Or provide path to certificate bundle: "/path/to/cert.pem"
        # Explicitly specify the provider for LiteLLM routing
        custom_llm_provider="openai",
        # Other configurations
        num_retries=1,
        timeout=30,
    )

    return llm


def create_llm_from_env():
    """
    Create an LLM instance using environment variables.

    Set these environment variables:
    - LLM_MODEL=gemini-2.5-flash
    - LLM_API_KEY=placeholder
    - LLM_BASE_URL=https://your-corporate-proxy.company.com/api/llm
    - LLM_SSL_VERIFY=false
    - LLM_CUSTOM_LLM_PROVIDER=openai
    - LLM_EXTRA_HEADERS='{"Authorization": "Bearer TOKEN", "x-correlation-id": "123"}'
    """

    # The load_from_env method automatically handles:
    # - Boolean parsing for ssl_verify (accepts: false, False, 0, no, off)
    # - JSON parsing for complex fields like extra_headers
    llm = LLM.load_from_env()

    return llm


def example_usage():
    """Demonstrate using the enterprise-configured LLM."""

    # Create the LLM instance
    llm = create_enterprise_llm()

    # Use the LLM for completion
    messages = [
        Message(
            role="system", content=[TextContent(text="You are a helpful assistant.")]
        ),
        Message(
            role="user", content=[TextContent(text="What is the capital of France?")]
        ),
    ]

    response = llm.completion(messages=messages)

    # Access the response content from LLMResponse
    print(f"Response: {content_to_str(response.message.content)}")

    # The extra_headers are automatically included in the request to the gateway
    # The ssl_verify setting is applied to the HTTPS connection
    # The custom_llm_provider ensures proper routing through LiteLLM


if __name__ == "__main__":
    # Example 1: Direct configuration
    print("Example 1: Direct configuration")
    llm = create_enterprise_llm()
    print(f"Created LLM with model: {llm.model}")
    print(f"Base URL: {llm.base_url}")
    print(f"SSL Verify: {llm.ssl_verify}")
    print(f"Extra headers configured: {bool(llm.extra_headers)}")

    # Example 2: Environment variable configuration
    print("\nExample 2: Environment variable configuration")
    # Set example environment variables (normally these would be set externally)
    os.environ["LLM_MODEL"] = "gpt-4"
    os.environ["LLM_BASE_URL"] = "https://api-gateway.example.com/v1"
    os.environ["LLM_SSL_VERIFY"] = "false"
    os.environ["LLM_CUSTOM_LLM_PROVIDER"] = "openai"
    os.environ["LLM_EXTRA_HEADERS"] = '{"x-api-key": "secret123"}'

    llm_env = LLM.load_from_env()
    print(f"Created LLM from env with model: {llm_env.model}")
    print(f"Base URL: {llm_env.base_url}")
    print(f"SSL Verify: {llm_env.ssl_verify}")
    print(f"Extra headers: {llm_env.extra_headers}")

    # Clean up environment variables set for this demonstration
    for key in (
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_SSL_VERIFY",
        "LLM_CUSTOM_LLM_PROVIDER",
        "LLM_EXTRA_HEADERS",
    ):
        os.environ.pop(key, None)
