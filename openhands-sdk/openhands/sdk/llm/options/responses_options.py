from __future__ import annotations

from typing import Any

from openhands.sdk.llm.options.common import apply_defaults_if_absent
from openhands.sdk.llm.utils.model_features import get_features


def _is_subscription_codex_transport(base_url: str | None) -> bool:
    """Check if this is a ChatGPT subscription Codex transport."""
    if not base_url:
        return False
    base = base_url.lower()
    return "chatgpt.com" in base and "backend-api" in base and "codex" in base


def select_responses_options(
    llm,
    user_kwargs: dict[str, Any],
    *,
    include: list[str] | None,
    store: bool | None,
) -> dict[str, Any]:
    """Behavior-preserving extraction of _normalize_responses_kwargs."""
    is_codex_subscription = _is_subscription_codex_transport(llm.base_url)

    # Apply defaults for keys that are not forced by policy
    out = apply_defaults_if_absent(
        user_kwargs,
        {
            "max_output_tokens": llm.max_output_tokens,
        },
    )

    # For Codex subscription, use minimal options to avoid validation errors
    if is_codex_subscription:
        # Codex subscription doesn't support these parameters
        out.pop("max_output_tokens", None)
        out.pop("temperature", None)
        out.pop("tool_choice", None)
        out.pop("reasoning", None)
        out.pop("include", None)
        out.pop("prompt_cache_retention", None)
        # Codex requires store=false
        out["store"] = False
        # Codex backend requires streaming
        out["stream"] = True
        # Extra headers from llm config
        if llm.extra_headers is not None and "extra_headers" not in out:
            out["extra_headers"] = dict(llm.extra_headers)
        # Pass through user-provided extra_body unchanged
        if llm.litellm_extra_body:
            out["extra_body"] = llm.litellm_extra_body
        return out

    # Enforce sampling/tool behavior for Responses path (non-Codex subscription)
    out["temperature"] = 1.0
    out["tool_choice"] = "auto"

    # If user didn't set extra_headers, propagate from llm config
    if llm.extra_headers is not None and "extra_headers" not in out:
        out["extra_headers"] = dict(llm.extra_headers)

    # Store defaults to False (stateless) unless explicitly provided
    if store is not None:
        out["store"] = bool(store)
    else:
        out.setdefault("store", False)

    # Include encrypted reasoning only when the user enables it on the LLM,
    # and only for stateless calls (store=False). Respect user choice.
    include_list = list(include) if include is not None else []

    if not out.get("store", False) and llm.enable_encrypted_reasoning:
        if "reasoning.encrypted_content" not in include_list:
            include_list.append("reasoning.encrypted_content")
    if include_list:
        out["include"] = include_list

    # Include reasoning effort only if explicitly set
    if llm.reasoning_effort:
        out["reasoning"] = {"effort": llm.reasoning_effort}
        # Optionally include summary if explicitly set (requires verified org)
        if llm.reasoning_summary:
            out["reasoning"]["summary"] = llm.reasoning_summary

    # Send prompt_cache_retention only if model supports it
    if (
        get_features(llm.model).supports_prompt_cache_retention
        and llm.prompt_cache_retention
    ):
        out["prompt_cache_retention"] = llm.prompt_cache_retention

    # Pass through user-provided extra_body unchanged
    if llm.litellm_extra_body:
        out["extra_body"] = llm.litellm_extra_body

    return out
