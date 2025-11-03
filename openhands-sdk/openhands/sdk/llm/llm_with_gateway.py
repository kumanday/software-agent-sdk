"""LLM subclass with enterprise gateway support.

This module provides LLMWithGateway, which extends the base LLM class to support
custom headers for enterprise API gateways.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from openhands.sdk.llm.llm import LLM
from openhands.sdk.logger import get_logger


__all__ = ["LLMWithGateway"]


logger = get_logger(__name__)


class LLMWithGateway(LLM):
    """LLM subclass with enterprise gateway support.

    Supports adding static custom headers on each request. Take care not to include
    raw secrets in headers unless the gateway is trusted and headers are never logged.
    """

    custom_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers to include with every LLM request.",
    )

    def _prepare_request_kwargs(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(super()._prepare_request_kwargs(call_kwargs))

        if not self.custom_headers:
            return prepared

        existing = prepared.get("extra_headers")
        base_headers: dict[str, Any]
        if isinstance(existing, Mapping):
            base_headers = dict(existing)
        elif existing is None:
            base_headers = {}
        else:
            base_headers = {}

        merged, collisions = self._merge_headers(base_headers, self.custom_headers)
        for header, old_val, new_val in collisions:
            logger.warning(
                "LLMWithGateway overriding header %s (existing=%r, new=%r)",
                header,
                old_val,
                new_val,
            )

        if merged:
            prepared["extra_headers"] = merged

        return prepared

    @staticmethod
    def _merge_headers(
        existing: dict[str, Any], new_headers: dict[str, Any]
    ) -> tuple[dict[str, Any], list[tuple[str, Any, Any]]]:
        """Merge header dictionaries case-insensitively.

        Returns the merged headers and a list of collisions where an existing
        header was replaced with a different value.
        """

        merged = dict(existing)
        lower_map = {k.lower(): k for k in merged}
        collisions: list[tuple[str, Any, Any]] = []

        for key, value in new_headers.items():
            lower = key.lower()
            if lower in lower_map:
                canonical = lower_map[lower]
                old_value = merged[canonical]
                if old_value != value:
                    collisions.append((canonical, old_value, value))
                merged[canonical] = value
            else:
                merged[key] = value
                lower_map[lower] = key

        return merged, collisions
