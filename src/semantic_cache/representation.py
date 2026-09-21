"""Shared construction of the text used for semantic lookup."""

from __future__ import annotations

from collections.abc import Sequence

from semantic_cache.models import ChatMessage


def build_semantic_lookup_text(messages: Sequence[ChatMessage]) -> str:
    """Preserve non-system roles exactly as the cache sees them."""
    return "\n".join(
        f"{message.role}: {message.content}"
        for message in messages
        if message.role != "system"
    )


def build_single_user_lookup_text(query: str) -> str:
    """Represent an evaluation query as the API's single-user-message request."""
    return build_semantic_lookup_text((ChatMessage(role="user", content=query),))
