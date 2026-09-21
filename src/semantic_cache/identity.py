"""Cache identity: the execution context that must match for reuse to be safe.

Semantic similarity decides *which* cached prompt is closest. Identity decides
whether that prompt is even *eligible*. Keep the two concerns separate: the
embedding never sees the model name, the system prompt, or sampling settings,
so those must be enforced here rather than hoped for in vector distance.
"""

from __future__ import annotations

from hashlib import sha256
import json

from semantic_cache.models import CacheIdentity, ChatRequest


def hash_system_prompt(system_prompt: str) -> str:
    """Digest used to isolate entries and to invalidate them by system prompt."""
    return sha256(system_prompt.encode("utf-8")).hexdigest()


def system_prompt_of(request: ChatRequest) -> str:
    return "\n".join(m.content for m in request.messages if m.role == "system")


def _params_hash(request: ChatRequest) -> str:
    params = {
        "top_p": request.top_p,
        "seed": request.seed,
        "stop": list(request.stop) if request.stop is not None else None,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
    }
    present = {name: value for name, value in params.items() if value is not None}
    if not present:
        return ""
    canonical = json.dumps(present, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_cache_identity(request: ChatRequest) -> CacheIdentity:
    return CacheIdentity(
        provider=request.provider,
        model=request.model,
        system_prompt_hash=hash_system_prompt(system_prompt_of(request)),
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        params_hash=_params_hash(request),
    )
