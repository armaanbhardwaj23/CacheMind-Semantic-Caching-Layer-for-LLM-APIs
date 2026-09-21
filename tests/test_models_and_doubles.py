import pytest

from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.testing import DeterministicEmbedder, RecordingProvider


def test_request_copies_messages_into_an_immutable_sequence() -> None:
    messages = [ChatMessage(role="user", content="Hello")]

    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=messages,
        temperature=0.0,
        max_tokens=20,
    )
    messages.append(ChatMessage(role="user", content="Changed later"))

    assert request.messages == (ChatMessage(role="user", content="Hello"),)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"provider": "", "model": "test", "messages": [ChatMessage("user", "Hi")]}, "Provider"),
        ({"provider": "test", "model": "", "messages": [ChatMessage("user", "Hi")]}, "Model"),
        ({"provider": "test", "model": "test", "messages": []}, "message"),
    ],
)
def test_request_rejects_missing_identity_or_messages(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ChatRequest(**kwargs, temperature=0.0, max_tokens=20)  # type: ignore[arg-type]


def test_deterministic_embedder_repeats_identical_text() -> None:
    embedder = DeterministicEmbedder()

    assert embedder.embed("same request") == embedder.embed("same request")
    assert embedder.embed("same request") != embedder.embed("different request")


def test_recording_provider_records_request_and_preserves_model() -> None:
    request = ChatRequest(
        provider="test-provider",
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.0,
        max_tokens=20,
    )
    provider = RecordingProvider(response_text="Hello back")

    completion = provider.complete(request)

    assert completion.content == "Hello back"
    assert completion.model == "test-model"
    assert provider.requests == [request]
    assert provider.call_count == 1
