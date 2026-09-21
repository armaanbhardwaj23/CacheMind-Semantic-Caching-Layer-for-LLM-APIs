import pytest

from semantic_cache.models import ChatMessage, ChatRequest
from semantic_cache.policy import CachePolicy


def request(text: str, temperature: float | None = None) -> ChatRequest:
    return ChatRequest(
        provider="p", model="m", messages=[ChatMessage(role="user", content=text)], temperature=temperature
    )


@pytest.mark.parametrize(
    "prompt",
    ["What is the weather in Paris?", "What's the current price of Bitcoin?", "Who is winning right now?", "Any breaking news today?"],
)
def test_realtime_prompts_bypass_the_cache(prompt) -> None:
    decision = CachePolicy().decide(request(prompt))
    assert not decision.cacheable and decision.reason == "realtime"


@pytest.mark.parametrize("prompt", ["What is the latest Python version?", "Summarise events from last year.", "Plans for next month?"])
def test_time_relative_prompts_get_the_short_ttl(prompt) -> None:
    decision = CachePolicy(time_relative_ttl_seconds=600).decide(request(prompt))
    assert decision.cacheable and decision.ttl_seconds == 600 and decision.reason == "time_relative"


def test_stable_prompts_get_the_long_ttl() -> None:
    decision = CachePolicy(stable_ttl_seconds=999).decide(request("What is a mutex?"))
    assert decision.cacheable and decision.ttl_seconds == 999 and decision.reason == "stable"


def test_only_the_last_user_message_is_classified() -> None:
    req = ChatRequest(
        provider="p",
        model="m",
        messages=[
            ChatMessage(role="system", content="Answer with today's date in mind."),
            ChatMessage(role="user", content="What is a mutex?"),
        ],
    )
    assert CachePolicy().decide(req).reason == "stable"


def test_optional_temperature_ceiling() -> None:
    policy = CachePolicy(max_temperature=0.5)
    assert not policy.decide(request("hi", temperature=0.9)).cacheable
    assert policy.decide(request("hi", temperature=0.2)).cacheable
    assert policy.decide(request("hi", temperature=None)).cacheable  # provider default is unknown


def test_classifier_can_be_disabled() -> None:
    decision = CachePolicy(classify_time_sensitivity=False).decide(request("weather today"))
    assert decision.cacheable and decision.reason == "stable"
