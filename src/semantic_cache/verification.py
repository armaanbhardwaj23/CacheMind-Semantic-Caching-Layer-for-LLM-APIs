"""Small, deterministic guard for semantic-cache candidates.

This is intentionally a rejection-only policy.  Embedding similarity proposes a
candidate; the guard blocks a few observable answer-changing differences.  It
does not establish that two prompts have the same meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


@dataclass(frozen=True)
class VerificationDecision:
    accepted: bool
    reason: str | None = None


class CandidateVerifier(Protocol):
    def verify(self, *, cached_text: str, incoming_text: str) -> VerificationDecision: ...


class AcceptAllVerifier:
    """No guard: similarity alone decides. The baseline the ConstraintGuard is compared against."""

    def verify(self, *, cached_text: str, incoming_text: str) -> VerificationDecision:
        return VerificationDecision(True)


class ConstraintGuard:
    """Reject candidates with simple, explicit constraint mismatches.

    The patterns are deliberately transparent rather than pretending to be a
    general natural-language verifier.  They are useful as a cheap defence in
    front of a provider call and as a baseline against a later learned verifier.
    """

    _negations = frozenset({"no", "not", "never", "without", "except"})
    # Capitalised only because they start a sentence: wh-words, auxiliaries, pronoun-like
    # openers and common imperative verbs. Any *other* sentence-initial capitalised word is
    # still compared as a possible name ("Paris is ..." vs "London is ...").
    _sentence_starters = frozenset(
        {
            "A", "An", "The", "Please", "If", "Let",
            "Who", "Whom", "Whose", "Which", "What", "When", "Where", "Why", "How",
            "Is", "Are", "Was", "Were", "Am", "Do", "Does", "Did", "Can", "Could", "Should",
            "Would", "Will", "May", "Might", "Must", "Has", "Have", "Had",
            "Calculate", "Compare", "Convert", "Create", "Define", "Describe", "Explain", "Find",
            "Fix", "Generate", "Give", "List", "Name", "Plan", "Rank", "Recommend", "Return",
            "Send", "Show", "Solve", "Suggest", "Summarize", "Tell", "Translate", "Write",
        }
    )
    _role_prefix = re.compile(r"^(?:user|assistant):\s*", re.MULTILINE)
    _sentence_split = re.compile(r"(?<=[.!?])\s+|\n+")
    _capitalised = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
    # Bounded so a long text of repeated "from" cannot make the search quadratic (a 40 KB prompt took 2.4 s).
    _direction_pattern = re.compile(
        r"\bfrom\s+([^\n]{1,200}?)\s+to\s+([^?.!,;\n]{1,200})", re.IGNORECASE
    )
    _ordering_pattern = re.compile(
        r"\b(lowest\s+to\s+highest|highest\s+to\s+lowest)\b", re.IGNORECASE
    )

    def verify(self, *, cached_text: str, incoming_text: str) -> VerificationDecision:
        checks = (
            ("numbers", _numbers),
            ("negation", _negation_tokens),
            ("named values", self._named_values),
            ("directions", self._directions),
            ("ordering", self._ordering),
        )
        for label, extractor in checks:
            if extractor(cached_text) != extractor(incoming_text):
                return VerificationDecision(False, f"{label} differ")
        return VerificationDecision(True)

    def _named_values(self, text: str) -> frozenset[str]:
        names: set[str] = set()
        for sentence in self._sentence_split.split(self._role_prefix.sub("", text)):
            sentence = sentence.lstrip(" \t\n\"'\u201c\u2018([{-*\u2022")  # opening quote, bracket or bullet
            for match in self._capitalised.finditer(sentence):
                word = match.group()
                if word == "I":  # the pronoun, not a name
                    continue
                if match.start() == 0 and word in self._sentence_starters:
                    continue
                names.add(word)
        return frozenset(names)

    def _directions(self, text: str) -> tuple[tuple[str, str], ...]:
        return tuple(
            (origin.strip().lower(), destination.strip().lower())
            for origin, destination in self._direction_pattern.findall(text)
        )

    def _ordering(self, text: str) -> tuple[str, ...]:
        return tuple(match.lower() for match in self._ordering_pattern.findall(text))


def _numbers(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _negation_tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\b\w+\b", text.lower())) & ConstraintGuard._negations
