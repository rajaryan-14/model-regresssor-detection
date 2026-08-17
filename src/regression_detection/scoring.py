"""Deterministic scoring interfaces used by the evaluator."""

import re
from collections.abc import Awaitable
from typing import Protocol

from openai import AsyncOpenAI


class SummaryScorer(Protocol):
    def score(self, expected: str, actual: str) -> float | Awaitable[float]:
        """Return a normalized summary relevance score between zero and one."""


class KeywordSummaryScorer:
    """Score summaries by overlap of meaningful words with the reference summary."""

    _stop_words = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "because",
            "customer",
            "for",
            "from",
            "in",
            "is",
            "of",
            "on",
            "that",
            "the",
            "to",
            "with",
        }
    )

    def __init__(self, pass_threshold: float = 0.5):
        if not 0 <= pass_threshold <= 1:
            raise ValueError("pass_threshold must be between zero and one")
        self.pass_threshold = pass_threshold

    def score(self, expected: str, actual: str) -> float:
        expected_words = self._meaningful_words(expected)
        actual_words = self._meaningful_words(actual)
        if not expected_words or not actual_words:
            return 0.0
        return len(expected_words & actual_words) / len(expected_words)

    @classmethod
    def _meaningful_words(cls, text: str) -> set[str]:
        return {
            word
            for word in re.findall(r"[a-z0-9]+", text.lower())
            if word not in cls._stop_words and len(word) > 2
        }


class OpenAISummaryJudge:
    """Optional model-based summary judge returning a normalized 0–1 score."""

    def __init__(self, model: str = "gpt-4o-mini", client: AsyncOpenAI | None = None):
        self.model = model
        self.client = client or AsyncOpenAI()
        self.pass_threshold = 0.5

    async def score(self, expected: str, actual: str) -> float:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation judge. Rate how well the candidate summary "
                        "captures the reference summary. Return only JSON: {\"score\": N}, "
                        "where N is an integer from 1 (irrelevant) to 5 (excellent)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Reference summary:\n{expected}\n\nCandidate summary:\n{actual}",
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Summary judge returned an empty response")
        import json

        raw_score = json.loads(content).get("score")
        if not isinstance(raw_score, (int, float)) or not 1 <= raw_score <= 5:
            raise ValueError("Summary judge score must be between 1 and 5")
        return (float(raw_score) - 1) / 4
