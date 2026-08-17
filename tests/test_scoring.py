from types import SimpleNamespace

import pytest

from regression_detection.scoring import KeywordSummaryScorer, OpenAISummaryJudge


def test_keyword_summary_scorer_is_normalized_and_case_insensitive():
    scorer = KeywordSummaryScorer()

    assert scorer.score("Customer reports a duplicate charge.", "Reports duplicate CHARGE.") == 1.0
    assert 0 < scorer.score("Customer reports a duplicate charge.", "Customer reports a billing charge.") < 1
    assert scorer.score("A question.", "") == 0.0


def test_keyword_summary_scorer_validates_threshold():
    try:
        KeywordSummaryScorer(pass_threshold=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid thresholds should be rejected")


class FakeJudgeClient:
    class Chat:
        class Completions:
            async def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"score": 5}'))]
                )

        completions = Completions()

    chat = Chat()


@pytest.mark.asyncio
async def test_openai_summary_judge_normalizes_one_to_five_score():
    judge = OpenAISummaryJudge(client=FakeJudgeClient())

    assert await judge.score("Customer cannot log in.", "Customer cannot log in.") == 1.0
