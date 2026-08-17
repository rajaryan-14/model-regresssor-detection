from datetime import UTC, datetime

from regression_detection.alerting import SlackNotifier, build_slack_payload
from regression_detection.evaluation import BaselineComparison, EvaluationRun, RunMetrics


def make_run():
    return EvaluationRun(
        run_id="run-123",
        prompt_version="v1",
        model="fake",
        dataset_version="v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        cases=[],
        metrics=RunMetrics(
            pass_rate=0.9,
            category_accuracy=0.95,
            average_summary_score=0.8,
            failed_cases=1,
            per_category_accuracy={},
        ),
    )


def test_slack_payload_contains_headline_and_report_link():
    payload = build_slack_payload(
        make_run(),
        BaselineComparison(
            status="warning",
            pass_rate_delta=-0.1,
            category_accuracy_deltas={},
            regressions=["case-1"],
            improvements=[],
        ),
        "https://example.test/report.html",
    )

    assert "WARNING" in payload["text"]
    assert "case-1" not in payload["text"]
    assert "https://example.test/report.html" in payload["text"]
    assert payload["blocks"][0]["type"] == "section"


def test_slack_notifier_posts_json_payload():
    captured = {}

    def fake_post(url, body):
        captured["url"] = url
        captured["body"] = body

    notifier = SlackNotifier("https://hooks.slack.test", post=fake_post)
    notifier.notify(
        make_run(),
        BaselineComparison(
            status="pass",
            pass_rate_delta=0,
            category_accuracy_deltas={},
            regressions=[],
            improvements=[],
        ),
    )

    assert captured["url"] == "https://hooks.slack.test"
    assert b"PASS" in captured["body"]
