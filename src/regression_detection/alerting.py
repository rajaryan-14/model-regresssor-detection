"""Optional Slack notifications for evaluation results."""

import json
import urllib.request
from collections.abc import Callable

from .evaluation import BaselineComparison, DriftDetection, EvaluationRun


def build_slack_payload(
    current: EvaluationRun,
    comparison: BaselineComparison,
    report_url: str | None = None,
    drift: DriftDetection | None = None,
) -> dict:
    """Build a Slack incoming-webhook payload without performing network I/O."""

    status_emoji = {"pass": ":white_check_mark:", "warning": ":warning:", "critical": ":rotating_light:"}
    report_suffix = f"\n<{report_url}|Open full HTML report>" if report_url else ""
    drift_suffix = f"\nSlow drift: {drift.status}" if drift else ""
    text = (
        f"{status_emoji[comparison.status]} *Model evaluation {comparison.status.upper()}*\n"
        f"Pass rate: {current.metrics.pass_rate:.1%} "
        f"({comparison.pass_rate_delta:+.1%})\n"
        f"Regressions: {len(comparison.regressions)} · "
        f"Improvements: {len(comparison.improvements)} · "
        f"Failed cases: {current.metrics.failed_cases}"
        f"{drift_suffix}{report_suffix}"
    )
    return {
        "text": text,
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Prompt `{current.prompt_version}` · Model `{current.model}` · Dataset `{current.dataset_version}`",
                    }
                ],
            },
        ],
    }


class SlackNotifier:
    """Send evaluation alerts to a Slack incoming webhook."""

    def __init__(self, webhook_url: str, post: Callable[[str, bytes], None] | None = None):
        if not webhook_url:
            raise ValueError("webhook_url must not be empty")
        self.webhook_url = webhook_url
        self._post = post or self._post_json

    def notify(
        self,
        current: EvaluationRun,
        comparison: BaselineComparison,
        report_url: str | None = None,
        drift: DriftDetection | None = None,
    ) -> None:
        payload = json.dumps(build_slack_payload(current, comparison, report_url, drift)).encode("utf-8")
        self._post(self.webhook_url, payload)

    @staticmethod
    def _post_json(url: str, payload: bytes) -> None:
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status >= 300:
                raise RuntimeError(f"Slack webhook returned HTTP {response.status}")
