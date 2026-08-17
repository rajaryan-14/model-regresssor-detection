"""Model regression detection package."""

from .alerting import SlackNotifier, build_slack_payload
from .contracts import EmailClassification, SupportEmail
from .datasets import GoldenCase, GoldenDataset, load_golden_dataset
from .evaluation import (
    BaselineComparison,
    CaseResult,
    ComparisonThresholds,
    DriftDetection,
    EvaluationRun,
    EvaluationRunner,
    RunMetrics,
    compare_runs,
    detect_slow_drift,
)
from .feature import EmailClassifier, OllamaEmailClassifier, OpenAIEmailClassifier
from .prompts import PromptConfig, load_prompt
from .reporting import render_html_report, write_html_report
from .scoring import KeywordSummaryScorer, OpenAISummaryJudge
from .storage import RunStore

__all__ = [
    "BaselineComparison",
    "CaseResult",
    "ComparisonThresholds",
    "DriftDetection",
    "EmailClassification",
    "EmailClassifier",
    "EvaluationRun",
    "EvaluationRunner",
    "GoldenCase",
    "GoldenDataset",
    "KeywordSummaryScorer",
    "OllamaEmailClassifier",
    "OpenAIEmailClassifier",
    "OpenAISummaryJudge",
    "PromptConfig",
    "RunMetrics",
    "RunStore",
    "SlackNotifier",
    "SupportEmail",
    "build_slack_payload",
    "compare_runs",
    "detect_slow_drift",
    "load_golden_dataset",
    "load_prompt",
    "render_html_report",
    "write_html_report",
]
