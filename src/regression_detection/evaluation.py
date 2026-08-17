"""Offline evaluation runner, result models, and baseline comparison."""

import asyncio
import inspect
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .contracts import EmailClassification
from .datasets import GoldenDataset
from .feature import EmailClassifier
from .scoring import KeywordSummaryScorer, SummaryScorer

CaseStatus = Literal["passed", "failed"]
RunStatus = Literal["pass", "warning", "critical"]
DriftStatus = Literal["stable", "warning", "insufficient_data"]


class CaseResult(BaseModel):
    case_id: str
    input: str
    expected: EmailClassification
    actual: EmailClassification | None = None
    category_match: bool = False
    summary_score: float = Field(default=0.0, ge=0, le=1)
    latency_ms: float = Field(ge=0)
    status: CaseStatus
    error: str | None = None


class RunMetrics(BaseModel):
    pass_rate: float = Field(ge=0, le=1)
    category_accuracy: float = Field(ge=0, le=1)
    average_summary_score: float = Field(ge=0, le=1)
    failed_cases: int = Field(ge=0)
    per_category_accuracy: dict[str, float]


class EvaluationRun(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    dataset_version: str
    started_at: datetime
    completed_at: datetime
    cases: list[CaseResult]
    metrics: RunMetrics

    def save_json(self, path: str | Path) -> None:
        """Persist this run as readable JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "EvaluationRun":
        """Load a previously persisted run."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class ComparisonThresholds(BaseModel):
    warning_delta: float = Field(default=0.03, ge=0, le=1)
    critical_delta: float = Field(default=0.08, ge=0, le=1)

    def validate_order(self) -> None:
        if self.critical_delta < self.warning_delta:
            raise ValueError("critical_delta must be greater than or equal to warning_delta")


class BaselineComparison(BaseModel):
    status: RunStatus
    pass_rate_delta: float
    category_accuracy_deltas: dict[str, float]
    regressions: list[str]
    improvements: list[str]


class DriftDetection(BaseModel):
    status: DriftStatus
    window: int
    runs_considered: int
    average_pass_rate: float | None = Field(default=None, ge=0, le=1)
    threshold: float = Field(ge=0, le=1)


def detect_slow_drift(
    history: Iterable[EvaluationRun],
    *,
    window: int = 7,
    threshold: float = 0.9,
) -> DriftDetection:
    """Detect sustained degradation using a rolling pass-rate average."""

    if window < 1:
        raise ValueError("window must be at least one")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    runs = list(history)[-window:]
    if len(runs) < window:
        return DriftDetection(
            status="insufficient_data",
            window=window,
            runs_considered=len(runs),
            threshold=threshold,
        )
    average = sum(run.metrics.pass_rate for run in runs) / len(runs)
    return DriftDetection(
        status="warning" if average < threshold else "stable",
        window=window,
        runs_considered=len(runs),
        average_pass_rate=average,
        threshold=threshold,
    )


def compare_runs(
    current: EvaluationRun,
    baseline: EvaluationRun,
    thresholds: ComparisonThresholds | None = None,
) -> BaselineComparison:
    """Compare two runs using stable case IDs and configurable percentage deltas."""

    thresholds = thresholds or ComparisonThresholds()
    thresholds.validate_order()
    baseline_cases = {case.case_id: case for case in baseline.cases}
    current_cases = {case.case_id: case for case in current.cases}

    regressions = sorted(
        case_id
        for case_id, old in baseline_cases.items()
        if case_id in current_cases and old.status == "passed" and current_cases[case_id].status == "failed"
    )
    improvements = sorted(
        case_id
        for case_id, old in baseline_cases.items()
        if case_id in current_cases and old.status == "failed" and current_cases[case_id].status == "passed"
    )
    drop = -min(0.0, current.metrics.pass_rate - baseline.metrics.pass_rate)
    epsilon = 1e-12
    if drop + epsilon >= thresholds.critical_delta:
        status: RunStatus = "critical"
    elif drop + epsilon >= thresholds.warning_delta:
        status = "warning"
    else:
        status = "pass"

    categories = set(baseline.metrics.per_category_accuracy) | set(current.metrics.per_category_accuracy)
    category_deltas = {
        category: current.metrics.per_category_accuracy.get(category, 0.0)
        - baseline.metrics.per_category_accuracy.get(category, 0.0)
        for category in sorted(categories)
    }
    return BaselineComparison(
        status=status,
        pass_rate_delta=current.metrics.pass_rate - baseline.metrics.pass_rate,
        category_accuracy_deltas=category_deltas,
        regressions=regressions,
        improvements=improvements,
    )


class EvaluationRunner:
    """Run a classifier against every case in a golden dataset."""

    def __init__(
        self,
        classifier: EmailClassifier,
        *,
        prompt_version: str,
        model: str,
        max_concurrency: int = 8,
        summary_scorer: SummaryScorer | None = None,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        self.classifier = classifier
        self.prompt_version = prompt_version
        self.model = model
        self.max_concurrency = max_concurrency
        self.summary_scorer = summary_scorer or KeywordSummaryScorer()

    async def run(self, dataset: GoldenDataset) -> EvaluationRun:
        started_at = datetime.now(UTC)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate(case):
            async with semaphore:
                start = time.perf_counter()
                try:
                    raw_actual = await self.classifier.classify(case.email)
                    actual = EmailClassification.model_validate(raw_actual)
                    category_match = actual.category == case.expected.category
                    summary_score = self.summary_scorer.score(
                        case.expected.summary, actual.summary
                    )
                    if inspect.isawaitable(summary_score):
                        summary_score = await summary_score
                    status: CaseStatus = (
                        "passed"
                        if category_match
                        and summary_score >= getattr(self.summary_scorer, "pass_threshold", 0.5)
                        else "failed"
                    )
                    error = None
                except Exception as exc:  # noqa: BLE001 - preserve one bad response and continue.
                    actual = None
                    category_match = False
                    summary_score = 0.0
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                return CaseResult(
                    case_id=case.id,
                    input=case.email.text,
                    expected=case.expected,
                    actual=actual,
                    category_match=category_match,
                    summary_score=summary_score,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    status=status,
                    error=error,
                )

        results = await asyncio.gather(*(evaluate(case) for case in dataset.cases))
        completed_at = datetime.now(UTC)
        return EvaluationRun(
            run_id=str(uuid4()),
            prompt_version=self.prompt_version,
            model=self.model,
            dataset_version=dataset.version,
            started_at=started_at,
            completed_at=completed_at,
            cases=results,
            metrics=_calculate_metrics(results, dataset),
        )


def _calculate_metrics(results: list[CaseResult], dataset: GoldenDataset) -> RunMetrics:
    total = len(results)
    passed = sum(result.status == "passed" for result in results)
    category_matches = sum(result.category_match for result in results)
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for result, case in zip(results, dataset.cases, strict=True):
        by_category[case.expected.category].append(result)
    return RunMetrics(
        pass_rate=passed / total,
        category_accuracy=category_matches / total,
        average_summary_score=sum(result.summary_score for result in results) / total,
        failed_cases=total - passed,
        per_category_accuracy={
            category: sum(result.category_match for result in category_results) / len(category_results)
            for category, category_results in sorted(by_category.items())
        },
    )
