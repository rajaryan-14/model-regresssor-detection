import asyncio
from dataclasses import dataclass

import pytest

from regression_detection.contracts import EmailClassification
from regression_detection.datasets import load_golden_dataset
from regression_detection.evaluation import (
    ComparisonThresholds,
    EvaluationRun,
    EvaluationRunner,
    compare_runs,
)


@dataclass
class FakeClassifier:
    responses: dict[str, object]
    active: int = 0
    max_active: int = 0
    delay: float = 0

    async def classify(self, email):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            response = self.responses[email.text]
            if isinstance(response, Exception):
                raise response
            return response
        finally:
            self.active -= 1


def expected_responses(dataset):
    return {case.email.text: case.expected for case in dataset.cases}


@pytest.mark.asyncio
async def test_runner_evaluates_all_cases_with_bounded_concurrency():
    dataset = load_golden_dataset("data/golden/v1.json")
    classifier = FakeClassifier(expected_responses(dataset), delay=0.01)

    run = await EvaluationRunner(
        classifier, prompt_version="v1", model="fake", max_concurrency=2
    ).run(dataset)

    assert [case.case_id for case in run.cases] == [case.id for case in dataset.cases]
    assert run.metrics.pass_rate == 1.0
    assert run.metrics.category_accuracy == 1.0
    assert run.metrics.failed_cases == 0
    assert classifier.max_active == 2
    assert set(run.metrics.per_category_accuracy) == {"billing", "technical", "account", "general"}


@pytest.mark.asyncio
async def test_runner_records_errors_and_invalid_outputs():
    dataset = load_golden_dataset("data/golden/v1.json")
    responses = expected_responses(dataset)
    responses[dataset.cases[0].email.text] = RuntimeError("provider unavailable")
    responses[dataset.cases[1].email.text] = {"category": "not-valid", "summary": "bad"}

    run = await EvaluationRunner(
        FakeClassifier(responses), prompt_version="v1", model="fake"
    ).run(dataset)

    assert run.metrics.failed_cases == 2
    assert run.cases[0].status == "failed"
    assert "provider unavailable" in run.cases[0].error
    assert run.cases[1].actual is None
    assert run.cases[1].error.startswith("ValidationError:")


@pytest.mark.asyncio
async def test_runner_requires_category_and_summary_quality():
    dataset = load_golden_dataset("data/golden/v1.json")
    responses = expected_responses(dataset)
    original = responses[dataset.cases[0].email.text]
    responses[dataset.cases[0].email.text] = EmailClassification(
        category=original.category, summary="Unrelated status update."
    )

    run = await EvaluationRunner(
        FakeClassifier(responses), prompt_version="v1", model="fake"
    ).run(dataset)

    assert run.cases[0].category_match is True
    assert run.cases[0].summary_score == 0.0
    assert run.cases[0].status == "failed"


@pytest.mark.asyncio
async def test_run_json_round_trip_and_baseline_comparison(tmp_path):
    dataset = load_golden_dataset("data/golden/v1.json")
    baseline = await EvaluationRunner(
        FakeClassifier(expected_responses(dataset)), prompt_version="v1", model="fake"
    ).run(dataset)
    changed = expected_responses(dataset)
    changed[dataset.cases[0].email.text] = RuntimeError("regressed")
    current = await EvaluationRunner(
        FakeClassifier(changed), prompt_version="v2", model="fake"
    ).run(dataset)

    path = tmp_path / "baseline.json"
    baseline.save_json(path)
    loaded = EvaluationRun.load_json(path)
    comparison = compare_runs(current, loaded)

    assert loaded.run_id == baseline.run_id
    assert comparison.status == "pass"
    assert comparison.regressions == ["billing-001"]
    assert comparison.improvements == []
    assert comparison.pass_rate_delta == pytest.approx(-1 / 50)


def test_comparison_threshold_boundaries():
    from datetime import UTC, datetime

    from regression_detection.evaluation import EvaluationRun, RunMetrics

    thresholds = ComparisonThresholds(warning_delta=0.03, critical_delta=0.08)
    assert thresholds.warning_delta == 0.03
    assert thresholds.critical_delta == 0.08
    metrics = RunMetrics(
        pass_rate=1.0,
        category_accuracy=1.0,
        average_summary_score=1.0,
        failed_cases=0,
        per_category_accuracy={},
    )
    baseline = EvaluationRun(
        run_id="baseline",
        prompt_version="v1",
        model="fake",
        dataset_version="v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        cases=[],
        metrics=metrics,
    )
    warning_current = baseline.model_copy(
        update={"metrics": metrics.model_copy(update={"pass_rate": 0.97})}
    )
    critical_current = baseline.model_copy(
        update={"metrics": metrics.model_copy(update={"pass_rate": 0.92})}
    )
    assert compare_runs(warning_current, baseline, thresholds).status == "warning"
    assert compare_runs(critical_current, baseline, thresholds).status == "critical"
    with pytest.raises(ValueError):
        invalid = ComparisonThresholds(warning_delta=0.08, critical_delta=0.03)
        invalid.validate_order()
