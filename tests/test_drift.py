from datetime import UTC, datetime

import pytest

from regression_detection.evaluation import EvaluationRun, RunMetrics, detect_slow_drift


def make_run(index: int, pass_rate: float) -> EvaluationRun:
    return EvaluationRun(
        run_id=f"run-{index}",
        prompt_version="v1",
        model="fake",
        dataset_version="v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        cases=[],
        metrics=RunMetrics(
            pass_rate=pass_rate,
            category_accuracy=pass_rate,
            average_summary_score=pass_rate,
            failed_cases=0,
            per_category_accuracy={},
        ),
    )


def test_slow_drift_requires_a_full_window():
    result = detect_slow_drift([make_run(1, 0.5)], window=7)

    assert result.status == "insufficient_data"
    assert result.average_pass_rate is None
    assert result.runs_considered == 1


def test_slow_drift_warns_when_rolling_average_drops():
    history = [make_run(index, 0.8) for index in range(7)]

    result = detect_slow_drift(history, window=7, threshold=0.9)

    assert result.status == "warning"
    assert result.average_pass_rate == pytest.approx(0.8)


def test_slow_drift_is_stable_above_threshold():
    history = [make_run(index, 0.95) for index in range(7)]

    result = detect_slow_drift(history)

    assert result.status == "stable"
    assert result.average_pass_rate == pytest.approx(0.95)
