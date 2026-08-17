from dataclasses import dataclass

import pytest

from regression_detection.datasets import load_golden_dataset
from regression_detection.evaluation import EvaluationRunner
from regression_detection.reporting import render_html_report, write_html_report
from regression_detection.storage import RunStore


@dataclass
class FixedClassifier:
    fail_first: bool = False

    async def classify(self, email):
        dataset = load_golden_dataset("data/golden/v1.json")
        for case in dataset.cases:
            if case.email.text == email.text:
                if self.fail_first and case.id == "billing-001":
                    raise RuntimeError("simulated regression")
                return case.expected
        raise LookupError(email.text)


@pytest.mark.asyncio
async def test_run_store_saves_loads_and_filters_recent_runs(tmp_path):
    dataset = load_golden_dataset("data/golden/v1.json")
    store = RunStore(tmp_path / "history.sqlite")
    run = await EvaluationRunner(
        FixedClassifier(), prompt_version="v1", model="fake"
    ).run(dataset)
    store.save(run)

    assert store.get(run.run_id).run_id == run.run_id
    assert [item.run_id for item in store.recent(5, model="fake")] == [run.run_id]
    assert store.recent(5, model="other") == []
    with pytest.raises(ValueError):
        store.recent(0)


@pytest.mark.asyncio
async def test_html_report_contains_scorecard_regression_and_trend(tmp_path):
    dataset = load_golden_dataset("data/golden/v1.json")
    baseline = await EvaluationRunner(
        FixedClassifier(), prompt_version="v1", model="fake"
    ).run(dataset)
    current = await EvaluationRunner(
        FixedClassifier(fail_first=True), prompt_version="v2", model="fake"
    ).run(dataset)

    html = render_html_report(current, baseline=baseline, history=[baseline])
    assert "PASS" in html
    assert "billing-001" in html
    assert "Previous output" in html
    assert "Pass-rate trend" in html
    assert "<svg" in html

    report_path = write_html_report(current, tmp_path / "reports" / "report.html", baseline=baseline)
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith("<!doctype html>")
