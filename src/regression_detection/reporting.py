"""Self-contained HTML reporting for evaluation runs."""

from collections.abc import Iterable
from html import escape
from pathlib import Path

from .evaluation import (
    BaselineComparison,
    DriftDetection,
    EvaluationRun,
    compare_runs,
    detect_slow_drift,
)


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _trend_svg(history: Iterable[EvaluationRun], current: EvaluationRun) -> str:
    runs = [run for run in history if run.run_id != current.run_id] + [current]
    if not runs:
        return "<p>No trend data available.</p>"
    width, height, padding = 720, 180, 24
    points = []
    for index, run in enumerate(runs):
        x = padding if len(runs) == 1 else padding + index * (width - 2 * padding) / (len(runs) - 1)
        y = height - padding - run.metrics.pass_rate * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    labels = "".join(
        f'<text x="{padding + (index * (width - 2 * padding) / max(1, len(runs) - 1)):.1f}" '
        f'y="{height - 4}" text-anchor="middle">{escape(run.run_id[:8])}</text>'
        for index, run in enumerate(runs)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Pass-rate trend">'
        f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="3"/>'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#cbd5e1"/>'
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#cbd5e1"/>'
        f'<text x="4" y="{padding + 4}">100%</text><text x="4" y="{height - padding + 4}">0%</text>'
        f"{labels}</svg>"
    )


def render_html_report(
    current: EvaluationRun,
    *,
    baseline: EvaluationRun | None = None,
    comparison: BaselineComparison | None = None,
    history: Iterable[EvaluationRun] = (),
    drift: DriftDetection | None = None,
) -> str:
    """Render a complete HTML report without external assets or network calls."""

    if comparison is None and baseline is not None:
        comparison = compare_runs(current, baseline)
    comparison = comparison or BaselineComparison(
        status="pass",
        pass_rate_delta=0.0,
        category_accuracy_deltas={},
        regressions=[],
        improvements=[],
    )
    baseline_cases = {case.case_id: case for case in baseline.cases} if baseline else {}
    regression_rows = []
    for case_id in comparison.regressions:
        new_case = next(case for case in current.cases if case.case_id == case_id)
        old_case = baseline_cases.get(case_id)
        regression_rows.append(
            "<tr>"
            f"<td>{escape(case_id)}</td>"
            f"<td>{escape(old_case.actual.model_dump_json() if old_case and old_case.actual else old_case.error if old_case else 'Unavailable')}</td>"
            f"<td>{escape(new_case.actual.model_dump_json() if new_case.actual else new_case.error or 'Unavailable')}</td>"
            "</tr>"
        )
    rows = "".join(regression_rows) or '<tr><td colspan="3">No regressions detected.</td></tr>'
    category_rows = "".join(
        f"<tr><td>{escape(category)}</td><td>{_percentage(current.metrics.per_category_accuracy.get(category, 0))}</td>"
        f"<td>{delta:+.1%}</td></tr>"
        for category, delta in comparison.category_accuracy_deltas.items()
    ) or '<tr><td colspan="3">No baseline category comparison.</td></tr>'
    trend = _trend_svg(history, current)
    drift = drift or detect_slow_drift([*history, current])
    drift_average = _percentage(drift.average_pass_rate) if drift.average_pass_rate is not None else "n/a"
    status = escape(comparison.status.upper())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Model Evaluation Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}}
.scorecard{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{border:1px solid #dbe2ea;border-radius:8px;padding:16px}} .value{{font-size:1.7rem;font-weight:700}}
table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border:1px solid #dbe2ea;padding:9px;text-align:left;vertical-align:top}}th{{background:#f1f5f9}}
code{{white-space:pre-wrap;word-break:break-word}} svg{{width:100%;max-height:220px;border:1px solid #dbe2ea}}
</style></head><body>
<h1>Model Evaluation Report</h1>
<p>Status: <strong>{status}</strong> · Prompt <code>{escape(current.prompt_version)}</code> · Model <code>{escape(current.model)}</code> · Dataset <code>{escape(current.dataset_version)}</code></p>
<section class="scorecard">
<div class="card">Pass rate<div class="value">{_percentage(current.metrics.pass_rate)}</div></div>
<div class="card">Category accuracy<div class="value">{_percentage(current.metrics.category_accuracy)}</div></div>
<div class="card">Summary score<div class="value">{_percentage(current.metrics.average_summary_score)}</div></div>
<div class="card">Failed cases<div class="value">{current.metrics.failed_cases}</div></div>
</section>
<h2>Baseline comparison</h2><p>Pass-rate delta: <strong>{comparison.pass_rate_delta:+.1%}</strong> · Regressions: <strong>{len(comparison.regressions)}</strong> · Improvements: <strong>{len(comparison.improvements)}</strong></p>
<h2>Regressed cases</h2><table><thead><tr><th>Case</th><th>Previous output</th><th>Current output</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Category accuracy</h2><table><thead><tr><th>Category</th><th>Current</th><th>Delta</th></tr></thead><tbody>{category_rows}</tbody></table>
<h2>Pass-rate trend</h2>{trend}
<h2>Slow drift</h2><p>Status: <strong>{escape(drift.status.upper())}</strong> · Rolling window: {drift.window} runs · Average pass rate: {drift_average} · Threshold: {_percentage(drift.threshold)}</p>
<h2>Run metadata</h2><p>Run ID: <code>{escape(current.run_id)}</code><br>Completed: {escape(current.completed_at.isoformat())}</p>
</body></html>"""


def write_html_report(
    current: EvaluationRun,
    path: str | Path,
    *,
    baseline: EvaluationRun | None = None,
    comparison: BaselineComparison | None = None,
    history: Iterable[EvaluationRun] = (),
    drift: DriftDetection | None = None,
) -> Path:
    """Write a rendered report and return its path."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_html_report(current, baseline=baseline, comparison=comparison, history=history, drift=drift),
        encoding="utf-8",
    )
    return report_path
