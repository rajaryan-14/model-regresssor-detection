"""Command-line entry point for running and reporting an evaluation."""

import argparse
import os
from pathlib import Path

from .alerting import SlackNotifier
from .datasets import load_golden_dataset
from .evaluation import EvaluationRun, EvaluationRunner, compare_runs, detect_slow_drift
from .feature import OpenAIEmailClassifier
from .prompts import load_prompt
from .reporting import write_html_report
from .scoring import OpenAISummaryJudge
from .storage import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a prompt against the golden dataset")
    parser.add_argument("--prompt", required=True, help="Versioned YAML prompt file")
    parser.add_argument("--dataset", required=True, help="Versioned JSON golden dataset")
    parser.add_argument("--db", default="eval_runs/history.sqlite", help="SQLite history path")
    parser.add_argument("--report", default="reports/evaluation.html", help="HTML report path")
    parser.add_argument("--summary-file", help="Optional Markdown summary output path")
    parser.add_argument("--baseline", help="Explicit baseline JSON run path")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", help="Optional OpenAI model for LLM-as-judge summary scoring")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--slack-webhook-url", default=os.getenv("SLACK_WEBHOOK_URL"))
    parser.add_argument("--report-url", default=os.getenv("EVALUATION_REPORT_URL"))
    return parser


async def run(args: argparse.Namespace) -> int:
    prompt = load_prompt(args.prompt)
    dataset = load_golden_dataset(args.dataset)
    judge = OpenAISummaryJudge(args.judge_model) if args.judge_model else None
    current = await EvaluationRunner(
        OpenAIEmailClassifier(prompt, model=args.model),
        prompt_version=prompt.version,
        model=args.model,
        max_concurrency=args.max_concurrency,
        summary_scorer=judge,
    ).run(dataset)

    store = RunStore(args.db)
    store.save(current)
    baseline = EvaluationRun.load_json(args.baseline) if args.baseline else None
    if baseline is None:
        compatible = store.recent(
            2, prompt_version=prompt.version, model=args.model, dataset_version=dataset.version
        )
        baseline = next((run for run in reversed(compatible) if run.run_id != current.run_id), None)
    comparison = compare_runs(current, baseline) if baseline else None
    history = store.recent(
        7, prompt_version=prompt.version, model=args.model, dataset_version=dataset.version
    )
    drift = detect_slow_drift(history)
    write_html_report(current, args.report, baseline=baseline, comparison=comparison, history=history, drift=drift)

    if comparison is not None and args.slack_webhook_url:
        SlackNotifier(args.slack_webhook_url).notify(current, comparison, args.report_url, drift)
    if args.summary_file:
        summary = [
            f"## Model evaluation: {(comparison.status if comparison else 'no-baseline').upper()}",
            f"- Pass rate: {current.metrics.pass_rate:.1%}",
            f"- Category accuracy: {current.metrics.category_accuracy:.1%}",
            f"- Failed cases: {current.metrics.failed_cases}",
            f"- Slow drift: {drift.status}",
        ]
        if comparison:
            summary.extend(
                [
                    f"- Pass-rate delta: {comparison.pass_rate_delta:+.1%}",
                    f"- Regressions: {len(comparison.regressions)}",
                    f"- Improvements: {len(comparison.improvements)}",
                ]
            )
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Evaluation {current.run_id}: {current.metrics.pass_rate:.1%} pass")
    if comparison:
        print(f"Comparison status: {comparison.status}; regressions: {len(comparison.regressions)}")
    print(f"Slow drift: {drift.status}")
    return 1 if comparison and comparison.status == "critical" else 0


def main() -> None:
    import asyncio

    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
