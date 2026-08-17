# Model Regression Detection

This project evaluates an LLM-powered customer-support email classifier against a human-verified golden dataset. Prompt or model changes can be evaluated locally and, later, in CI before they reach users.

## Current scope

The initial slice defines the contracts that the evaluation engine will consume:

- versioned YAML prompt configurations in `prompts/`;
- a Pydantic input/output contract for the classifier;
- an OpenAI-backed feature adapter with dependency injection for tests;
- versioned JSON golden data in `data/golden/`.

The evaluator, scoring, reports, Slack notifications, and GitHub Actions workflow are built on top of these interfaces.

## Run history and reports

Evaluation runs can be stored in SQLite and later used for trend data:

```python
from regression_detection import RunStore, write_html_report

store = RunStore("eval_runs/history.sqlite")
store.save(run)
baseline = store.get("previous-run-id")
history = store.recent(7, prompt_version=run.prompt_version, model=run.model)
write_html_report(run, "reports/evaluation.html", baseline=baseline, history=history)
```

The HTML report is self-contained: it includes the scorecard, baseline regressions, per-category deltas, and an inline pass-rate trend chart. No web server or external charting dependency is required.

The CLI is available after installation:

```powershell
model-eval --prompt prompts/v1.yaml --dataset data/golden/v1.json
```

Set `OPENAI_API_KEY` to run the live OpenAI adapter. Set `SLACK_WEBHOOK_URL` to send an optional Slack alert; the webhook is never required for local evaluation.

For semantic summary judging, add `--judge-model gpt-4o-mini`. Without this flag, the evaluator uses the deterministic keyword scorer and does not make additional judge requests.

The evaluator also computes a slow-drift signal from the latest seven compatible runs. It reports `insufficient_data` until a complete window exists, then warns when the rolling pass-rate average falls below 90%.

## GitHub Actions setup

The workflow in `.github/workflows/evaluate-prompts.yml` runs for pull requests that change prompts, golden data, evaluator code, or project dependencies. Add these repository secrets under **Settings → Secrets and variables → Actions**:

- `OPENAI_API_KEY` — required for the live classifier evaluation;
- `SLACK_WEBHOOK_URL` — optional; enables Slack alerts.

The workflow always runs the offline test suite, uploads the HTML and Markdown reports, and comments the Markdown scorecard on the pull request. Local runs can compare against an explicit baseline with `--baseline path/to/run.json`; CI runs without a baseline until a trusted baseline artifact or committed baseline is configured.

## Container usage

```powershell
docker build -t model-regression-detection .
docker run --rm -e OPENAI_API_KEY=$env:OPENAI_API_KEY model-regression-detection
```

Mount a directory when retaining SQLite history or reports outside the container:

```powershell
docker run --rm -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -v "${PWD}/eval_runs:/app/eval_runs" -v "${PWD}/reports:/app/reports" `
  model-regression-detection --db eval_runs/history.sqlite --report reports/evaluation.html
```

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

To call the OpenAI adapter, set `OPENAI_API_KEY`. The adapter expects the model to return a JSON object with `category` and `summary` fields.

## Adding golden cases

Add a case to the next dataset version in `data/golden/`. Each case must have a stable `id`, input email, expected category, ideal summary, difficulty, and a note explaining why the case belongs in the set. Labels are human-authored ground truth; generated examples must be reviewed before inclusion.

The current v1 dataset contains 25 cases spanning all four categories, including typos, sarcasm, mixed-language input, ambiguous requests, and extremely short messages.

## Architecture direction

The LLM provider is isolated behind a small async adapter so the evaluator can be tested without network calls. Prompt files are data rather than Python code, which makes prompt changes visible to version control and CI. SQLite and JSON will remain the default persistence layer to keep local runs reproducible and portable.
