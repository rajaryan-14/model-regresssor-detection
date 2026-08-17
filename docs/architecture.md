# Model Regression Detection: Architecture Notes

## Problem

Prompt and model changes can alter production behavior without changing application code. This system treats model behavior as a versioned artifact and evaluates it against human-verified cases before a change is merged.

## Data flow

```text
versioned prompt + golden dataset
              ↓
      classifier adapter
              ↓
      async evaluation runner
              ↓
  case results + aggregate metrics
       ↙              ↘
 SQLite history       baseline diff
       ↓              ↓
 trend/drift      HTML + Slack + CI status
```

## Design decisions

- Golden labels are stored in JSON and reviewed like code. Failure cases can be added without changing evaluator logic.
- The feature is isolated behind `EmailClassifier`, allowing deterministic fakes in tests and alternate providers later.
- Category accuracy and summary quality are separate dimensions. The default keyword scorer is deterministic; the optional LLM judge is explicitly selected with `--judge-model`.
- A failed model call is retained as a failed case rather than aborting the complete run.
- SQLite stores the complete JSON run plus indexed metadata, keeping reports reproducible while supporting trend queries.
- Per-run regression thresholds and rolling slow-drift detection address different failure modes: sharp changes versus gradual degradation.

## Demo walkthrough

1. Change the system prompt version or wording.
2. Run `model-eval` with an explicit baseline.
3. Open `reports/evaluation.html` and review the scorecard and old/new regression outputs.
4. Show the SQLite history and seven-run drift status.
5. Push the prompt change and show the GitHub Actions report artifact and PR comment.
6. Optionally set `SLACK_WEBHOOK_URL` to demonstrate the alert payload.

## Evaluation caveat

The live LLM judge introduces model cost and judgment variance. CI should use a pinned judge model and retain the deterministic score alongside the judge score when this mode is enabled.
