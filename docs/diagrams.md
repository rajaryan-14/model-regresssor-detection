# System Diagrams

## 1. System architecture

```mermaid
flowchart LR
    P[Versioned prompt YAML] --> F[Classifier adapter]
    D[Golden dataset JSON] --> E[Async evaluation runner]
    F --> E
    E --> S[Category and summary scoring]
    S --> R[Evaluation run JSON]
    R --> DB[(SQLite run history)]
    R --> B[Baseline comparison]
    DB --> T[Trend and slow-drift detection]
    B --> H[HTML report]
    T --> H
    B --> SL[Slack alert]
    H --> CI[GitHub Actions artifact]
```

## 2. Evaluation flow

```mermaid
sequenceDiagram
    participant Runner as Evaluation runner
    participant Dataset as Golden dataset
    participant Model as Ollama/OpenAI adapter
    participant Scorer as Scoring layer
    participant Store as SQLite
    participant Report as Report generator

    Runner->>Dataset: Load and validate cases
    loop Each case with bounded concurrency
        Runner->>Model: Classify support email
        Model-->>Runner: Structured category and summary
        Runner->>Scorer: Score category and summary
        Scorer-->>Runner: Case result
    end
    Runner->>Store: Save complete evaluation run
    Runner->>Store: Load compatible baseline/history
    Runner->>Report: Generate diff and trend report
    Report-->>Runner: HTML and Markdown artifacts
```

## 3. CI/CD regression workflow

```mermaid
flowchart TD
    A[Developer changes prompt] --> PR[Open pull request]
    PR --> T[Run offline tests]
    T --> O[Install Ollama and qwen2.5vl:3b]
    O --> E[Evaluate 50 golden cases]
    E --> C{Compare with baseline}
    C -->|Pass| P[Post passing PR summary]
    C -->|Warning| W[Post warning and report artifact]
    C -->|Critical| X[Fail workflow and block merge]
    E --> R[Upload HTML, JSON, and Markdown reports]
    E --> S[Optional Slack notification]
```

## Talking points

- The prompt and dataset are versioned inputs, so behavior changes are reviewable.
- The evaluator is provider-agnostic: the same interface supports Ollama locally and OpenAI when credits are available.
- Baseline comparison catches sharp regressions, while the seven-run rolling average catches slow drift.
- Failed model calls are retained as failed cases instead of hiding the problem by aborting the run.
