"""SQLite persistence for evaluation run history."""

import sqlite3
from pathlib import Path

from .evaluation import EvaluationRun


class RunStore:
    """Persist complete evaluation runs and query recent trend data."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    prompt_version TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    pass_rate REAL NOT NULL,
                    category_accuracy REAL NOT NULL,
                    average_summary_score REAL NOT NULL,
                    failed_cases INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def save(self, run: EvaluationRun) -> None:
        """Insert or replace a run by its stable run ID."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, prompt_version, model, dataset_version,
                    started_at, completed_at, pass_rate, category_accuracy,
                    average_summary_score, failed_cases, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.prompt_version,
                    run.model,
                    run.dataset_version,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat(),
                    run.metrics.pass_rate,
                    run.metrics.category_accuracy,
                    run.metrics.average_summary_score,
                    run.metrics.failed_cases,
                    run.model_dump_json(),
                ),
            )

    def get(self, run_id: str) -> EvaluationRun | None:
        """Return one run, or ``None`` when it is not present."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM evaluation_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return EvaluationRun.model_validate_json(row["payload_json"]) if row else None

    def recent(
        self,
        limit: int = 10,
        *,
        prompt_version: str | None = None,
        model: str | None = None,
        dataset_version: str | None = None,
    ) -> list[EvaluationRun]:
        """Return compatible runs in chronological order for trend charts."""

        if limit < 1:
            raise ValueError("limit must be at least one")
        clauses: list[str] = []
        parameters: list[str | int] = []
        for column, value in (
            ("prompt_version", prompt_version),
            ("model", model),
            ("dataset_version", dataset_version),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT payload_json FROM evaluation_runs
                {where}
                ORDER BY completed_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [EvaluationRun.model_validate_json(row["payload_json"]) for row in reversed(rows)]
