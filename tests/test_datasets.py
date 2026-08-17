import json

import pytest
from pydantic import ValidationError

from regression_detection.datasets import GoldenDataset, load_golden_dataset


def test_load_golden_dataset():
    dataset = load_golden_dataset("data/golden/v1.json")

    assert dataset.version == "v1"
    assert len(dataset.cases) == 50
    assert dataset.cases[0].email.text.startswith("Why was I charged")


def test_dataset_rejects_duplicate_case_ids():
    payload = {
        "version": "v1",
        "created_at": "2026-08-17T00:00:00+00:00",
        "cases": [
            {
                "id": "same",
                "input": "one",
                "expected": {"category": "general", "summary": "A question."},
                "expected_difficulty": "easy",
                "notes": "test",
            },
            {
                "id": "same",
                "input": "two",
                "expected": {"category": "general", "summary": "Another question."},
                "expected_difficulty": "easy",
                "notes": "test",
            },
        ],
    }

    with pytest.raises(ValidationError):
        GoldenDataset.model_validate(payload)


def test_dataset_loader_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": "v1"}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_golden_dataset(path)
