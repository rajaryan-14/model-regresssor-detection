"""Versioned golden-dataset models and loading helpers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import EmailClassification, SupportEmail

Difficulty = Literal["easy", "medium", "hard"]


class GoldenCase(BaseModel):
    """One human-labelled evaluation example."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1)
    email: SupportEmail = Field(validation_alias="input")
    expected: EmailClassification
    expected_difficulty: Difficulty
    notes: str = Field(min_length=1)

    @field_validator("email", mode="before")
    @classmethod
    def string_input_becomes_email(cls, value):
        if isinstance(value, str):
            return {"text": value}
        return value

    @field_validator("id")
    @classmethod
    def id_must_be_stable(cls, value: str) -> str:
        return value.strip()


class GoldenDataset(BaseModel):
    """A versioned collection of golden cases."""

    version: str = Field(min_length=1)
    created_at: datetime
    cases: list[GoldenCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def ids_must_be_unique(cls, cases: list[GoldenCase]) -> list[GoldenCase]:
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Golden case IDs must be unique")
        return cases


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    """Load and validate a JSON golden dataset."""

    dataset_path = Path(path)
    with dataset_path.open(encoding="utf-8") as handle:
        return GoldenDataset.model_validate(json.load(handle))
