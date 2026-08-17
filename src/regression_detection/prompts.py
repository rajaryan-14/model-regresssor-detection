"""Prompt configuration loading and validation."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptConfig:
    version: str
    created_at: datetime
    system_prompt: str
    examples: tuple[dict[str, str], ...] = ()


def load_prompt(path: str | Path) -> PromptConfig:
    """Load one versioned YAML prompt configuration."""

    prompt_path = Path(path)
    with prompt_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise TypeError(f"Prompt file must contain a mapping: {prompt_path}")
    required = {"version", "created_at", "system_prompt"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Prompt file is missing fields: {', '.join(sorted(missing))}")

    examples = raw.get("examples", [])
    if not isinstance(examples, list):
        raise TypeError("Prompt examples must be a list")

    return PromptConfig(
        version=str(raw["version"]),
        created_at=datetime.fromisoformat(str(raw["created_at"])),
        system_prompt=str(raw["system_prompt"]),
        examples=tuple(examples),
    )
