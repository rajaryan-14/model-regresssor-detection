from datetime import UTC, datetime

from regression_detection.prompts import PromptConfig, load_prompt


def test_load_versioned_prompt():
    prompt = load_prompt("prompts/v1.yaml")

    assert prompt.version == "v1"
    assert prompt.created_at == datetime(2026, 8, 17, tzinfo=UTC)
    assert "exactly one category" in prompt.system_prompt
    assert prompt.examples == ()


def test_prompt_config_is_immutable():
    prompt = PromptConfig("v1", datetime.now(UTC), "classify")

    try:
        prompt.version = "v2"
    except AttributeError:
        pass
    else:
        raise AssertionError("PromptConfig should be immutable")
