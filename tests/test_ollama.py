from datetime import UTC, datetime

from regression_detection.feature import OllamaEmailClassifier
from regression_detection.prompts import PromptConfig


def test_ollama_classifier_uses_local_openai_compatible_endpoint():
    classifier = OllamaEmailClassifier(
        PromptConfig("v1", datetime.now(UTC), "classify"),
        model="llama3.2:3b",
        base_url="http://ollama.test/v1",
    )

    assert classifier.model == "llama3.2:3b"
    assert str(classifier.client.base_url).rstrip("/") == "http://ollama.test/v1"
