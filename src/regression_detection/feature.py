"""Customer-support email classification feature."""

from typing import Protocol

from openai import AsyncOpenAI

from .contracts import EmailClassification, SupportEmail
from .prompts import PromptConfig


class EmailClassifier(Protocol):
    async def classify(self, email: SupportEmail) -> EmailClassification:
        """Classify one support email."""


class OpenAIEmailClassifier:
    """OpenAI implementation of the feature contract."""

    def __init__(self, prompt: PromptConfig, model: str = "gpt-4o-mini", client: AsyncOpenAI | None = None):
        self.prompt = prompt
        self.model = model
        self.client = client or AsyncOpenAI()

    async def classify(self, email: SupportEmail) -> EmailClassification:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.prompt.system_prompt},
                {"role": "user", "content": email.text},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned an empty classification")
        return EmailClassification.model_validate_json(content)
