"""Typed contracts shared by the feature and evaluation pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]


class SupportEmail(BaseModel):
    """A customer-support email presented to the model."""

    text: str = Field(min_length=1)


class EmailClassification(BaseModel):
    """The structured response required from the model."""

    category: Category
    summary: str = Field(min_length=1)
