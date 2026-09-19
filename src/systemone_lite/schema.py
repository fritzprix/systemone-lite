"""Jev-compatible System One request/response schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: str | None = None
    false: str | None = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: str
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str | None] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _non_empty_keys(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        if not value:
            raise ValueError("choice criteria must contain at least one option")
        if any(not key for key in value):
            raise ValueError("choice criteria keys must be non-empty")
        return value


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: str
    criteria: list[str] = Field(min_length=2, max_length=10)

    @field_validator("criteria")
    @classmethod
    def _non_empty_levels(cls, value: list[str]) -> list[str]:
        if any(not level.strip() for level in value):
            raise ValueError("score criteria levels must be non-empty strings")
        return value


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "systemone-lite-latest"
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)

    @field_validator("questions", mode="before")
    @classmethod
    def _questions_must_be_map(cls, value: Any) -> Any:
        if isinstance(value, list):
            raise ValueError("questions must be a map/object, not an array")
        return value

    @model_validator(mode="after")
    def _non_empty_question_ids(self) -> SystemOneRequest:
        if any(not key for key in self.questions):
            raise ValueError("question ids must be non-empty")
        return self


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0, default=0)


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError("choice must be a key in probabilities")
        return self


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> ScoreAnswer:
        if set(self.legend) != set(self.probabilities):
            raise ValueError("legend and probabilities keys must match")
        return self


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    answers: dict[str, Answer]
    usage: Usage
