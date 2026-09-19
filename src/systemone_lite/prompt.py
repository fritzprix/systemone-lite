"""Prompt construction for System One questions."""

from __future__ import annotations

import json
from typing import Any

from systemone_lite.schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
)


def render_state(state: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_criteria(question: Question) -> str:
    if isinstance(question, NoulQuestion):
        lines = ["- yes: affirmative / true", "- no: negative / false"]
        if question.criteria is not None:
            if question.criteria.true:
                lines[0] = f"- yes: {question.criteria.true}"
            if question.criteria.false:
                lines[1] = f"- no: {question.criteria.false}"
        return "\n".join(lines)

    if isinstance(question, ChoiceQuestion):
        lines: list[str] = []
        for key, description in question.criteria.items():
            if description:
                lines.append(f"- {key}: {description}")
            else:
                lines.append(f"- {key}")
        return "\n".join(lines)

    if isinstance(question, ScoreQuestion):
        return "\n".join(
            f"- {idx}: {label}" for idx, label in enumerate(question.criteria)
        )

    raise TypeError(f"unsupported question type: {type(question)!r}")


def build_state_prefix(state: str | dict[str, Any] | list[Any]) -> str:
    """Shared prompt prefix for all questions against the same state."""
    return f"### State\n{render_state(state)}\n\n"


def build_question_suffix(question: Question) -> str:
    """Per-question prompt suffix (appended after the shared state prefix)."""
    answer_hint = {
        NoulQuestion: "Reply with exactly one token: yes or no.",
        ChoiceQuestion: "Reply with exactly one option key from Criteria.",
        ScoreQuestion: "Reply with exactly one level index from Criteria.",
    }[type(question)]

    return (
        "### Question\n"
        f"{question.instructions}\n\n"
        "### Criteria\n"
        f"{render_criteria(question)}\n\n"
        f"### Instructions\n{answer_hint}\n\n"
        "### Answer\n"
    )


def build_prompt(state: str | dict[str, Any] | list[Any], question: Question) -> str:
    return build_state_prefix(state) + build_question_suffix(question)


def build_prompts(request: SystemOneRequest) -> dict[str, str]:
    return {
        qid: build_prompt(request.state, question)
        for qid, question in request.questions.items()
    }
