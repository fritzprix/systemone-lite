"""Assemble Jev-shaped answers from option/level probability maps."""

from __future__ import annotations

from systemone_lite.confidence import distribution_confidence
from systemone_lite.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


def assemble_noul(probabilities: dict[str, float]) -> NoulAnswer:
    yes = float(probabilities.get("yes", probabilities.get("true", 0.0)))
    return NoulAnswer(noul=min(1.0, max(0.0, yes)))


def assemble_choice(probabilities: dict[str, float]) -> ChoiceAnswer:
    if not probabilities:
        raise ValueError("choice probabilities must not be empty")
    choice = max(probabilities, key=probabilities.get)
    return ChoiceAnswer(
        choice=choice,
        probabilities=probabilities,
        confidence=distribution_confidence(probabilities),
    )


def assemble_score(
    level_probabilities: dict[str, float],
    legend: dict[str, str],
) -> ScoreAnswer:
    if set(level_probabilities) != set(legend):
        raise ValueError("score probabilities and legend keys must match")
    score = sum(int(idx) * prob for idx, prob in level_probabilities.items())
    return ScoreAnswer(
        score=float(score),
        legend=legend,
        probabilities=level_probabilities,
        confidence=distribution_confidence(level_probabilities),
    )


def expected_symbols(question: Question) -> list[str]:
    """Symbols the model scores for this question (internal next-token labels)."""
    if isinstance(question, NoulQuestion):
        return ["yes", "no"]
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria.keys())
    if isinstance(question, ScoreQuestion):
        return [str(i) for i in range(len(question.criteria))]
    raise TypeError(f"unsupported question type: {type(question)!r}")


def assemble_answer(question: Question, probabilities: dict[str, float]):
    if isinstance(question, NoulQuestion):
        return assemble_noul(probabilities)
    if isinstance(question, ChoiceQuestion):
        return assemble_choice(probabilities)
    if isinstance(question, ScoreQuestion):
        legend = {str(i): label for i, label in enumerate(question.criteria)}
        return assemble_score(probabilities, legend)
    raise TypeError(f"unsupported question type: {type(question)!r}")
