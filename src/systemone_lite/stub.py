from __future__ import annotations

from systemone_lite.assemble import assemble_answer, expected_symbols
from systemone_lite.schema import SystemOneRequest, SystemOneResponse, Usage


class StubEngine:
    """Deterministic engine for schema/API tests without model weights."""

    concrete_model_id = "stub-systemone-lite"

    def decide(self, request: SystemOneRequest) -> SystemOneResponse:
        answers = {}
        for qid, question in request.questions.items():
            symbols = expected_symbols(question)
            n = len(symbols)
            # Slightly peaked on the first symbol for stable fixtures.
            probs = {symbols[0]: 0.7}
            rest = (1.0 - 0.7) / max(n - 1, 1)
            for symbol in symbols[1:]:
                probs[symbol] = rest
            if n == 1:
                probs[symbols[0]] = 1.0
            answers[qid] = assemble_answer(question, probs)

        # Rough token estimate for usage field.
        input_tokens = max(1, len(str(request.state).split()) + 20 * len(request.questions))
        return SystemOneResponse(
            model=self.concrete_model_id,
            answers=answers,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        )
