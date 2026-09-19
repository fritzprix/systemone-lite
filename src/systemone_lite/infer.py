"""Batched next-token inference for System One questions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from systemone_lite.assemble import assemble_answer, expected_symbols
from systemone_lite.prompt import build_question_suffix, build_state_prefix
from systemone_lite.schema import (
    Answer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 4096
MODEL_ALIASES: dict[str, str] = {
    "systemone-lite-latest": DEFAULT_MODEL_ID,
    "systemone-lite-preview": DEFAULT_MODEL_ID,
    # Temporary aliases from the former project name.
    "gpt2-jev-latest": DEFAULT_MODEL_ID,
    "gpt2-jev-preview": DEFAULT_MODEL_ID,
}


def resolve_model_id(model: str) -> str:
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    if model.startswith("jev-"):
        # Accept TypeSafe-looking ids but serve our default backbone.
        return DEFAULT_MODEL_ID
    return model


@dataclass
class LoadedModel:
    model_id: str
    resolved_id: str
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: torch.device


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str = DEFAULT_MODEL_ID) -> LoadedModel:
    resolved = resolve_model_id(model_id)
    device = _pick_device()
    logger.info("Loading %s on %s", resolved, device)

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is safer when continuing from a shared prefix cache.
    tokenizer.padding_side = "right"

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        dtype=dtype,
        trust_remote_code=True,
    )
    try:
        model.to(device)
    except torch.OutOfMemoryError:
        if device.type == "cpu":
            raise
        logger.warning("CUDA OOM while loading; falling back to CPU")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = torch.device("cpu")
        model.to(device)
    model.eval()
    return LoadedModel(
        model_id=model_id,
        resolved_id=resolved,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def _encode_symbol(
    tokenizer: PreTrainedTokenizerBase,
    symbol: str,
) -> int:
    """Map an answer symbol to a single vocabulary token id."""
    candidates = [symbol, f" {symbol}", f"\n{symbol}"]
    for candidate in candidates:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]

    ids = tokenizer.encode(symbol, add_special_tokens=False)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for symbol {symbol!r}")
    logger.warning(
        "Symbol %r is multi-token (%s); using first token id only",
        symbol,
        ids,
    )
    return ids[0]


def _softmax_over_ids(
    logits: torch.Tensor,
    token_ids: list[int],
) -> list[float]:
    selected = logits[token_ids]
    probs = torch.softmax(selected.float(), dim=-1)
    return [float(p) for p in probs.tolist()]


def _repeat_past_key_values(past_key_values: Any, batch_size: int) -> Any:
    """Expand a batch-1 KV cache to batch_size by repeating along batch dim."""
    if past_key_values is None or batch_size == 1:
        return past_key_values

    # transformers>=4.5x DynamicCache
    if hasattr(past_key_values, "batch_repeat_interleave"):
        past_key_values.batch_repeat_interleave(batch_size)
        return past_key_values

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:  # pragma: no cover
        DynamicCache = None  # type: ignore[misc, assignment]

    if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            repeated = tuple(
                (key.repeat(batch_size, 1, 1, 1), value.repeat(batch_size, 1, 1, 1))
                for key, value in legacy
            )
            return DynamicCache.from_legacy_cache(repeated)

    # Legacy tuple[(k, v), ...]
    return tuple(
        (key.repeat(batch_size, 1, 1, 1), value.repeat(batch_size, 1, 1, 1))
        for key, value in past_key_values
    )


def _suffix_position_ids(
    prefix_len: int,
    suffix_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Absolute position ids for suffix tokens given a shared prefix length."""
    batch, suffix_len = suffix_attention_mask.shape
    # Cumulative positions within the suffix, then offset by prefix_len.
    # Pad positions stay unused by the model thanks to attention_mask.
    suffix_pos = suffix_attention_mask.long().cumsum(dim=1) - 1
    suffix_pos = suffix_pos.clamp(min=0)
    return suffix_pos + prefix_len


class SystemOneEngine:
    def __init__(self, loaded: LoadedModel, *, use_prefix_cache: bool = True) -> None:
        self._loaded = loaded
        self.use_prefix_cache = use_prefix_cache

    @property
    def concrete_model_id(self) -> str:
        return self._loaded.resolved_id

    def decide(self, request: SystemOneRequest) -> SystemOneResponse:
        items = list(request.questions.items())
        if not items:
            return SystemOneResponse(
                model=self.concrete_model_id,
                answers={},
                usage=Usage(input_tokens=0, output_tokens=0),
            )

        if self.use_prefix_cache and len(items) >= 2:
            return self._decide_with_prefix_cache(request, items)
        return self._decide_naive_batch(request, items)

    def _decide_naive_batch(
        self,
        request: SystemOneRequest,
        items: list[tuple[str, Any]],
    ) -> SystemOneResponse:
        from systemone_lite.prompt import build_prompt

        prompts = [build_prompt(request.state, question) for _, question in items]
        encoded = self._loaded.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded = {key: value.to(self._loaded.device) for key, value in encoded.items()}
        total_input_tokens = int(encoded["attention_mask"].sum().item())

        with torch.inference_mode():
            outputs = self._loaded.model(**encoded)
            last_idx = encoded["attention_mask"].sum(dim=1) - 1
            batch_logits = outputs.logits[
                torch.arange(outputs.logits.size(0), device=outputs.logits.device),
                last_idx,
            ]

        return self._assemble(items, batch_logits, total_input_tokens)

    def _decide_with_prefix_cache(
        self,
        request: SystemOneRequest,
        items: list[tuple[str, Any]],
    ) -> SystemOneResponse:
        tokenizer = self._loaded.tokenizer
        device = self._loaded.device
        batch_size = len(items)

        prefix_text = build_state_prefix(request.state)
        suffixes = [build_question_suffix(question) for _, question in items]

        # Reserve room for the longest suffix so the shared state still fits.
        suffix_token_lens = [
            len(tokenizer.encode(suffix, add_special_tokens=False)) for suffix in suffixes
        ]
        max_suffix_len = max(suffix_token_lens)
        prefix_budget = max(16, MAX_LENGTH - max_suffix_len)

        prefix_ids = tokenizer(
            prefix_text,
            return_tensors="pt",
            truncation=True,
            max_length=prefix_budget,
            add_special_tokens=True,
        )["input_ids"].to(device)
        prefix_len = int(prefix_ids.shape[1])
        prefix_mask = torch.ones((1, prefix_len), dtype=torch.long, device=device)

        suffix_encoded = tokenizer(
            suffixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(8, MAX_LENGTH - prefix_len),
            add_special_tokens=False,
        )
        suffix_ids = suffix_encoded["input_ids"].to(device)
        suffix_mask = suffix_encoded["attention_mask"].to(device)

        # Logical token usage: state once + each question suffix (Jev-like).
        total_input_tokens = prefix_len + int(suffix_mask.sum().item())

        with torch.inference_mode():
            prefix_out = self._loaded.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
            )
            past = _repeat_past_key_values(prefix_out.past_key_values, batch_size)

            full_mask = torch.cat(
                [
                    torch.ones((batch_size, prefix_len), dtype=torch.long, device=device),
                    suffix_mask,
                ],
                dim=1,
            )
            position_ids = _suffix_position_ids(prefix_len, suffix_mask)

            suffix_out = self._loaded.model(
                input_ids=suffix_ids,
                attention_mask=full_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=False,
            )

            last_idx = suffix_mask.sum(dim=1) - 1
            batch_logits = suffix_out.logits[
                torch.arange(batch_size, device=device),
                last_idx,
            ]

        return self._assemble(items, batch_logits, total_input_tokens)

    def _assemble(
        self,
        items: list[tuple[str, Any]],
        batch_logits: torch.Tensor,
        total_input_tokens: int,
    ) -> SystemOneResponse:
        answers: dict[str, Answer] = {}
        for row, (qid, question) in enumerate(items):
            symbols = expected_symbols(question)
            token_ids = [
                _encode_symbol(self._loaded.tokenizer, symbol) for symbol in symbols
            ]
            probs_list = _softmax_over_ids(batch_logits[row], token_ids)
            probabilities = {
                symbol: prob for symbol, prob in zip(symbols, probs_list, strict=True)
            }
            answers[qid] = assemble_answer(question, probabilities)

        return SystemOneResponse(
            model=self.concrete_model_id,
            answers=answers,
            usage=Usage(input_tokens=total_input_tokens, output_tokens=0),
        )


_ENGINE: SystemOneEngine | None = None


def get_engine(model_id: str = DEFAULT_MODEL_ID) -> SystemOneEngine:
    global _ENGINE
    resolved = resolve_model_id(model_id)
    if _ENGINE is None or _ENGINE.concrete_model_id != resolved:
        _ENGINE = SystemOneEngine(load_model(model_id), use_prefix_cache=True)
    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    _ENGINE = None
