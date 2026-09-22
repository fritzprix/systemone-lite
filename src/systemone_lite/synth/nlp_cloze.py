"""NLP cloze (masked-word multiple choice) for System One distill.

Uses WikiText-2 with **document-level** train/eval separation so eval passages
never appear in train. Fits alias-CE: one blank, four closed options.

Distractors: same-ish length content words from the split's vocabulary
(grammatical but contextually wrong). Short / underspecified sentences are
filtered out.
"""

from __future__ import annotations

import re
import random
from functools import lru_cache
from typing import Any, Literal

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample

Split = Literal["train", "eval"]

_STOP = frozenset(
    """
    a an the and or but if then else when while of to in on at by for from
    with without into onto over under about as is are was were be been being
    it its this that these those he she they them we you i my your his her
    their our not no nor so than too very can could should would will may
    might must shall do does did done have has had having just also only
    than then there here where what which who whom whose how why whom
    """.split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_HEADER = re.compile(r"^\s*=+\s*.*\s*=+\s*$")


def _normalize_line(text: str) -> str:
    return " ".join(text.replace("@-@", "-").replace("@.@", ".").split())


def _is_content(word: str) -> bool:
    w = word.lower()
    if w in _STOP or len(w) < 4:
        return False
    if not w.isalpha():
        return False
    return True


def _sentences_from_rows(rows: list[str]) -> list[str]:
    """Flatten WikiText rows into usable sentences; drop headers/empties."""
    out: list[str] = []
    buf: list[str] = []
    for raw in rows:
        line = _normalize_line(raw or "")
        if not line or _HEADER.match(line):
            if buf:
                block = " ".join(buf)
                buf = []
                out.extend(_split_sentences(block))
            continue
        buf.append(line)
    if buf:
        out.extend(_split_sentences(" ".join(buf)))
    # Filter: enough context for a non-ambiguous cloze
    kept: list[str] = []
    for s in out:
        toks = _TOKEN.findall(s)
        if len(toks) < 10 or len(toks) > 48:
            continue
        if sum(1 for t in toks if _is_content(t)) < 3:
            continue
        # Prefer sentences with a clear content noun/verb mid-span
        kept.append(s.strip())
    return kept


def _split_sentences(block: str) -> list[str]:
    parts = _SENT_SPLIT.split(block)
    return [p.strip() for p in parts if len(p.strip()) >= 40]


@lru_cache(maxsize=4)
def _load_wikitext_sentences(split: Split) -> tuple[str, ...]:
    """Load WikiText-2 sentences. train←train, eval←test (document-disjoint)."""
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "test"
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=hf_split)
    texts = list(ds["text"])
    sents = _sentences_from_rows(texts)
    if len(sents) < 200:
        raise RuntimeError(
            f"wikitext {hf_split}: only {len(sents)} usable sentences after filter"
        )
    return tuple(sents)


def _vocab_from_sentences(sentences: list[str] | tuple[str, ...]) -> dict[int, list[str]]:
    """Length-bucketed content vocabulary for distractors."""
    buckets: dict[int, set[str]] = {}
    for s in sentences:
        for tok in _TOKEN.findall(s):
            if not _is_content(tok):
                continue
            low = tok.lower()
            buckets.setdefault(len(low), set()).add(low)
    return {k: sorted(v) for k, v in buckets.items()}


def _pick_mask_index(rng: random.Random, tokens: list[str]) -> int | None:
    """Mask a content word away from the edges (stronger local context)."""
    candidates = [
        i
        for i, t in enumerate(tokens)
        if _is_content(t) and 2 <= i <= len(tokens) - 3
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _distractors(
    rng: random.Random,
    answer: str,
    vocab: dict[int, list[str]],
    k: int = 3,
) -> list[str]:
    ans = answer.lower()
    n = len(ans)
    # Sample from nearby length buckets — O(1) per draw, not O(|V|).
    candidates: list[str] = []
    for width in (0, 1, 2, 3):
        for length in (n - width, n + width) if width else (n,):
            if length < 4:
                continue
            bucket = vocab.get(length)
            if not bucket:
                continue
            # Cap draws per bucket to keep this cheap on large WikiText vocabs.
            take = min(len(bucket), 32)
            if take == len(bucket):
                picks = list(bucket)
            else:
                picks = rng.sample(bucket, take)
            for w in picks:
                if w != ans and w not in candidates:
                    candidates.append(w)
        if len(candidates) >= 24:
            break
    rng.shuffle(candidates)
    out = candidates[:k]
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    while len(out) < k:
        letters = list(ans)
        i = rng.randrange(len(letters))
        letters[i] = rng.choice([c for c in alphabet if c != letters[i]])
        cand = "".join(letters)
        if cand != ans and cand not in out:
            out.append(cand)
    return out


def _cloze_from_sentence(
    rng: random.Random,
    sentence: str,
    vocab: dict[int, list[str]],
    *,
    hard: bool,
    split: Split,
    doc_id: int,
) -> DistillSample | None:
    tokens = _TOKEN.findall(sentence)
    idx = _pick_mask_index(rng, tokens)
    if idx is None:
        return None
    answer = tokens[idx]
    # Rebuild passage with blank (preserve non-token chars roughly via join)
    # Use a simple whitespace reconstruction of matched tokens for clarity.
    blanked = list(tokens)
    blanked[idx] = "[___]"
    passage = " ".join(blanked)
    # Restore sentence-final punctuation if present on original
    if sentence.rstrip().endswith((".", "!", "?")):
        if not passage.endswith((".", "!", "?")):
            passage = passage + sentence.rstrip()[-1]

    distract = _distractors(rng, answer, vocab, k=3)
    opts = {answer.lower(): answer.lower()}
    for d in distract:
        opts[d] = d

    # Reject if any distractor equals answer (safety)
    if len(opts) < 4:
        return None

    state: dict[str, Any] = {
        "sentence": passage,
        "legend": "Select the most contextually appropriate word to fill in the blank.",
    }
    return choice_sample(
        task="nlp.cloze",
        state=state,
        instructions=(
            "Which word correctly completes the sentence? "
            "Reply with one option letter."
        ),
        options=opts,
        label_key=answer.lower(),
        meta={
            "gym": "nlp_cloze",
            "kind": "cloze",
            "corpus": "wikitext-2-raw-v1",
            "corpus_split": split,
            "doc_id": doc_id,
            "answer_len": len(answer),
        },
        rng=rng,
        hard=hard,
    )


def generate_nlp_cloze_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
    split: Split = "train",
    sentences: tuple[str, ...] | None = None,
) -> list[DistillSample]:
    """Generate ``n_samples`` cloze items from WikiText (or provided sentences)."""
    if n_samples <= 0:
        return []
    rng = random.Random(seed)
    bank = list(sentences) if sentences is not None else list(_load_wikitext_sentences(split))
    if not bank:
        raise RuntimeError("nlp_cloze: empty sentence bank")
    vocab = _vocab_from_sentences(bank)
    samples: list[DistillSample] = []
    spins = 0
    max_spins = max(n_samples * 40, 1000)
    while len(samples) < n_samples and spins < max_spins:
        spins += 1
        doc_id = rng.randrange(len(bank))
        sent = bank[doc_id]
        item = _cloze_from_sentence(
            rng, sent, vocab, hard=hard, split=split, doc_id=doc_id
        )
        if item is None:
            continue
        samples.append(item)
    if len(samples) < n_samples:
        raise RuntimeError(
            f"nlp_cloze: only produced {len(samples)}/{n_samples} "
            f"(split={split}, bank={len(bank)})"
        )
    return samples


# Tiny offline bank for unit tests (no Hub dependency).
_TEST_TRAIN_SENTS = (
    "The astronomer adjusted the telescope to get a clearer view of the distant nebula.",
    "Farmers harvested the wheat before the autumn storms arrived over the valley.",
    "Engineers reinforced the bridge after inspectors found cracks in the steel beams.",
    "The chemist measured the solvent carefully before mixing the reactive compounds together.",
    "Sailors checked the compass repeatedly while navigating through the dense fog bank.",
    "The librarian shelved the manuscript beside volumes of medieval history texts.",
    "Pilots delayed the departure until the runway was cleared of lingering ice patches.",
    "Biologists tracked the migration patterns of whales across the northern Pacific ocean.",
)

_TEST_EVAL_SENTS = (
    "Geologists mapped the canyon walls to understand ancient river erosion patterns.",
    "The violinist rehearsed the sonata until every phrase felt precise and balanced.",
    "Archaeologists uncovered pottery fragments beneath layers of volcanic ash deposits.",
    "Programmers refactored the module after profiling showed unexpected memory growth.",
    "The botanist catalogued rare orchids growing along the shaded mountain trail.",
    "Divers explored the coral reef before rising temperatures bleached nearby habitats.",
)


def generate_nlp_cloze_samples_offline(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
    split: Split = "train",
) -> list[DistillSample]:
    """Deterministic cloze from embedded banks (tests / no network)."""
    bank = _TEST_TRAIN_SENTS if split == "train" else _TEST_EVAL_SENTS
    return generate_nlp_cloze_samples(
        n_samples, seed=seed, hard=hard, split=split, sentences=bank
    )
