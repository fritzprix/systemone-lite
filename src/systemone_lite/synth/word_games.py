"""Closed-option word / language-rule games for System One distill.

Purpose: keep a verifiable linguistic-rule anchor in spatial/CA-heavy mixes
(~10–20% target at the builder). Not an MMLU knowledge dump.
"""

from __future__ import annotations

import random
from typing import Any

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample

# Small curated banks — ASCII only, no coaching keywords.
WORDS = [
    "apple", "brave", "crane", "delta", "eagle", "flame", "grape", "house",
    "ivory", "joker", "knife", "lemon", "mango", "night", "ocean", "piano",
    "queen", "river", "stone", "tiger", "umbra", "vivid", "whale", "xenon",
    "yacht", "zebra", "anchor", "bridge", "candle", "dragon", "engine",
    "forest", "garden", "hammer", "island", "jungle", "kernel", "ladder",
    "mirror", "needle", "orange", "planet", "quartz", "rocket", "silver",
    "tunnel", "violet", "window", "yellow", "zipper",
]

DEFINITIONS: dict[str, str] = {
    "apple": "A common round fruit",
    "bridge": "A structure spanning a gap",
    "candle": "A stick of wax with a wick",
    "dragon": "A mythical winged reptile",
    "engine": "A machine that converts energy to motion",
    "forest": "A large area covered chiefly with trees",
    "hammer": "A tool with a weighted head for striking",
    "island": "Land surrounded by water",
    "mirror": "A surface that reflects an image",
    "ocean": "A vast body of salt water",
    "planet": "A large body orbiting a star",
    "rocket": "A vehicle propelled by ejected exhaust",
    "silver": "A shiny grayish-white metal",
    "tunnel": "An underground passageway",
    "window": "An opening in a wall that admits light",
}

SIMPLE_PLURALS: dict[str, str] = {
    "cat": "cats",
    "dog": "dogs",
    "book": "books",
    "car": "cars",
    "tree": "trees",
    "box": "boxes",
    "fox": "foxes",
    "bus": "buses",
    "baby": "babies",
    "city": "cities",
    "leaf": "leaves",
    "knife": "knives",
    "child": "children",
    "mouse": "mice",
}


def _scramble(rng: random.Random, word: str) -> str:
    chars = list(word)
    for _ in range(20):
        rng.shuffle(chars)
        scrambled = "".join(chars)
        if scrambled != word:
            return scrambled
    return word[1:] + word[0] if len(word) > 1 else word


def _distractors(rng: random.Random, correct: str, pool: list[str], k: int = 3) -> list[str]:
    others = [w for w in pool if w != correct]
    rng.shuffle(others)
    out = others[:k]
    while len(out) < k:
        # synthetic near-miss
        base = correct
        i = rng.randrange(len(base))
        letters = list(base)
        letters[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        cand = "".join(letters)
        if cand != correct and cand not in out:
            out.append(cand)
    return out


def _anagram_sample(rng: random.Random, *, hard: bool) -> DistillSample:
    word = rng.choice(WORDS)
    scrambled = _scramble(rng, word)
    opts = {word: word}
    for d in _distractors(rng, word, WORDS, k=3):
        opts[d] = d
    state = {
        "scrambled": scrambled,
        "legend": "Unscramble into a valid English word from the options.",
    }
    return choice_sample(
        task="word.anagram",
        state=state,
        instructions=f"Unscramble letters '{scrambled}' into the correct word.",
        options=opts,
        label_key=word,
        meta={"gym": "word_games", "kind": "anagram"},
        rng=rng,
        hard=hard,
    )


def _definition_sample(rng: random.Random, *, hard: bool) -> DistillSample:
    word = rng.choice(list(DEFINITIONS))
    definition = DEFINITIONS[word]
    opts = {word: word}
    for d in _distractors(rng, word, list(DEFINITIONS), k=3):
        opts[d] = d
    state = {
        "definition": definition,
        "legend": "Choose the word that matches the definition.",
    }
    return choice_sample(
        task="word.definition",
        state=state,
        instructions="Which option matches the given definition?",
        options=opts,
        label_key=word,
        meta={"gym": "word_games", "kind": "definition"},
        rng=rng,
        hard=hard,
    )


def _one_edit_sample(rng: random.Random, *, hard: bool) -> DistillSample:
    word = rng.choice([w for w in WORDS if len(w) >= 4])
    i = rng.randrange(len(word))
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_ch = rng.choice([c for c in alphabet if c != word[i]])
    edited = word[:i] + new_ch + word[i + 1 :]
    # Options: the true one-edit word + unrelated words + maybe original
    opts = {edited: edited}
    for d in _distractors(rng, edited, WORDS, k=3):
        opts[d] = d
    state = {
        "base_word": word,
        "legend": "Exactly one letter differs from the base word.",
    }
    return choice_sample(
        task="word.one_edit",
        state=state,
        instructions=f"Which option differs from '{word}' by exactly one letter?",
        options=opts,
        label_key=edited,
        meta={"gym": "word_games", "kind": "one_edit", "base": word},
        rng=rng,
        hard=hard,
    )


def _plural_sample(rng: random.Random, *, hard: bool) -> DistillSample:
    singular = rng.choice(list(SIMPLE_PLURALS))
    plural = SIMPLE_PLURALS[singular]
    wrongs = [
        singular + "s" if not plural.endswith("s") else singular + "es",
        singular + "es",
        singular[:-1] + "ies" if singular.endswith("y") else singular + "ies",
        singular,
    ]
    opts = {plural: plural}
    for w in wrongs:
        if w != plural:
            opts[w] = w
        if len(opts) >= 4:
            break
    while len(opts) < 4:
        filler = rng.choice(list(SIMPLE_PLURALS.values()))
        if filler not in opts:
            opts[filler] = filler
    state = {
        "singular": singular,
        "rule": "Form the regular/irregular English plural as listed in standard usage.",
        "legend": "Apply English pluralization; pick the correct plural form.",
    }
    return choice_sample(
        task="word.plural",
        state=state,
        instructions=f"What is the plural of '{singular}'?",
        options=opts,
        label_key=plural,
        meta={"gym": "word_games", "kind": "plural"},
        rng=rng,
        hard=hard,
    )


def _vowel_count_sample(rng: random.Random, *, hard: bool) -> DistillSample:
    word = rng.choice(WORDS)
    vowels = sum(1 for c in word.lower() if c in "aeiou")
    # Closed numeric options around the truth
    candidates = sorted({max(0, vowels + d) for d in (-2, -1, 0, 1, 2)})
    rng.shuffle(candidates)
    opts = {str(v): f"{v} vowels" for v in candidates[:4]}
    if str(vowels) not in opts:
        opts[str(vowels)] = f"{vowels} vowels"
    state = {
        "word": word,
        "legend": "Count letters a,e,i,o,u only (not y).",
    }
    return choice_sample(
        task="word.vowel_count",
        state=state,
        instructions=f"How many vowel letters (a,e,i,o,u) are in '{word}'?",
        options=opts,
        label_key=str(vowels),
        meta={"gym": "word_games", "kind": "vowel_count"},
        rng=rng,
        hard=hard,
    )


_GENERATORS = (
    _anagram_sample,
    _definition_sample,
    _one_edit_sample,
    _plural_sample,
    _vowel_count_sample,
)


def generate_word_game_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
) -> list[DistillSample]:
    rng = random.Random(seed)
    samples: list[DistillSample] = []
    while len(samples) < n_samples:
        gen = rng.choice(_GENERATORS)
        samples.append(gen(rng, hard=hard))
    return samples
