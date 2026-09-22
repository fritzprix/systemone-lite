"""Closed-option word / language-rule games for System One distill.

Train and eval use **disjoint vocabularies** so high accuracy cannot come from
seeing the same lemmas in both splits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample

Split = Literal["train", "eval"]

# --- Train banks (expanded) -------------------------------------------------
TRAIN_WORDS = [
    "apple", "brave", "crane", "delta", "eagle", "flame", "grape", "house",
    "ivory", "joker", "knife", "lemon", "mango", "night", "ocean", "piano",
    "queen", "river", "stone", "tiger", "umbra", "vivid", "whale", "xenon",
    "yacht", "zebra", "anchor", "bridge", "candle", "dragon", "engine",
    "forest", "garden", "hammer", "island", "jungle", "kernel", "ladder",
    "mirror", "needle", "orange", "planet", "quartz", "rocket", "silver",
    "tunnel", "violet", "window", "yellow", "zipper", "barrel", "copper",
    "desert", "fabric", "glider", "helmet", "insect", "jasper", "kettle",
    "lizard", "magnet", "nectar", "oxygen", "pepper", "quiver", "ribbon",
    "saddle", "tomato", "ulcer", "vector", "walnut", "yearn",
]

TRAIN_DEFINITIONS: dict[str, str] = {
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
    "magnet": "An object that attracts iron",
    "nectar": "A sugary fluid from flowers",
    "pepper": "A pungent spice from dried berries",
    "ribbon": "A narrow strip of fabric",
    "tomato": "A red edible berry used as a vegetable",
}

TRAIN_PLURALS: dict[str, str] = {
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
    "goose": "geese",
    "tooth": "teeth",
    "foot": "feet",
    "man": "men",
    "woman": "women",
    "sheep": "sheep",
}

# --- Eval banks (disjoint lemmas; never appear in TRAIN_*) -----------------
EVAL_WORDS = [
    "amber", "bison", "coral", "dawn", "ember", "flint", "grain", "harbor",
    "iris", "jade", "kite", "lotus", "moss", "nickel", "opal", "pearl",
    "quill", "reef", "sage", "tide", "urn", "vine", "willow", "xylem",
    "yolk", "zinc", "anvil", "basalt", "citrus", "dune", "echo", "fjord",
    "geyser", "hazel", "igloo", "beacon", "canyon", "drift", "grove",
    "haven", "inlet", "jewel", "knoll", "lagoon", "meadow", "notch",
    "oasis", "prairie", "ravine", "summit", "thicket", "upland", "valley",
]
assert len(EVAL_WORDS) == len(set(EVAL_WORDS))
assert not (set(TRAIN_WORDS) & set(EVAL_WORDS)), set(TRAIN_WORDS) & set(EVAL_WORDS)

EVAL_DEFINITIONS: dict[str, str] = {
    "amber": "Fossilized tree resin, often golden",
    "bison": "A large shaggy wild ox of North America",
    "canyon": "A deep gorge with steep sides",
    "fjord": "A long narrow sea inlet between cliffs",
    "geyser": "A hot spring that intermittently erupts",
    "harbor": "A sheltered place for ships",
    "lagoon": "A shallow body of water separated from the sea",
    "meadow": "A field of grassland often with wildflowers",
    "oasis": "A fertile spot in a desert with water",
    "prairie": "A wide area of flat grassland",
    "ravine": "A deep narrow valley with steep sides",
    "summit": "The highest point of a hill or mountain",
    "thicket": "A dense growth of shrubs or small trees",
    "valley": "Low land between hills or mountains",
    "willow": "A tree with slender flexible branches",
}
assert not (set(TRAIN_DEFINITIONS) & set(EVAL_DEFINITIONS))

EVAL_PLURALS: dict[str, str] = {
    "ox": "oxen",
    "die": "dice",
    "cactus": "cacti",
    "fungus": "fungi",
    "axis": "axes",
    "crisis": "crises",
    "thesis": "theses",
    "analysis": "analyses",
    "phenomenon": "phenomena",
    "criterion": "criteria",
    "radius": "radii",
    "index": "indices",
    "appendix": "appendices",
    "matrix": "matrices",
    "vertex": "vertices",
    "elf": "elves",
    "wolf": "wolves",
    "loaf": "loaves",
    "thief": "thieves",
    "wife": "wives",
}
assert not (set(TRAIN_PLURALS) & set(EVAL_PLURALS))
assert not (set(TRAIN_PLURALS.values()) & set(EVAL_PLURALS.values()))


@dataclass(frozen=True)
class WordBanks:
    words: list[str]
    definitions: dict[str, str]
    plurals: dict[str, str]


def banks_for(split: Split) -> WordBanks:
    if split == "train":
        return WordBanks(TRAIN_WORDS, TRAIN_DEFINITIONS, TRAIN_PLURALS)
    return WordBanks(EVAL_WORDS, EVAL_DEFINITIONS, EVAL_PLURALS)


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
        base = correct
        i = rng.randrange(len(base))
        letters = list(base)
        letters[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        cand = "".join(letters)
        if cand != correct and cand not in out:
            out.append(cand)
    return out


def _anagram_sample(rng: random.Random, banks: WordBanks, *, hard: bool) -> DistillSample:
    word = rng.choice(banks.words)
    scrambled = _scramble(rng, word)
    opts = {word: word}
    for d in _distractors(rng, word, banks.words, k=3):
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


def _definition_sample(rng: random.Random, banks: WordBanks, *, hard: bool) -> DistillSample:
    word = rng.choice(list(banks.definitions))
    definition = banks.definitions[word]
    opts = {word: word}
    for d in _distractors(rng, word, list(banks.definitions), k=3):
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


def _one_edit_sample(rng: random.Random, banks: WordBanks, *, hard: bool) -> DistillSample:
    word = rng.choice([w for w in banks.words if len(w) >= 4])
    i = rng.randrange(len(word))
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    new_ch = rng.choice([c for c in alphabet if c != word[i]])
    edited = word[:i] + new_ch + word[i + 1 :]
    opts = {edited: edited}
    for d in _distractors(rng, edited, banks.words, k=3):
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


def _plural_sample(rng: random.Random, banks: WordBanks, *, hard: bool) -> DistillSample:
    singular = rng.choice(list(banks.plurals))
    plural = banks.plurals[singular]
    wrongs = [
        singular + "s",
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
        filler = rng.choice(list(banks.plurals.values()))
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


def _vowel_count_sample(rng: random.Random, banks: WordBanks, *, hard: bool) -> DistillSample:
    word = rng.choice(banks.words)
    vowels = sum(1 for c in word.lower() if c in "aeiou")
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


def generate_word_game_samples(
    n_samples: int,
    *,
    seed: int = 42,
    hard: bool = False,
    split: Split = "train",
) -> list[DistillSample]:
    rng = random.Random(seed)
    banks = banks_for(split)
    generators = (
        _anagram_sample,
        _definition_sample,
        _one_edit_sample,
        _plural_sample,
        _vowel_count_sample,
    )
    samples: list[DistillSample] = []
    while len(samples) < n_samples:
        gen = rng.choice(generators)
        s = gen(rng, banks, hard=hard)
        s.meta = {**(s.meta or {}), "vocab_split": split}
        samples.append(s)
    return samples
