"""DebateJudge — short claim adjudication with evidence-based labels."""

from __future__ import annotations

import random
from typing import Any

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample, paraphrase

TOPICS = [
    ("deploy_friday", "We should deploy on Friday"),
    ("remote_first", "Remote-first improves productivity"),
    ("raise_prices", "We should raise prices 10%"),
    ("rewrite_backend", "We should rewrite the backend in Rust"),
    ("hire_junior", "We should hire more junior engineers"),
]

CLAIMS = {
    "deploy_friday": {
        "pro": [
            ("Customers asked for the fix this week", True),
            ("Staging tests are green", True),
            ("Fridays feel lucky", False),
        ],
        "con": [
            ("On-call coverage is thin on weekends", True),
            ("Last Friday deploy caused a Sev-1", True),
            ("I just don't like Fridays", False),
        ],
    },
    "remote_first": {
        "pro": [
            ("Attrition dropped 20% after remote policy", True),
            ("Hiring pipeline widened internationally", True),
            ("My friend likes remote", False),
        ],
        "con": [
            ("New-grad ramp surveys show slower onboarding", True),
            ("Some customers require on-site workshops", True),
            ("Offices look empty and sad", False),
        ],
    },
    "raise_prices": {
        "pro": [
            ("Gross margin is below target", True),
            ("Competitors already priced 12% higher", True),
            ("Prices going up sounds cool", False),
        ],
        "con": [
            ("Churn spiked after last increase", True),
            ("Enterprise deals are price-sensitive this quarter", True),
            ("Customers will be mad forever", False),
        ],
    },
    "rewrite_backend": {
        "pro": [
            ("p99 latency is 4x SLO on hot path", True),
            ("Rust prototype cut CPU 35% in bench", True),
            ("Rust is trendy on HN", False),
        ],
        "con": [
            ("Team has 1 Rust-experienced engineer", True),
            ("Rewrite estimated at 2 quarters with feature freeze", True),
            ("I dream in garbage collectors", False),
        ],
    },
    "hire_junior": {
        "pro": [
            ("Senior mentors have spare capacity", True),
            ("Internship NPS was high and conversion strong", True),
            ("Juniors are cheaper vibes", False),
        ],
        "con": [
            ("Critical path needs senior ownership now", True),
            ("Ramp cost exceeds budget this half", True),
            ("Juniors ask too many questions", False),
        ],
    },
}

WIN_INSTR = [
    "Which side is better supported by evidence? Reply with one option letter.",
    "Given the claims and evidence flags, who should win the debate? One letter.",
]

ENOUGH_INSTR = [
    "Is there enough hard evidence to decide confidently? Reply with one letter.",
    "Do we have sufficient evidence-backed claims (not vibes)? Reply with one letter.",
]

CONF_INSTR = [
    "How confident should we be in the judgment? Reply with one level letter.",
    "Rate decision confidence from the evidence balance. One letter.",
]


def generate_debate_episode(rng: random.Random) -> list[DistillSample]:
    topic_id, question = rng.choice(TOPICS)
    pack = CLAIMS[topic_id]
    n_pro = rng.randint(1, 3)
    n_con = rng.randint(1, 3)
    pro = rng.sample(pack["pro"], n_pro)
    con = rng.sample(pack["con"], n_con)

    pro_ev = sum(1 for _, hard in pro if hard)
    con_ev = sum(1 for _, hard in con if hard)
    pro_soft = sum(1 for _, hard in pro if not hard)
    con_soft = sum(1 for _, hard in con if not hard)

    state: dict[str, Any] = {
        "question": question,
        "side_a": {
            "label": "support",
            "claims": [{"text": t, "hard_evidence": h} for t, h in pro],
        },
        "side_b": {
            "label": "oppose",
            "claims": [{"text": t, "hard_evidence": h} for t, h in con],
        },
        "rubric": "Prefer hard evidence over vibes; break ties toward the side with more hard evidence.",
    }

    if pro_ev > con_ev:
        winner = "support"
    elif con_ev > pro_ev:
        winner = "oppose"
    elif pro_soft != con_soft:
        winner = "support" if pro_soft > con_soft else "oppose"
    else:
        winner = rng.choice(["support", "oppose"])

    samples = [
        choice_sample(
            task="debate.winner",
            state=state,
            instructions=paraphrase(rng, WIN_INSTR),
            options={
                "support": "Side A — support the proposal",
                "oppose": "Side B — oppose the proposal",
            },
            label_key=winner,
            meta={"gym": "debate_judge", "pro_ev": pro_ev, "con_ev": con_ev},
            rng=rng,
        )
    ]

    enough = (pro_ev + con_ev) >= 2 and abs(pro_ev - con_ev) >= 1
    samples.append(
        choice_sample(
            task="debate.enough_evidence",
            state=state,
            instructions=paraphrase(rng, ENOUGH_INSTR),
            options={
                "yes": "Enough hard evidence to decide",
                "no": "Too thin or tied — needs more evidence",
            },
            label_key="yes" if enough else "no",
            meta={"gym": "debate_judge", "schema_hint": "noul"},
            rng=rng,
        )
    )

    margin = abs(pro_ev - con_ev)
    if not enough:
        conf = "0"
    elif margin >= 2:
        conf = "2"
    else:
        conf = "1"
    samples.append(
        choice_sample(
            task="debate.confidence",
            state=state,
            instructions=paraphrase(rng, CONF_INSTR),
            options={
                "0": "low confidence",
                "1": "medium confidence",
                "2": "high confidence",
            },
            label_key=conf,
            meta={"gym": "debate_judge", "schema_hint": "score"},
            rng=rng,
        )
    )
    return samples
