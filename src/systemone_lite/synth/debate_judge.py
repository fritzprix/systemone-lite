"""DebateJudge — short claim adjudication with evidence-based labels.

Train and eval use **disjoint topics** so score cannot come from memorizing
the same six proposals in both splits.
"""

from __future__ import annotations

import random
from typing import Any, Literal

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample, paraphrase

Split = Literal["train", "eval"]

TRAIN_TOPICS = [
    ("deploy_friday", "We should deploy on Friday"),
    ("remote_first", "Remote-first improves productivity"),
    ("raise_prices", "We should raise prices 10%"),
    ("rewrite_backend", "We should rewrite the backend in Rust"),
    ("hire_junior", "We should hire more junior engineers"),
]

EVAL_TOPICS = [
    ("open_source_core", "We should open-source the core library"),
    ("four_day_week", "We should move to a four-day work week"),
    ("kill_feature_x", "We should sunset Feature X this quarter"),
    ("buy_competitor", "We should acquire Competitor Y"),
]

# Backward-compatible alias (union) — prefer TRAIN_/EVAL_ explicitly.
TOPICS = TRAIN_TOPICS + EVAL_TOPICS

CLAIMS: dict[str, dict[str, list[tuple[str, bool]]]] = {
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
    "open_source_core": {
        "pro": [
            ("Inbound PRs already fix bugs we lack time for", True),
            ("Design partners ask for source escrow", True),
            ("Open source is always good actually", False),
        ],
        "con": [
            ("Core contains proprietary scoring IP", True),
            ("Support load estimates exceed current headcount", True),
            ("Competitors will steal our vibes", False),
        ],
    },
    "four_day_week": {
        "pro": [
            ("Pilot showed same output with 10% less overtime", True),
            ("Candidate offer-accept rate rose 18%", True),
            ("Long weekends are trendy", False),
        ],
        "con": [
            ("Customer SLAs require weekday coverage five days", True),
            ("Payroll cost model assumes 5-day capacity", True),
            ("Fridays are when inspiration strikes", False),
        ],
    },
    "kill_feature_x": {
        "pro": [
            ("Feature X costs 30% of eng hours for 2% revenue", True),
            ("NPS for X users is below company average", True),
            ("I never liked Feature X", False),
        ],
        "con": [
            ("Top-3 enterprise contracts list X as must-have", True),
            ("Migration path for X users is unfinished", True),
            ("Sunsets always feel mean", False),
        ],
    },
    "buy_competitor": {
        "pro": [
            ("Diligence shows 40% customer overlap with low churn", True),
            ("Their sales channel fills a geo we lack", True),
            ("Acquisitions are exciting", False),
        ],
        "con": [
            ("Integration would freeze our roadmap for two quarters", True),
            ("Ask exceeds board-approved M&A budget", True),
            ("Their logo looks cooler than ours", False),
        ],
    },
}

_TRAIN_IDS = {t[0] for t in TRAIN_TOPICS}
_EVAL_IDS = {t[0] for t in EVAL_TOPICS}
assert not (_TRAIN_IDS & _EVAL_IDS)

WIN_INSTR = [
    "Which side is better supported by evidence? Reply with one option letter.",
    "Given the claims and evidence flags, who should win the debate? One letter.",
    "Apply the rubric: hard evidence beats vibes. Pick the winning side letter.",
    "Who wins under the stated evidence policy? One Criteria letter.",
]

ENOUGH_INSTR = [
    "Is there enough hard evidence to decide confidently? Reply with one letter.",
    "Do we have sufficient evidence-backed claims (not vibes)? Reply with one letter.",
    "Evidence sufficiency check — yes or no via option letter.",
]

CONF_INSTR = [
    "How confident should we be in the judgment? Reply with one level letter.",
    "Rate decision confidence from the evidence balance. One letter.",
    "Confidence score for this adjudication — pick one Criteria letter.",
]


def topics_for(split: Split) -> list[tuple[str, str]]:
    return TRAIN_TOPICS if split == "train" else EVAL_TOPICS


def generate_debate_episode(
    rng: random.Random,
    *,
    hard: bool = False,
    split: Split = "train",
) -> list[DistillSample]:
    topic_id, question = rng.choice(topics_for(split))
    pack = CLAIMS[topic_id]
    n_pro = rng.randint(1, 3)
    n_con = rng.randint(1, 3)
    pro = rng.sample(pack["pro"], n_pro)
    con = rng.sample(pack["con"], n_con)

    pro_ev = sum(1 for _, hard_ev in pro if hard_ev)
    con_ev = sum(1 for _, hard_ev in con if hard_ev)
    pro_soft = sum(1 for _, hard_ev in pro if not hard_ev)
    con_soft = sum(1 for _, hard_ev in con if not hard_ev)

    if hard and rng.random() < 0.45:
        state: dict[str, Any] = {
            "motion": question,
            "affirmative": {
                "name": "support",
                "points": [{"claim": t, "evidence": h} for t, h in pro],
            },
            "negative": {
                "name": "oppose",
                "points": [{"claim": t, "evidence": h} for t, h in con],
            },
            "scoring": "Count hard evidence first; soft claims only break ties.",
        }
    else:
        state = {
            "question": question,
            "side_a": {
                "label": "support",
                "claims": [{"text": t, "hard_evidence": h} for t, h in pro],
            },
            "side_b": {
                "label": "oppose",
                "claims": [{"text": t, "hard_evidence": h} for t, h in con],
            },
            "rubric": (
                "Prefer hard evidence over vibes; break ties toward the side "
                "with more hard evidence."
            ),
        }

    if pro_ev > con_ev:
        winner = "support"
    elif con_ev > pro_ev:
        winner = "oppose"
    elif pro_soft != con_soft:
        winner = "support" if pro_soft > con_soft else "oppose"
    else:
        winner = rng.choice(["support", "oppose"])

    if hard and rng.random() < 0.5:
        win_opts = {
            "support": "Affirmative / support",
            "oppose": "Negative / oppose",
        }
    else:
        win_opts = {
            "support": "Side A — support the proposal",
            "oppose": "Side B — oppose the proposal",
        }

    topic_meta = {
        "gym": "debate_judge",
        "topic_id": topic_id,
        "topic_split": split,
        "pro_ev": pro_ev,
        "con_ev": con_ev,
    }
    samples = [
        choice_sample(
            task="debate.winner",
            state=state,
            instructions=paraphrase(rng, WIN_INSTR),
            options=win_opts,
            label_key=winner,
            meta=topic_meta,
            rng=rng,
            hard=hard,
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
            meta={**topic_meta, "schema_hint": "noul"},
            rng=rng,
            hard=hard,
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
            meta={**topic_meta, "schema_hint": "score"},
            rng=rng,
            hard=hard,
        )
    )
    return samples
