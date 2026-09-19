"""TicketDungeon — support-ticket routing gym with rule-based labels."""

from __future__ import annotations

import random
from typing import Any

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample, paraphrase

TEAMS = {
    "billing": "payment, charges, invoices, refunds",
    "technical": "bugs, crashes, outages, integrations",
    "shipping": "delivery, packages, tracking",
    "sales": "pricing, upgrades, new accounts",
    "other": "unclear or general questions",
}

TEAMS_ALT = {
    "billing": "Finance / payments desk",
    "technical": "Engineering support",
    "shipping": "Logistics & fulfillment",
    "sales": "Commercial / AE team",
    "other": "General inbox / catch-all",
}

HUMAN_KW = ("lawyer", "legal", "fraud", "police", "sue", "threat", "urgent asap")
ANGER_KW = ("furious", "angry", "ridiculous", "worst", "hate")

SUBJECTS = {
    "billing": [
        "Duplicate card charge on order {oid}",
        "Refund not received for cancelled sub",
        "Invoice shows unexpected billing amount",
        "Charged twice for the same invoice {oid}",
    ],
    "technical": [
        "App crashes when opening settings",
        "Cannot log in after password reset",
        "API returns 500 on /v1/invoices",
        "Production outage: checkout broken",
    ],
    "shipping": [
        "Package stuck in transit for {days} days",
        "Where is my shipment {oid}?",
        "Delivery marked complete but missing",
        "Courier says package damaged",
    ],
    "sales": [
        "Need enterprise pricing for 200 seats",
        "Interested in upgrading to Pro",
        "Request a demo and pricing quote",
        "Compare Business vs Enterprise plans",
    ],
    "other": [
        "General question about account",
        "How do I change my display name?",
        "Where is the documentation?",
        "Thanks — just saying hello",
    ],
}

BODIES = [
    "Customer says: {blurb}. Wait time so far: {days} days. Tier: {tier}.",
    "Message: {blurb}. Priority flag: {priority}. Account tier: {tier}.",
    "{blurb} They have waited {days} days. Segment={tier}.",
    "From customer ({tier}): {blurb}. SLA wait={days}d priority={priority}.",
    "Thread summary — {blurb} | wait_days={days} | tier={tier} | priority={priority}",
]

BLURBS = {
    "billing": [
        "I was charged twice and want a refund immediately",
        "Invoice shows wrong amount on my card",
        "Please reverse the duplicate payment",
        "Billing portal shows an incorrect charge",
    ],
    "technical": [
        "The app crashes every time I open settings",
        "Login fails with an internal error",
        "Production API outage on invoices endpoint",
        "Stack trace on checkout — looks like a bug",
    ],
    "shipping": [
        "Tracking has not moved for a week",
        "Package marked delivered but nothing arrived",
        "Courier lost my shipment",
        "Delivery ETA keeps slipping",
    ],
    "sales": [
        "We need a quote for upgrading seats",
        "Interested in enterprise pricing and a demo",
        "What plan is best for a 200 person team?",
        "Can sales walk us through annual pricing?",
    ],
    "other": [
        "How do I change my display name?",
        "Thanks for the help last week",
        "Where can I find documentation?",
        "Is there a status page?",
    ],
}

ROUTE_INSTR = [
    "Route this support ticket to the best team. Reply with one option letter.",
    "Which team should own this ticket? Reply with exactly one option letter.",
    "Select the correct routing destination for the ticket state.",
    "Choose the owning queue from Criteria. One letter only.",
    "Best team for this case? Answer with a single option letter.",
]

NOUL_INSTR = [
    "Does this ticket need a human agent (not a bot)? Reply with one option letter.",
    "Should a human review this before automated handling? Reply with one letter.",
    "Human escalation required? Pick yes or no via the option letter.",
    "Can a bot handle this safely, or must a person intervene? One letter.",
]

SCORE_INSTR = [
    "Rate urgency for this ticket on the given scale. Reply with one option letter.",
    "How urgent is this ticket? Pick one level letter from Criteria.",
    "Assign an urgency score using the listed levels. One letter.",
    "Urgency classification — choose exactly one Criteria letter.",
]


def _needs_human(text: str, days: int, priority: str) -> bool:
    t = text.lower()
    if any(k in t for k in HUMAN_KW) or any(k in t for k in ANGER_KW):
        return True
    if days >= 5 or priority == "p0":
        return True
    if "refund" in t and days >= 2:
        return True
    return False


def _urgency(text: str, days: int, priority: str) -> int:
    t = text.lower()
    score = 0
    if priority == "p0":
        score += 2
    elif priority == "p1":
        score += 1
    score += min(days, 6) // 3
    if any(k in t for k in ANGER_KW) or any(k in t for k in HUMAN_KW):
        score += 1
    if "outage" in t or "crash" in t:
        score += 1
    return int(min(2, score))


def generate_ticket_episode(
    rng: random.Random, *, hard: bool = False
) -> list[DistillSample]:
    domain = rng.choice(list(BLURBS))
    blurb = rng.choice(BLURBS[domain])
    days = rng.randint(0, 10)
    tier = rng.choice(["free", "pro", "enterprise"])
    priority = rng.choice(["p0", "p1", "p2", "p3"])
    oid = f"A-{rng.randint(100, 999)}"
    subject = rng.choice(SUBJECTS[domain]).format(oid=oid, days=days)
    body = rng.choice(BODIES).format(
        blurb=blurb, days=days, tier=tier, priority=priority
    )
    text = f"{subject}. {body}"
    if rng.random() < (0.25 if hard else 0.15):
        text = body

    if hard and rng.random() < 0.5:
        state: dict[str, Any] = {
            "subject": subject,
            "body": body,
            "meta": {"tier": tier, "priority": priority, "wait_days": days},
            "channel": rng.choice(["email", "chat", "phone", "twitter"]),
        }
    else:
        state = {
            "ticket": {
                "subject": subject,
                "body": body,
                "tier": tier,
                "priority": priority,
                "wait_days": days,
            },
            "channel": rng.choice(["email", "chat", "phone"]),
        }

    teams = TEAMS_ALT if (hard and rng.random() < 0.5) else TEAMS
    route = domain
    samples = [
        choice_sample(
            task="ticket.route",
            state=state,
            instructions=paraphrase(rng, ROUTE_INSTR),
            options=dict(teams),
            label_key=route,
            meta={"gym": "ticket_dungeon", "domain_seed": domain},
            rng=rng,
            hard=hard,
        )
    ]

    human = _needs_human(text, days, priority)
    samples.append(
        choice_sample(
            task="ticket.needs_human",
            state=state,
            instructions=paraphrase(rng, NOUL_INSTR),
            options={
                "yes": "Needs a human (money, legal, anger, long wait, outage)",
                "no": "Safe for bot / routine",
            },
            label_key="yes" if human else "no",
            meta={"gym": "ticket_dungeon", "schema_hint": "noul"},
            rng=rng,
            hard=hard,
        )
    )

    urg = _urgency(text, days, priority)
    level_keys = ["0", "1", "2"]
    if hard and rng.random() < 0.5:
        level_desc = {
            "0": "P3 — can wait",
            "1": "P2 — same business day",
            "2": "P1/P0 — drop everything",
        }
    else:
        level_desc = {
            "0": "low — can wait",
            "1": "medium — same day",
            "2": "high — immediate",
        }
    samples.append(
        choice_sample(
            task="ticket.urgency",
            state=state,
            instructions=paraphrase(rng, SCORE_INSTR),
            options=level_desc,
            label_key=level_keys[urg],
            meta={"gym": "ticket_dungeon", "schema_hint": "score", "urgency": urg},
            rng=rng,
            hard=hard,
        )
    )
    return samples
