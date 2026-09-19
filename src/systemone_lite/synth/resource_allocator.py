"""ResourceAllocator — budget / request funding decisions with rule labels."""

from __future__ import annotations

import random
from typing import Any

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.common import choice_sample, paraphrase

REQUEST_POOL = [
    ("security_patch", "Critical security patch", 40, 3),
    ("marketing_campaign", "Q3 marketing campaign", 120, 1),
    ("laptop_refresh", "Engineer laptop refresh", 80, 1),
    ("customer_refund_pool", "Customer refund reserve", 60, 2),
    ("ml_experiment", "GPU hours for ML experiment", 90, 1),
    ("office_snacks", "Office snacks", 15, 0),
    ("compliance_audit", "Compliance audit support", 70, 3),
    ("hiring_bonus", "Hiring referral bonuses", 50, 1),
    ("cdn_upgrade", "CDN capacity upgrade", 55, 2),
    ("docs_rewrite", "Docs rewrite contractor", 35, 0),
    ("oncall_stipend", "On-call stipend top-up", 45, 2),
    ("design_sprint", "Design sprint facilitator", 25, 0),
]

FUND_INSTR = [
    "Which single request should be funded next given the budget? Reply with one letter.",
    "Pick the best request to fund now. Prefer high urgency that fits the remaining budget.",
    "Select the next funding target under the policy. One option letter.",
    "Given budget_remaining and urgencies, which request wins? One letter.",
]

CAN_ALL_INSTR = [
    "Can we fund ALL remaining requests within the budget? Reply with one letter.",
    "Is the budget enough to approve every request? Reply with one option letter.",
    "Would summing all costs fit under budget_remaining? Answer with one letter.",
]

PRESSURE_INSTR = [
    "Rate overall budget pressure. Reply with one level letter.",
    "How constrained is the budget relative to demand? Pick one letter.",
    "Budget tightness score — choose one Criteria letter.",
]


def generate_allocator_episode(
    rng: random.Random, *, hard: bool = False
) -> list[DistillSample]:
    n = rng.randint(3, 6 if hard else 5)
    picked = rng.sample(REQUEST_POOL, n)
    requests = []
    for key, title, cost, urg in picked:
        c = max(5, cost + rng.randint(-15 if hard else -10, 15 if hard else 10))
        u = urg
        if hard and rng.random() < 0.2:
            u = max(0, min(3, urg + rng.choice([-1, 1])))
        requests.append({"id": key, "title": title, "cost": c, "urgency": u})

    total_need = sum(r["cost"] for r in requests)
    if rng.random() < 0.5:
        budget = rng.randint(max(r["cost"] for r in requests), total_need + 20)
    else:
        budget = rng.randint(total_need, total_need + 80)

    policy = (
        "Fund at most one request per step; prefer higher urgency among affordable."
        if not hard
        else "Single grant per step. Max urgency among cost<=budget; ties → cheaper, then id."
    )
    if hard and rng.random() < 0.4:
        state: dict[str, Any] = {
            "wallet": {"remaining": budget, "ccy": "USD"},
            "queue": requests,
            "rules": policy,
        }
    else:
        state = {
            "budget_remaining": budget,
            "currency": "USD",
            "requests": requests,
            "policy": policy,
        }

    affordable = [r for r in requests if r["cost"] <= budget]
    if not affordable:
        cheapest = min(requests, key=lambda r: r["cost"])
        budget = cheapest["cost"]
        wallet = state.get("wallet")
        if isinstance(wallet, dict):
            wallet["remaining"] = budget
        else:
            state["budget_remaining"] = budget
        affordable = [cheapest]

    def rank(r: dict[str, Any]) -> tuple[int, int, str]:
        return (r["urgency"], -r["cost"], r["id"])

    best = max(affordable, key=rank)
    if hard and rng.random() < 0.5:
        options = {
            r["id"]: f"{r['title']} | ${r['cost']} | urg={r['urgency']}" for r in requests
        }
    else:
        options = {
            r["id"]: f"{r['title']} (cost={r['cost']}, urgency={r['urgency']})"
            for r in requests
        }

    samples = [
        choice_sample(
            task="alloc.fund_next",
            state=state,
            instructions=paraphrase(rng, FUND_INSTR),
            options=options,
            label_key=best["id"],
            meta={"gym": "resource_allocator"},
            rng=rng,
            hard=hard,
        )
    ]

    can_all = total_need <= budget
    samples.append(
        choice_sample(
            task="alloc.can_fund_all",
            state=state,
            instructions=paraphrase(rng, CAN_ALL_INSTR),
            options={
                "yes": "Budget covers the sum of all request costs",
                "no": "Budget is insufficient for all requests",
            },
            label_key="yes" if can_all else "no",
            meta={"gym": "resource_allocator", "schema_hint": "noul"},
            rng=rng,
            hard=hard,
        )
    )

    ratio = total_need / max(budget, 1)
    if ratio <= 0.8:
        pressure = "0"
    elif ratio <= 1.2:
        pressure = "1"
    else:
        pressure = "2"
    samples.append(
        choice_sample(
            task="alloc.pressure",
            state=state,
            instructions=paraphrase(rng, PRESSURE_INSTR),
            options={
                "0": "low — budget comfortably covers demand",
                "1": "medium — close to capacity",
                "2": "high — demand exceeds budget",
            },
            label_key=pressure,
            meta={"gym": "resource_allocator", "schema_hint": "score", "ratio": ratio},
            rng=rng,
            hard=hard,
        )
    )
    return samples
