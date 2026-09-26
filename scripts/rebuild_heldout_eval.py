#!/usr/bin/env python3
"""Rebuild Phase-2 train/eval slices for zero train∩eval contamination.

Fixes:
  - word_games: held-out EVAL vocabulary (train keeps TRAIN vocab)
  - debate_judge: held-out EVAL topics (train keeps TRAIN topics)
  - connect4 / chess / other spatial: eval states rejected vs train fingerprints
  - cellular_automata: regen eval with rejection (already low overlap)

Does **not** regenerate the full 200k spatial train pool — only replaces the
contaminated gyms and rewrites eval.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from systemone_lite.chess_data import DistillSample
from systemone_lite.synth.cellular_automata import generate_ca_samples
from systemone_lite.synth.connect4 import generate_connect4_samples
from systemone_lite.synth.debate_judge import generate_debate_episode
from systemone_lite.synth.game2048 import generate_2048_samples
from systemone_lite.synth.gridworld import generate_gridworld_samples
from systemone_lite.synth.leakage import (
    collect_fingerprints,
    fingerprint,
    generate_disjoint,
)
from systemone_lite.synth.resource_allocator import generate_allocator_episode
from systemone_lite.synth.sokoban import generate_sokoban_samples
from systemone_lite.synth.ticket_dungeon import generate_ticket_episode
from systemone_lite.synth.word_games import generate_word_game_samples

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_gym(row: dict, default: str) -> dict:
    meta = dict(row.get("meta") or {})
    meta["gym"] = meta.get("gym") or default
    out = dict(row)
    out["meta"] = meta
    return out


def by_gym(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g = str((r.get("meta") or {}).get("gym") or "?")
        out[g].append(r)
    return out


def regen_debate(n: int, *, split: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples: list[DistillSample] = []
    while len(samples) < n:
        ep_rng = random.Random(rng.randint(0, 2**31 - 1))
        samples.extend(
            generate_debate_episode(ep_rng, hard=False, split=split)  # type: ignore[arg-type]
        )
    return [ensure_gym(s.to_json(), "debate_judge") for s in samples[:n]]


def overlap_report(
    train: list[dict],
    eval_rows: list[dict],
    *,
    mode: str = "state_task",
) -> dict[str, float]:
    from systemone_lite.synth.leakage import FpMode

    mode_t: FpMode = mode  # type: ignore[assignment]
    tr = by_gym(train)
    ev = by_gym(eval_rows)
    rates: dict[str, float] = {}
    for gym in sorted(set(tr) | set(ev)):
        if gym not in ev or not ev[gym]:
            continue
        tset = collect_fingerprints(tr.get(gym, []), mode=mode_t)
        hits = sum(1 for r in ev[gym] if fingerprint(r, mode=mode_t) in tset)
        rates[gym] = hits / len(ev[gym])
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "phase2_train_200k.jsonl",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=ROOT / "data" / "phase2_eval_4k.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=0.0,
        help="Fail if any gym state_task overlap exceeds this (default 0.0)",
    )
    parser.add_argument(
        "--also-diversity",
        action="store_true",
        default=True,
        help="Also rewrite synth_diversity_{train,eval}.jsonl",
    )
    parser.add_argument("--no-diversity", action="store_true")
    args = parser.parse_args()
    if args.no_diversity:
        args.also_diversity = False

    print(f"Loading train {args.train} …", flush=True)
    train = load_jsonl(args.train)
    print(f"Loading eval  {args.eval} …", flush=True)
    old_eval = load_jsonl(args.eval)
    tr_gym = by_gym(train)
    ev_gym = by_gym(old_eval)
    print("train gyms:", {k: len(v) for k, v in sorted(tr_gym.items())})
    print("eval  gyms:", {k: len(v) for k, v in sorted(ev_gym.items())})

    before = overlap_report(train, old_eval, mode="state_task")
    print("BEFORE state_task overlap:", {k: f"{v:.1%}" for k, v in before.items()})

    rng = random.Random(args.seed)

    # --- Replace contaminated TRAIN gyms ------------------------------------
    n_debate_train = len(tr_gym.get("debate_judge", []))
    n_word_train = len(tr_gym.get("word_games", []))

    if n_debate_train:
        print(f"Regenerating debate_judge train n={n_debate_train} (TRAIN topics)…", flush=True)
        new_debate_train = regen_debate(n_debate_train, split="train", seed=args.seed + 11)
        train = [r for r in train if (r.get("meta") or {}).get("gym") != "debate_judge"]
        train.extend(new_debate_train)

    if n_word_train:
        print(f"Regenerating word_games train n={n_word_train} (TRAIN vocab)…", flush=True)
        new_word_train = [
            ensure_gym(s.to_json(), "word_games")
            for s in generate_word_game_samples(
                n_word_train, seed=args.seed + 22, split="train"
            )
        ]
        train = [r for r in train if (r.get("meta") or {}).get("gym") != "word_games"]
        train.extend(new_word_train)

    rng.shuffle(train)
    tr_gym = by_gym(train)

    # Train exclude sets (state_only) per gym for eval rejection
    exclude: dict[str, set[str]] = {
        g: collect_fingerprints(rows, mode="state_only") for g, rows in tr_gym.items()
    }

    # --- Rebuild EVAL gym-by-gym --------------------------------------------
    new_eval: list[dict] = []

    def add_eval(gym: str, rows: list[dict]) -> None:
        for r in rows:
            new_eval.append(ensure_gym(r, gym))

    # Spatial generators with rejection
    spatial_gens: list[tuple[str, Callable[..., list[DistillSample]]]] = [
        ("sokoban", generate_sokoban_samples),
        ("game2048", generate_2048_samples),
        ("gridworld", generate_gridworld_samples),
        ("connect4", generate_connect4_samples),
    ]
    for i, (gym, gen) in enumerate(spatial_gens):
        n = len(ev_gym.get(gym, []))
        if not n:
            continue
        print(f"Eval {gym}: {n} with rejection…", flush=True)

        def factory(count: int, seed: int, _gen: Callable[..., list[DistillSample]] = gen) -> list[DistillSample]:
            return _gen(count, seed=seed)

        rows = generate_disjoint(
            factory,
            n,
            exclude=exclude.setdefault(gym, set()),
            mode="state_only",
            seed=args.seed + 1000 + i * 17,
            batch=max(64, n),
        )
        add_eval(gym, rows)

    # Chess — reject against train chess states; synthesize more if pool short
    n_chess = len(ev_gym.get("chess", []))
    if n_chess:
        print(f"Eval chess: {n_chess} with rejection…", flush=True)
        from systemone_lite.chess_data import (
            build_samples_for_position,
            generate_positions,
            heuristic_best_move,
        )

        chess_excl = exclude.setdefault("chess", set())
        kept: list[dict] = []

        def _try_add(row: dict) -> bool:
            if row.get("task") == "move":
                return False  # staged_v1: piece+destination only
            row = ensure_gym(row, "chess")
            fp = fingerprint(row, mode="state_only")
            if fp in chess_excl:
                return False
            kept.append(row)
            chess_excl.add(fp)
            return True

        staged_pool = ROOT / "data" / "chess_eval_staged.jsonl"
        legacy_pool = ROOT / "data" / "chess_eval_5k_2d.jsonl"
        pool_path = staged_pool if staged_pool.exists() else legacy_pool
        for row in load_jsonl(pool_path):
            _try_add(row)
            if len(kept) >= n_chess:
                break

        synth_seed = args.seed + 777
        attempts = 0
        while len(kept) < n_chess and attempts < 40:
            attempts += 1
            need = n_chess - len(kept)
            boards = generate_positions(
                n_positions=max(need * 2, 64),
                max_plies=48,
                seed=synth_seed + attempts,
                engine_path=None,
            )
            for board in boards:
                if len(kept) >= n_chess:
                    break
                if board.is_game_over() or not list(board.legal_moves):
                    continue
                best = heuristic_best_move(board)
                for s in build_samples_for_position(
                    board,
                    best_move=best,
                    source="heuristic",
                    include_stages=True,
                    include_move=False,
                ):
                    if len(kept) >= n_chess:
                        break
                    _try_add(s.to_json())

        if len(kept) < n_chess:
            raise SystemExit(f"chess eval: only {len(kept)}/{n_chess} disjoint FENs")
        add_eval("chess", kept[:n_chess])
        print(f"    ✓ chess: {n_chess} disjoint staged (pool+synth)", flush=True)

    # Ticket / allocator — regenerate with state rejection
    for gym, gen_ep in [
        ("ticket_dungeon", generate_ticket_episode),
        ("resource_allocator", generate_allocator_episode),
    ]:
        n = len(ev_gym.get(gym, []))
        if not n:
            continue
        print(f"Eval {gym}: {n} with state rejection…", flush=True)
        excl = exclude.setdefault(gym, set())
        kept = []
        ep_rng = random.Random(args.seed + hash(gym) % 10_000)
        attempts = 0
        while len(kept) < n and attempts < n * 50:
            attempts += 1
            for s in gen_ep(random.Random(ep_rng.randint(0, 2**31 - 1)), hard=False):
                row = ensure_gym(s.to_json(), gym)
                fp = fingerprint(row, mode="state_only")
                if fp in excl:
                    continue
                kept.append(row)
                excl.add(fp)
                if len(kept) >= n:
                    break
        if len(kept) < n:
            raise SystemExit(f"{gym} eval: only {len(kept)}/{n}")
        add_eval(gym, kept[:n])

    # Debate — EVAL topics (disjoint by construction)
    n_debate_eval = len(ev_gym.get("debate_judge", []))
    if n_debate_eval:
        print(f"Eval debate_judge: {n_debate_eval} (EVAL topics)…", flush=True)
        train_excl = set(exclude.get("debate_judge", set()))
        kept = []
        ep_rng = random.Random(args.seed + 333)
        spins = 0
        while len(kept) < n_debate_eval and spins < n_debate_eval * 100:
            spins += 1
            for s in generate_debate_episode(
                random.Random(ep_rng.randint(0, 2**31 - 1)),
                hard=False,
                split="eval",
            ):
                row = ensure_gym(s.to_json(), "debate_judge")
                # Topic disjoint ⇒ train miss; allow within-eval state reuse
                if fingerprint(row, mode="state_only") in train_excl:
                    continue
                kept.append(row)
                if len(kept) >= n_debate_eval:
                    break
        if len(kept) < n_debate_eval:
            raise SystemExit(f"debate eval: only {len(kept)}/{n_debate_eval}")
        add_eval("debate_judge", kept[:n_debate_eval])

    # Word games — EVAL vocab
    n_word_eval = len(ev_gym.get("word_games", []))
    if n_word_eval:
        print(f"Eval word_games: {n_word_eval} (EVAL vocab)…", flush=True)

        def word_factory(count: int, seed: int) -> list[DistillSample]:
            return generate_word_game_samples(count, seed=seed, split="eval")

        rows = generate_disjoint(
            word_factory,
            n_word_eval,
            exclude=exclude.setdefault("word_games", set()),
            mode="state_only",
            seed=args.seed + 444,
            batch=max(64, n_word_eval),
        )
        add_eval("word_games", rows)

    # CA
    n_ca = len(ev_gym.get("cellular_automata", []))
    if n_ca:
        print(f"Eval cellular_automata: {n_ca} with rejection…", flush=True)

        def ca_factory(count: int, seed: int) -> list[DistillSample]:
            return generate_ca_samples(count, seed=seed)

        rows = generate_disjoint(
            ca_factory,
            n_ca,
            exclude=exclude.setdefault("cellular_automata", set()),
            mode="state_only",
            seed=args.seed + 555,
            batch=max(64, n_ca),
        )
        add_eval("cellular_automata", rows)

    # Preserve any unexpected gyms from old eval (shouldn't happen)
    known = {r.get("meta", {}).get("gym") for r in new_eval}
    for gym, rows in ev_gym.items():
        if gym not in known and gym != "?":
            print(f"WARNING: carrying over unevaluated gym {gym} n={len(rows)}")
            add_eval(gym, rows)

    rng.shuffle(new_eval)

    after = overlap_report(train, new_eval, mode="state_task")
    print("AFTER  state_task overlap:", {k: f"{v:.1%}" for k, v in after.items()})
    after_state = overlap_report(train, new_eval, mode="state_only")
    print("AFTER  state_only overlap:", {k: f"{v:.1%}" for k, v in after_state.items()})

    bad = {g: r for g, r in after.items() if r > args.max_overlap + 1e-12}
    if bad:
        raise SystemExit(f"overlap still above {args.max_overlap}: {bad}")

    write_jsonl(args.train, train)
    write_jsonl(args.eval, new_eval)
    print(f"Wrote train n={len(train)} → {args.train}")
    print(f"Wrote eval  n={len(new_eval)} → {args.eval}")
    print("eval gyms:", dict(Counter((r.get("meta") or {}).get("gym") for r in new_eval)))

    if args.also_diversity:
        print("Rewriting synth_diversity_*.jsonl …", flush=True)
        ca_n_tr = len(tr_gym.get("cellular_automata", [])) or 12_000
        div_train_path = ROOT / "data" / "synth_diversity_train.jsonl"
        div_eval_path = ROOT / "data" / "synth_diversity_eval.jsonl"
        if div_train_path.exists():
            old_div_tr = by_gym(load_jsonl(div_train_path))
            ca_n_tr = len(old_div_tr.get("cellular_automata", [])) or ca_n_tr
            word_n_tr = len(old_div_tr.get("word_games", [])) or n_word_train or 12_000
        else:
            word_n_tr = n_word_train or 12_000
        if div_eval_path.exists():
            old_div_ev = by_gym(load_jsonl(div_eval_path))
            ca_n_ev = len(old_div_ev.get("cellular_automata", [])) or 400
            word_n_ev = len(old_div_ev.get("word_games", [])) or 400
        else:
            ca_n_ev, word_n_ev = 400, 400

        ca_tr = [
            ensure_gym(s.to_json(), "cellular_automata")
            for s in generate_ca_samples(ca_n_tr, seed=args.seed + 7)
        ]
        word_tr = [
            ensure_gym(s.to_json(), "word_games")
            for s in generate_word_game_samples(
                word_n_tr, seed=args.seed + 8, split="train"
            )
        ]
        ca_excl = collect_fingerprints(ca_tr, mode="state_only")
        word_excl = collect_fingerprints(word_tr, mode="state_only")

        def ca_f(c: int, s: int) -> list[DistillSample]:
            return generate_ca_samples(c, seed=s)

        def word_f(c: int, s: int) -> list[DistillSample]:
            return generate_word_game_samples(c, seed=s, split="eval")

        ca_ev = [
            ensure_gym(r, "cellular_automata")
            for r in generate_disjoint(
                ca_f, ca_n_ev, exclude=ca_excl, mode="state_only", seed=args.seed + 9
            )
        ]
        word_ev = [
            ensure_gym(r, "word_games")
            for r in generate_disjoint(
                word_f,
                word_n_ev,
                exclude=word_excl,
                mode="state_only",
                seed=args.seed + 10,
            )
        ]
        div_tr = ca_tr + word_tr
        div_ev = ca_ev + word_ev
        rng.shuffle(div_tr)
        rng.shuffle(div_ev)
        write_jsonl(div_train_path, div_tr)
        write_jsonl(div_eval_path, div_ev)
        print(f"  diversity train={len(div_tr)} eval={len(div_ev)}")

    # Rebuild general_train / general_eval debate portions for future builds
    gen_train_path = ROOT / "data" / "general_train.jsonl"
    gen_eval_path = ROOT / "data" / "general_eval.jsonl"
    if gen_train_path.exists() and gen_eval_path.exists():
        print("Scrubbing debate rows in general_train/eval …", flush=True)
        gtr = load_jsonl(gen_train_path)
        gev = load_jsonl(gen_eval_path)
        n_gtr_d = sum(1 for r in gtr if (r.get("meta") or {}).get("gym") == "debate_judge")
        n_gev_d = sum(1 for r in gev if (r.get("meta") or {}).get("gym") == "debate_judge")
        gtr = [r for r in gtr if (r.get("meta") or {}).get("gym") != "debate_judge"]
        gev = [r for r in gev if (r.get("meta") or {}).get("gym") != "debate_judge"]
        if n_gtr_d:
            gtr.extend(regen_debate(n_gtr_d, split="train", seed=args.seed + 60))
        if n_gev_d:
            # EVAL topics only — topic disjoint ⇒ zero train contamination
            kept = regen_debate(n_gev_d, split="eval", seed=args.seed + 61)
            gev.extend(kept)
        rng.shuffle(gtr)
        rng.shuffle(gev)
        write_jsonl(gen_train_path, gtr)
        write_jsonl(gen_eval_path, gev)
        # Keep general_distill in sync
        write_jsonl(ROOT / "data" / "general_distill.jsonl", gtr + gev)
        print(f"  general_train={len(gtr)} general_eval={len(gev)}")

    print("DONE — zero-contamination rebuild complete.")


if __name__ == "__main__":
    main()
