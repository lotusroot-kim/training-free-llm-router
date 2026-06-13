#!/usr/bin/env python3
"""GEPA v2 for the SCORE prompt — adds the GEPA techniques our v1 was missing.

v1 (gepa_score.py) had: reflective mutation, a Pareto front, minibatch eval,
actionable calibration feedback. Reading the GEPA paper (arXiv:2507.19457) and
the gepa-ai reference, v1 was missing four ingredients that the paper credits
for most of its gains. v2 adds all four:

  1. PARETO-FREQUENCY SAMPLING (not best-mean).  The paper selects a parent by
     how many task instances a candidate is *uniquely best* on, then samples
     proportionally. This preserves diversity (a candidate that wins only a few
     hard queries still gets to reproduce) instead of collapsing to the mean-best.

  2. SYSTEM-AWARE MERGE / CROSSOVER.  Periodically take two Pareto candidates
     that excel on DIFFERENT cost regions and ask the reflector to merge their
     complementary strengths into one instruction — the paper's crossover move.

  3. ANCESTOR LINEAGE.  Each mutation is "informed by accumulated lessons from
     all ancestors": we thread a short running list of what past edits tried, so
     the reflector stops re-proposing the same fixes.

  4. MINIBATCH RESAMPLING.  Each round draws a fresh dev minibatch (paper: avoids
     overfitting one batch). We re-score survivors on the new batch so scores
     stay comparable within a round.

Objective unchanged: maximize how far the honest cost-aware threshold curve sits
ABOVE the haiku<->opus interpolation line. Optimization spend is its own ledger.
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import (opus_reflect, OUT_RATE, IN_RATE,
                    OPUS_IN_RATE, OPUS_OUT_RATE)
from router import MemoryRouter, SEED_SCORE_INSTRUCTION
from gepa_score import (ScoreEvaluator, feedback_notes, clean,
                        REFLECT_TEMPLATE)


# ---------------- GEPA technique 1: Pareto-frequency sampling ----------------

def pareto_sample_parent(pool, rng):
    """Pick a parent by how many grid points it is (uniquely) BEST on.

    Build the matrix [cand x grid] of above-line scores; for each grid column
    find the winner(s); a candidate's weight = number of columns it wins. Sample
    proportionally so niche specialists survive (GEPA's diversity-preserving
    selection), not just the global mean-best.
    """
    G = np.array([c["per_grid"] for c in pool])          # [C, grid]
    winners = G.argmax(axis=0)                            # best cand per grid col
    weight = np.bincount(winners, minlength=len(pool)).astype(float)
    weight += 0.1                                         # smoothing: everyone can reproduce
    weight /= weight.sum()
    return pool[rng.choices(range(len(pool)), weights=weight, k=1)[0]]


# ---------------- GEPA technique 2: system-aware merge ----------------

MERGE_TEMPLATE = """You are merging two INSTRUCTIONS that each tell a cheap model how to estimate, for a math query, two models' (HAIKU cheap, OPUS strong) probability-correct and output-token counts from similar past problems. Each instruction is strong on a DIFFERENT part of the cost/accuracy trade-off, so their strengths are complementary.

INSTRUCTION A (excels at the cheap / low-cost end — deciding when HAIKU suffices):
\"\"\"{a}\"\"\"

INSTRUCTION B (excels at the expensive / high-accuracy end — spotting queries that truly need OPUS):
\"\"\"{b}\"\"\"

Write ONE merged instruction that keeps the best, most concrete rules from BOTH — A's discipline about when the cheap model is enough, and B's sharpness about when only OPUS will do. It must tell the model concretely how to turn neighbor correctness AND token counts into the two probabilities and two token estimates, produce decisive, spread-out probabilities when neighbor evidence is clear, stay concise (<= 8 sentences), and not mention this process.

Return ONLY the merged instruction text, no preamble or quotes."""


def split_strength(pool):
    """Return (low_cost_specialist, high_cost_specialist) from the Pareto pool:
    the candidates that win the most grid columns in the cheap half vs dear half."""
    G = np.array([c["per_grid"] for c in pool])
    half = G.shape[1] // 2
    lo = G[:, :half].mean(axis=1).argmax()
    hi = G[:, half:].mean(axis=1).argmax()
    return pool[lo], pool[hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="memory.jsonl")
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument("--pool", type=int, default=400, help="dev pool to resample minibatches from")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--merge_every", type=int, default=3, help="merge round cadence (0=off)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cot", action="store_true",
                    help="evolve the prompt for CoT scoring (haiku reasons before scores)")
    ap.add_argument("--seed_prompt", default="",
                    help="warm-start instruction file ('' = built-in SEED_SCORE_INSTRUCTION)")
    ap.add_argument("--out", default="gepa_score_v2_instruction.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    router = MemoryRouter(args.memory, k=args.k)
    idx = list(range(len(router.mem)))
    rng.shuffle(idx)
    dev_pool = [router.mem[i] for i in idx[: args.pool]]   # resample minibatches from here
    start_instr = (open(args.seed_prompt).read().strip() if args.seed_prompt
                   else SEED_SCORE_INSTRUCTION)
    print(f"dev_pool={len(dev_pool)}  minibatch={args.dev}  k={args.k}  cot={args.cot}  "
          f"warm-start={args.seed_prompt or 'built-in'}  (GEPA v2: "
          f"pareto-sample + merge + lineage + resample)")

    reflect_cost = [0.0]
    n_reflect = [0]
    lineage = []           # accumulated lessons across ancestors (technique 3)

    def new_engine():      # technique 4: fresh minibatch each round
        mb = rng.sample(dev_pool, k=min(args.dev, len(dev_pool)))
        return ScoreEvaluator(router, mb, args.workers, cot=args.cot)

    def mutate(instr, ev, eng):
        notes = feedback_notes(eng, ev, rng=rng)
        hist = ("\n\nLessons already tried by ancestors (do NOT just repeat these; "
                "go further):\n" + "\n".join(f"- {l}" for l in lineage[-6:])) if lineage else ""
        prompt = REFLECT_TEMPLATE.format(
            instruction=instr, corr_po=ev["corr_po"], corr_ph=ev["corr_ph"],
            corr_dcost=ev["corr_dcost"],
            feedback="\n".join(f"- {s}" for s in notes) or "- (few calibration errors)") + hist
        txt, it, ot = opus_reflect(prompt, max_tokens=750)
        reflect_cost[0] += it * OPUS_IN_RATE + ot * OPUS_OUT_RATE
        n_reflect[0] += 1
        return clean(txt)

    def merge(a, b):
        prompt = MERGE_TEMPLATE.format(a=a["instruction"], b=b["instruction"])
        txt, it, ot = opus_reflect(prompt, max_tokens=750)
        reflect_cost[0] += it * OPUS_IN_RATE + ot * OPUS_OUT_RATE
        n_reflect[0] += 1
        return clean(txt)

    # seed
    eng = new_engine()
    seed_ev = eng.evaluate(start_instr)
    pool = [{"instruction": start_instr, **seed_ev, "gen": 0}]
    best = pool[0]
    print(f"[seed] above_line={seed_ev['mean_above']:+.4f} corr_po={seed_ev['corr_po']:.2f}")

    rollout_cost = eng.rollout_cost
    for it in range(1, args.iters + 1):
        eng = new_engine()                                  # technique 4
        # re-score the whole pool on THIS round's minibatch so every comparison
        # (including against the incumbent best) is on identical data.
        for c in pool:
            c.update(eng.evaluate(c["instruction"]))
        rollout_cost += eng.rollout_cost
        round_best = max(pool, key=lambda c: c["mean_above"])

        do_merge = args.merge_every and it % args.merge_every == 0 and len(pool) >= 2
        children = []
        if do_merge:                                        # technique 2
            a, b = split_strength(pool)
            if a is not b:
                children.append(("merge", merge(a, b)))
        while len(children) < args.children:
            parent = pareto_sample_parent(pool, rng)        # technique 1
            children.append(("mut", mutate(parent["instruction"], parent, eng)))

        round_cost0 = eng.rollout_cost
        for kind, ni in children:
            if not ni or len(ni) < 20:
                continue
            ev = eng.evaluate(ni)
            cand = {"instruction": ni, **ev, "gen": it, "kind": kind}
            pool.append(cand)
            tag = "  *NEW BEST*" if ev["mean_above"] > round_best["mean_above"] else ""
            print(f"[iter {it} {kind:5}] above_line={ev['mean_above']:+.4f} "
                  f"corr_po={ev['corr_po']:.2f} corr_dcost={ev['corr_dcost']:.2f}{tag}")
            if ev["mean_above"] > round_best["mean_above"]:
                round_best = cand
            # technique 3: record a one-line lesson from this edit's outcome
            lineage.append(f"a {kind} reaching above_line={ev['mean_above']:+.3f}, "
                           f"corr_po={ev['corr_po']:.2f}")
        rollout_cost += eng.rollout_cost - round_cost0
        best = round_best                                   # best on the latest batch

    print("\n================ GEPA-SCORE v2 DONE ================")
    print(f"seed above_line={seed_ev['mean_above']:+.4f} (corr_po={seed_ev['corr_po']:.2f})")
    print(f"BEST above_line={best['mean_above']:+.4f} (corr_po={best['corr_po']:.2f}, "
          f"gen {best['gen']}, via {best.get('kind','seed')})")
    print("\n--- best score instruction ---\n" + best["instruction"])

    opt = rollout_cost + reflect_cost[0]
    print("\n---------------- OPTIMIZATION COST (separate ledger) ----------------")
    print(f"haiku dev-rollout $ : {rollout_cost:.5f}")
    print(f"opus reflection $   : {reflect_cost[0]:.5f}  ({n_reflect[0]} reflect/merge calls)")
    print(f"total optimization $: {opt:.5f}  (one-time, NOT serving cost)")

    with open(args.out, "w") as f:
        f.write(best["instruction"])
    with open("gepa_score_v2_run.json", "w") as f:
        json.dump({"seed_above": seed_ev["mean_above"], "best_above": best["mean_above"],
                   "best_corr_po": best["corr_po"], "best_via": best.get("kind", "seed"),
                   "best_instruction": best["instruction"], "optimization_cost_usd": opt}, f, indent=2)
    print(f"\nsaved {args.out} and gepa_score_v2_run.json")


if __name__ == "__main__":
    main()
