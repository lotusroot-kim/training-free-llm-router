#!/usr/bin/env python3
"""Compact GEPA-style prompt optimization for the memory router (no DSPy).

GEPA (Agrawal et al. 2025, "Reflective Prompt Evolution") in three moves, which
we implement directly for our single optimizable module — the routing
INSTRUCTION block in MemoryRouter._prompt:

  1. ROLLOUT + RICH FEEDBACK: evaluate a candidate instruction on a dev minibatch.
     For each query we don't just score it — we emit a natural-language note:
       * picked OPUS but HAIKU was also correct   -> "wasteful: paid 5x for nothing"
       * picked HAIKU but it was wrong, OPUS right -> "under-routed: lost accuracy"
       * correct & cheap                           -> "good"
  2. REFLECTIVE MUTATION: a strong model (OPUS) reads the current instruction plus
     a sample of those feedback notes and proposes a BETTER instruction.
  3. PARETO SELECTION: keep a pool of candidates on the Pareto front of
     per-example scores (not just the single best mean) to preserve diversity
     and escape local optima. New candidates are mutated from a Pareto-sampled
     parent.

Objective (cost-aware): per query  reward = perf - LAMBDA * serving_cost_usd.

Cost ledger: dev rollouts (haiku decisions) + opus reflections are the router's
OPTIMIZATION cost — tracked and reported separately, never folded into serving.
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import (opus_reflect, TITAN_RATE, HAIKU_IN_RATE, HAIKU_OUT_RATE,
                    OPUS_IN_RATE, OPUS_OUT_RATE)
from router import MemoryRouter, SEED_INSTRUCTION


def reward(perf, cost, lam):
    """perf in [0,1]; cost ~$0.003..$0.02. lam scales the cost penalty."""
    return perf - lam * cost


def make_feedback(r):
    """Natural-language note for one routed dev example (reflection fuel)."""
    c, hp, op = r["choice"], r["haiku_perf"], r["opus_perf"]
    hc, oc = r["haiku_cost"], r["opus_cost"]
    if c == "opus" and hp >= 0.5:
        return (f"WASTEFUL: routed OPUS (${oc:.4f}) but HAIKU was also correct "
                f"(${hc:.4f}) — paid ~{oc/max(hc,1e-9):.0f}x for the same answer.")
    if c == "opus" and op >= 0.5 and hp < 0.5:
        return "GOOD: routed OPUS and it was needed (HAIKU failed, OPUS solved)."
    if c == "opus" and op < 0.5:
        return "BAD: routed OPUS (expensive) but even OPUS failed — wasted money."
    if c == "haiku" and hp >= 0.5:
        return "GOOD: routed HAIKU and it was correct — cheapest right answer."
    if c == "haiku" and hp < 0.5 and op >= 0.5:
        return ("UNDER-ROUTED: routed HAIKU but it was WRONG; OPUS would have "
                "solved it — lost accuracy to save a little money.")
    return "routed HAIKU; both models failed (hard problem)."


# ----------------------------- evaluation -----------------------------

class Evaluator:
    """Evaluates an instruction over dev examples using leave-self-out kNN.

    Dev examples come straight from memory (they already have embeddings), so
    rollouts cost only the haiku decision call — no re-embedding.
    """

    def __init__(self, router, dev, lam, workers):
        self.router = router
        self.dev = dev            # list of memory records (have 'embedding')
        self.lam = lam
        self.workers = workers
        self.rollout_cost = 0.0   # haiku decision $ spent during optimization

    def evaluate(self, instruction):
        self.router.instruction = instruction

        def one(rec):
            choice, info = self.router.route(
                rec["query"], qvec=rec["embedding"], q_embed_tokens=0,
                exclude_id=rec["id"])
            perf = rec[f"{choice}_perf"]
            cost = rec[f"{choice}_cost"]
            return {
                "id": rec["id"], "choice": choice,
                "perf": perf, "serving_cost": cost,
                "haiku_perf": rec["haiku_perf"], "opus_perf": rec["opus_perf"],
                "haiku_cost": rec["haiku_cost"], "opus_cost": rec["opus_cost"],
                "reward": reward(perf, cost, self.lam),
            }

        before = self.router.router_cost
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            rows = list(ex.map(one, self.dev))
        self.rollout_cost += self.router.router_cost - before

        rewards = np.array([r["reward"] for r in rows])
        return {
            "rows": rows,
            "per_example": rewards,
            "mean_reward": float(rewards.mean()),
            "acc": float(np.mean([r["perf"] for r in rows])),
            "serving_cost": float(np.mean([r["serving_cost"] for r in rows])),
            "pct_opus": float(np.mean([r["choice"] == "opus" for r in rows])),
        }


# ----------------------------- reflection -----------------------------

REFLECT_TEMPLATE = """You are optimizing the INSTRUCTION given to a cheap model (HAIKU) that routes a math query to either HAIKU (cheap) or OPUS (~5x pricier but stronger). The router sees, for the most similar past problems, whether HAIKU/OPUS were correct and their costs, then must answer HAIKU or OPUS.

The optimization objective is cost-aware: maximize  accuracy - {lam} * serving_cost_usd  per query. So routing OPUS when HAIKU would also have been correct is wasteful; routing HAIKU when only OPUS can solve it loses accuracy.

CURRENT INSTRUCTION:
\"\"\"{instruction}\"\"\"

On a dev sample, this instruction produced accuracy={acc:.3f}, serving_cost=${cost:.5f}/query, {pct_opus:.0%} routed to OPUS. Here are outcome notes from individual routed queries (a representative mix of its mistakes and successes):

{feedback}

Diagnose the dominant failure mode (over-routing to OPUS? under-routing? mis-reading the neighbor stats?) and write an IMPROVED instruction that fixes it while keeping what works. The instruction must:
- be self-contained guidance the router reads before the neighbor history and query,
- tell it concretely how to weigh the neighbor success-rates and costs,
- stay concise (at most ~8 sentences),
- NOT mention this optimization process or these notes.

Return ONLY the improved instruction text, with no preamble, quotes, or explanation."""


class Reflector:
    def __init__(self):
        self.reflect_cost = 0.0
        self.n = 0

    def mutate(self, instruction, ev, lam, n_notes=18, rng=random):
        rows = ev["rows"]
        # bias the sample toward informative (non-"GOOD") cases
        notes = [make_feedback(r) for r in rows]
        bad = [n for n in notes if not n.startswith("GOOD")]
        good = [n for n in notes if n.startswith("GOOD")]
        rng.shuffle(bad); rng.shuffle(good)
        sample = (bad[: int(n_notes * 0.7)] + good[: n_notes - int(n_notes * 0.7)])
        rng.shuffle(sample)
        prompt = REFLECT_TEMPLATE.format(
            lam=lam, instruction=instruction, acc=ev["acc"],
            cost=ev["serving_cost"], pct_opus=ev["pct_opus"],
            feedback="\n".join(f"- {s}" for s in sample))
        txt, it, ot = opus_reflect(prompt)
        self.reflect_cost += it * OPUS_IN_RATE + ot * OPUS_OUT_RATE
        self.n += 1
        return _clean_instruction(txt)


def _clean_instruction(txt):
    """Strip reflector meta-text (a leading **Diagnosis:** paragraph, surrounding
    quotes/markdown fences) so only the usable instruction remains."""
    t = txt.strip().strip('"').strip()
    t = re.sub(r"^```[a-z]*\n?|\n?```$", "", t).strip()
    # drop a leading diagnosis/analysis paragraph if the model prepended one
    paras = re.split(r"\n\s*\n", t)
    if paras and re.match(r"^\**\s*(diagnosis|analysis|the dominant)", paras[0], re.I):
        paras = paras[1:]
    return "\n\n".join(paras).strip()


# ----------------------------- Pareto pool -----------------------------

def pareto_front(cands):
    """Keep candidates not strictly dominated on per-example reward vectors.

    Candidate A dominates B if A >= B on every dev example and > on at least one.
    GEPA selects parents from this front (here: pick the highest mean among a
    random subset of the front) so diverse winners survive.
    """
    front = []
    for i, c in enumerate(cands):
        ci = c["per_example"]
        dominated = False
        for j, o in enumerate(cands):
            if i == j:
                continue
            oj = o["per_example"]
            if np.all(oj >= ci) and np.any(oj > ci):
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front


def run_gepa(router, dev, lam, iters, children, workers, rng, verbose=True):
    """Run the GEPA loop for one lambda. Returns (best_candidate, opt_cost_breakdown).

    best_candidate has the winning instruction plus its dev metrics; the cost
    breakdown is the optimization ledger (haiku rollouts + opus reflections),
    kept entirely separate from serving cost.
    """
    ev_engine = Evaluator(router, dev, lam, workers)
    reflector = Reflector()

    def score(instr):
        return ev_engine.evaluate(instr)

    seed_ev = score(SEED_INSTRUCTION)
    pool = [{"instruction": SEED_INSTRUCTION, **seed_ev, "gen": 0}]
    if verbose:
        print(f"[seed] reward={seed_ev['mean_reward']:.4f} acc={seed_ev['acc']:.3f} "
              f"cost=${seed_ev['serving_cost']:.5f} %opus={seed_ev['pct_opus']:.2f}")

    best = pool[0]
    for it in range(1, iters + 1):
        front = pareto_front(pool)
        subset = rng.sample(front, k=min(len(front), 3))
        parent = max(subset, key=lambda c: c["mean_reward"])
        for ch in range(children):
            new_instr = reflector.mutate(parent["instruction"], parent, lam, rng=rng)
            if not new_instr or len(new_instr) < 20:
                continue
            ev = score(new_instr)
            cand = {"instruction": new_instr, **ev, "gen": it}
            pool.append(cand)
            if verbose:
                tag = "  *NEW BEST*" if ev["mean_reward"] > best["mean_reward"] else ""
                print(f"[iter {it}.{ch}] reward={ev['mean_reward']:.4f} "
                      f"acc={ev['acc']:.3f} cost=${ev['serving_cost']:.5f} "
                      f"%opus={ev['pct_opus']:.2f}{tag}")
            if ev["mean_reward"] > best["mean_reward"]:
                best = cand

    cost = {
        "haiku_rollout_usd": ev_engine.rollout_cost,
        "opus_reflect_usd": reflector.reflect_cost,
        "n_reflections": reflector.n,
        "total_optimization_usd": ev_engine.rollout_cost + reflector.reflect_cost,
    }
    return best, seed_ev, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="memory.jsonl")
    ap.add_argument("--dev", type=int, default=120, help="dev minibatch size")
    ap.add_argument("--iters", type=int, default=8, help="reflection rounds")
    ap.add_argument("--children", type=int, default=2, help="mutations per round")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lam", type=float, default=20.0, help="cost penalty weight")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gepa_best_instruction.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    router = MemoryRouter(args.memory, k=args.k)
    idx = list(range(len(router.mem)))
    rng.shuffle(idx)
    dev = [router.mem[i] for i in idx[: args.dev]]
    print(f"dev minibatch: {len(dev)} queries  | lambda={args.lam} | k={args.k}")

    best, seed_ev, cost = run_gepa(router, dev, args.lam, args.iters,
                                   args.children, args.workers, rng)

    print("\n================ GEPA DONE ================")
    print(f"seed   reward={seed_ev['mean_reward']:.4f} acc={seed_ev['acc']:.3f} "
          f"cost=${seed_ev['serving_cost']:.5f} %opus={seed_ev['pct_opus']:.2f}")
    print(f"BEST   reward={best['mean_reward']:.4f} acc={best['acc']:.3f} "
          f"cost=${best['serving_cost']:.5f} %opus={best['pct_opus']:.2f} (gen {best['gen']})")
    print("\n--- best instruction ---\n" + best["instruction"])

    print("\n---------------- OPTIMIZATION COST (separate ledger) ----------------")
    print(f"haiku dev-rollout $ : {cost['haiku_rollout_usd']:.5f}")
    print(f"opus reflection $   : {cost['opus_reflect_usd']:.5f}  ({cost['n_reflections']} reflections)")
    print(f"total optimization $: {cost['total_optimization_usd']:.5f}  (one-time, NOT serving cost)")

    with open(args.out, "w") as f:
        f.write(best["instruction"])
    with open("gepa_run.json", "w") as f:
        json.dump({
            "seed": {k: seed_ev[k] for k in ("mean_reward", "acc", "serving_cost", "pct_opus")},
            "best": {k: best[k] for k in ("mean_reward", "acc", "serving_cost", "pct_opus", "gen")},
            "best_instruction": best["instruction"],
            "lambda": args.lam, "dev": args.dev, "iters": args.iters,
            "optimization_cost_usd": cost["total_optimization_usd"],
            "haiku_rollout_usd": cost["haiku_rollout_usd"],
            "opus_reflect_usd": cost["opus_reflect_usd"],
        }, f, indent=2)
    print(f"\nsaved {args.out} and gepa_run.json")


if __name__ == "__main__":
    main()
