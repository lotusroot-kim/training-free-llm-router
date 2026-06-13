#!/usr/bin/env python3
"""GEPA for the SCORE prompt: sharpen the router's continuous estimates so the
threshold cost/accuracy curve rises toward the oracle.

Unlike gepa_optimize.py (which tuned the BINARY decision for a single cost-aware
reward), here the optimized signal is CONTINUOUS — per query the router emits
P(correct) and predicted output tokens for both models. The objective is the
quality of the resulting curve:

    metric = mean( curve_accuracy(cost) - interpolation_line_accuracy(cost) )

i.e. how far the honest cost-aware threshold curve sits ABOVE the
haiku<->opus mix line. A sharper, better-calibrated signal orders queries better
and lifts the whole curve.

GEPA loop (same three moves as before):
  1. rollout on a dev minibatch -> per-query estimates + recorded truth.
  2. natural-language feedback on the worst calibration errors
     (confident-but-wrong P, badly-mis-estimated token counts).
  3. OPUS reflects on the current score-instruction + those notes -> improved one.
Pareto pool over per-(cost-grid) curve accuracy preserves diverse winners.

Optimization spend (haiku rollouts + opus reflections) is its own ledger.
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import (opus_reflect, OUT_RATE, IN_RATE,
                    TITAN_RATE, HAIKU_IN_RATE, HAIKU_OUT_RATE,
                    OPUS_IN_RATE, OPUS_OUT_RATE)
from router import MemoryRouter, SEED_SCORE_INSTRUCTION

COST_GRID = None   # set after we know Hc..Oc


def curve_pts(score, hp, op, hc, oc):
    n = len(score)
    idx = np.argsort(-score)
    to_o = np.zeros(n, bool)
    pts = [(hc.mean(), hp.mean())]
    for j in idx:
        to_o[j] = True
        pts.append((np.where(to_o, oc, hc).mean(), np.where(to_o, op, hp).mean()))
    return np.array(sorted(pts))


def acc_at(c, b):
    c = c[c[:, 0].argsort()]
    return float(np.interp(b, c[:, 0], c[:, 1]))


def above_line(c, grid, Hc, Ha, Oc, Oa):
    line = Ha + (grid - Hc) * (Oa - Ha) / (Oc - Hc)
    cur = np.array([acc_at(c, b) for b in grid])
    return cur - line                                # per-grid vector (for Pareto)


class ScoreEvaluator:
    """Rollout the score prompt on dev; build the honest cost-aware curve and
    measure how far above the interpolation line it sits."""

    def __init__(self, router, dev, workers, cot=False):
        self.router = router
        self.dev = dev
        self.workers = workers
        self.cot = cot                     # CoT scoring mode (haiku reasons first)
        self.rollout_cost = 0.0
        # truth arrays from dev (recorded)
        self.hp = np.array([d["haiku_perf"] for d in dev], float)
        self.op = np.array([d["opus_perf"] for d in dev], float)
        self.hc = np.array([d["haiku_cost"] for d in dev], float)
        self.oc = np.array([d["opus_cost"] for d in dev], float)
        self.it_h = np.array([d.get("haiku_in", 0) for d in dev], float)
        self.it_o = np.array([d.get("opus_in", 0) for d in dev], float)
        self.Ha, self.Hc = self.hp.mean(), self.hc.mean()
        self.Oa, self.Oc = self.op.mean(), self.oc.mean()
        self.grid = np.linspace(self.Hc, self.Oc, 40)

    def evaluate(self, score_instruction):
        self.router.score_instruction = score_instruction

        def one(d):
            p_h, p_o, info = self.router.score(
                d["query"], qvec=d["embedding"], q_embed_tokens=0, exclude_id=d["id"],
                cot=self.cot)
            return p_h, p_o, info["pred_haiku_out"], info["pred_opus_out"]

        before = self.router.router_cost
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            res = list(ex.map(one, self.dev))
        self.rollout_cost += self.router.router_cost - before

        p_h = np.array([r[0] for r in res], float)
        p_o = np.array([r[1] for r in res], float)
        oh = np.array([r[2] for r in res], float)
        oo = np.array([r[3] for r in res], float)
        # honest predicted cost from the router's own token estimates
        pred_hc = OUT_RATE["haiku"] * oh + IN_RATE["haiku"] * self.it_h
        pred_oc = OUT_RATE["opus"] * oo + IN_RATE["opus"] * self.it_o
        signal = (p_o - p_h) / np.maximum(pred_oc - pred_hc, 1e-6)

        c = curve_pts(signal, self.hp, self.op, self.hc, self.oc)
        diff = above_line(c, self.grid, self.Hc, self.Ha, self.Oc, self.Oa)
        return {
            "mean_above": float(diff.mean()),
            "per_grid": diff,
            "corr_po": float(np.corrcoef(p_o, self.op)[0, 1]) if p_o.std() > 0 else 0.0,
            "corr_ph": float(np.corrcoef(p_h, self.hp)[0, 1]) if p_h.std() > 0 else 0.0,
            "corr_dcost": float(np.corrcoef(pred_oc - pred_hc, self.oc - self.hc)[0, 1])
                          if (pred_oc - pred_hc).std() > 0 else 0.0,
            "rows": list(zip(p_h, p_o, oh, oo)),
        }


def feedback_notes(ev_engine, ev, n=18, rng=random):
    """Worst calibration errors as natural-language notes for reflection."""
    notes = []
    for i, (p_h, p_o, oh, oo) in enumerate(ev["rows"]):
        thp, top = ev_engine.hp[i], ev_engine.op[i]
        tho, too = ev_engine.oc[i], ev_engine.hc[i]  # not tokens; use cost proxy below
        # correctness calibration errors
        if p_o >= 0.7 and top < 0.5:
            notes.append((abs(p_o - top), f"Over-confident on OPUS: estimated {p_o:.0%} correct "
                          f"but OPUS was WRONG."))
        if p_o <= 0.4 and top >= 0.5:
            notes.append((abs(p_o - top), f"Under-rated OPUS: estimated {p_o:.0%} correct "
                          f"but OPUS was RIGHT."))
        if p_h >= 0.7 and thp < 0.5:
            notes.append((abs(p_h - thp), f"Over-confident on HAIKU: estimated {p_h:.0%} "
                          f"correct but HAIKU was WRONG."))
        if p_h <= 0.4 and thp >= 0.5:
            notes.append((abs(p_h - thp), f"Under-rated HAIKU: estimated {p_h:.0%} correct "
                          f"but HAIKU was RIGHT."))
    notes.sort(key=lambda x: -x[0])
    picked = [m for _, m in notes[: n * 2]]
    rng.shuffle(picked)
    return picked[:n]


REFLECT_TEMPLATE = """You are improving an INSTRUCTION given to a cheap model (HAIKU) that, for a math query, must ESTIMATE for two models — HAIKU (cheap) and OPUS (stronger) — (a) the probability each answers correctly and (b) how many output tokens each would generate. It sees the correctness and token counts of the most similar past problems, then outputs: HAIKU=<correct%>,<tokens> OPUS=<correct%>,<tokens>.

These estimates feed a router: queries are escalated to OPUS in order of (P_opus_correct - P_haiku_correct) / predicted_extra_cost. So the estimates must be well CALIBRATED and DISCRIMINATIVE — separating queries OPUS will get right but HAIKU won't, from queries both handle, and predicting token counts that track real difficulty.

CURRENT INSTRUCTION:
\"\"\"{instruction}\"\"\"

On a dev sample this gave: correlation(P_opus, opus_correct)={corr_po:.2f}, correlation(P_haiku, haiku_correct)={corr_ph:.2f}, correlation(pred Δcost, true Δcost)={corr_dcost:.2f}. Higher is sharper. Calibration errors observed:

{feedback}

Diagnose why the estimates are weakly correlated with truth (e.g. clustering all probabilities near the middle, ignoring neighbor disagreement, ignoring token evidence, over-trusting weak similarities) and write an IMPROVED instruction that makes the estimates sharper and better separated. It must:
- tell the model concretely how to turn the neighbors' correctness AND token counts into the two probabilities and two token estimates,
- encourage decisive, spread-out probabilities when the neighbor evidence is clear (not everything near 50%),
- stay concise (<= 8 sentences),
- NOT mention this optimization or these notes.

Return ONLY the improved instruction text, no preamble or quotes."""


def clean(txt):
    t = txt.strip().strip('"').strip()
    t = re.sub(r"^```[a-z]*\n?|\n?```$", "", t).strip()
    paras = re.split(r"\n\s*\n", t)
    if paras and re.match(r"^\**\s*(diagnosis|analysis|the )", paras[0], re.I) and len(paras) > 1:
        paras = paras[1:]
    return "\n\n".join(paras).strip()


def pareto_front(cands):
    front = []
    for i, c in enumerate(cands):
        ci = c["per_grid"]
        if not any(i != j and np.all(o["per_grid"] >= ci) and np.any(o["per_grid"] > ci)
                   for j, o in enumerate(cands)):
            front.append(c)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="memory.jsonl")
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gepa_score_instruction.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    router = MemoryRouter(args.memory, k=args.k)
    idx = list(range(len(router.mem)))
    rng.shuffle(idx)
    dev = [router.mem[i] for i in idx[: args.dev]]
    print(f"dev={len(dev)}  k={args.k}  (optimizing the SCORE prompt)")

    ev_engine = ScoreEvaluator(router, dev, args.workers)
    reflect_cost = [0.0]
    n_reflect = [0]

    def mutate(instr, ev):
        notes = feedback_notes(ev_engine, ev, rng=rng)
        prompt = REFLECT_TEMPLATE.format(
            instruction=instr, corr_po=ev["corr_po"], corr_ph=ev["corr_ph"],
            corr_dcost=ev["corr_dcost"],
            feedback="\n".join(f"- {s}" for s in notes) or "- (few calibration errors)")
        txt, it, ot = opus_reflect(prompt, max_tokens=700)
        reflect_cost[0] += it * OPUS_IN_RATE + ot * OPUS_OUT_RATE
        n_reflect[0] += 1
        return clean(txt)

    seed_ev = ev_engine.evaluate(SEED_SCORE_INSTRUCTION)
    pool = [{"instruction": SEED_SCORE_INSTRUCTION, **seed_ev, "gen": 0}]
    best = pool[0]
    print(f"[seed] above_line={seed_ev['mean_above']:+.4f} "
          f"corr_po={seed_ev['corr_po']:.2f} corr_ph={seed_ev['corr_ph']:.2f} "
          f"corr_dcost={seed_ev['corr_dcost']:.2f}")

    for it in range(1, args.iters + 1):
        front = pareto_front(pool)
        parent = max(rng.sample(front, k=min(len(front), 3)), key=lambda c: c["mean_above"])
        for ch in range(args.children):
            ni = mutate(parent["instruction"], parent)
            if not ni or len(ni) < 20:
                continue
            ev = ev_engine.evaluate(ni)
            cand = {"instruction": ni, **ev, "gen": it}
            pool.append(cand)
            tag = "  *NEW BEST*" if ev["mean_above"] > best["mean_above"] else ""
            print(f"[iter {it}.{ch}] above_line={ev['mean_above']:+.4f} "
                  f"corr_po={ev['corr_po']:.2f} corr_ph={ev['corr_ph']:.2f} "
                  f"corr_dcost={ev['corr_dcost']:.2f}{tag}")
            if ev["mean_above"] > best["mean_above"]:
                best = cand

    print("\n================ GEPA-SCORE DONE ================")
    print(f"seed above_line={seed_ev['mean_above']:+.4f} (corr_po={seed_ev['corr_po']:.2f})")
    print(f"BEST above_line={best['mean_above']:+.4f} (corr_po={best['corr_po']:.2f}, gen {best['gen']})")
    print("\n--- best score instruction ---\n" + best["instruction"])

    opt = ev_engine.rollout_cost + reflect_cost[0]
    print("\n---------------- OPTIMIZATION COST (separate ledger) ----------------")
    print(f"haiku dev-rollout $ : {ev_engine.rollout_cost:.5f}")
    print(f"opus reflection $   : {reflect_cost[0]:.5f}  ({n_reflect[0]} reflections)")
    print(f"total optimization $: {opt:.5f}  (one-time, NOT serving cost)")

    with open(args.out, "w") as f:
        f.write(best["instruction"])
    with open("gepa_score_run.json", "w") as f:
        json.dump({"seed_above": seed_ev["mean_above"], "best_above": best["mean_above"],
                   "seed_corr_po": seed_ev["corr_po"], "best_corr_po": best["corr_po"],
                   "best_instruction": best["instruction"], "optimization_cost_usd": opt}, f, indent=2)
    print(f"\nsaved {args.out} and gepa_score_run.json")


if __name__ == "__main__":
    main()
