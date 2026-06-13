#!/usr/bin/env python3
"""Sweep the cost-penalty lambda to trace GEPA's cost/accuracy Pareto frontier.

For each lambda in --lams we run the GEPA loop (reward = acc - lambda*cost) on a
fixed dev minibatch, take the best evolved instruction, then evaluate it on the
HELD-OUT test set. Plotting the resulting (serving_cost, accuracy) points gives
the frontier of operating points GEPA can reach:

  small lambda -> cost barely penalized -> accuracy-first  (more OPUS, up-right)
  large lambda -> cost heavily penalized -> thrift-first   (more HAIKU, down-left)

Each lambda's evolved instruction is saved (gepa_instr_lam{lam}.txt) and its test
metrics go into gepa_pareto.json; the frontier is drawn to gepa_pareto.png.

Optimization spend (haiku dev rollouts + opus reflections) is summed across all
lambdas and reported separately — never folded into serving cost.
"""
import argparse
import json
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gepa_optimize import run_gepa
from router import MemoryRouter, SEED_INSTRUCTION


def eval_on_test(router, test, instruction, workers):
    """Route every test query with `instruction`; score from recorded perf/cost."""
    from concurrent.futures import ThreadPoolExecutor
    router.instruction = instruction

    def one(r):
        choice, _ = router.route(r["query"])      # test queries: real embed + haiku call
        return r[f"{choice}_perf"], r[f"{choice}_cost"], choice == "opus"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, test))
    perf = np.array([x[0] for x in rows], float)
    cost = np.array([x[1] for x in rows], float)
    pop = np.array([x[2] for x in rows], float)
    return {"acc": float(perf.mean()), "serving_cost": float(cost.mean()),
            "pct_opus": float(pop.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="memory.jsonl")
    ap.add_argument("--test", default="../llmrouter/router_multihead_test.jsonl")
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[2.0, 5.0, 10.0, 20.0, 40.0])
    ap.add_argument("--dev", type=int, default=120)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_limit", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()

    test = [json.loads(l) for l in open(args.test)]
    if args.test_limit:
        test = test[: args.test_limit]

    # one router/memory shared across lambdas; dev minibatch fixed for fairness
    router = MemoryRouter(args.memory, k=args.k)
    rng_dev = random.Random(args.seed)
    idx = list(range(len(router.mem)))
    rng_dev.shuffle(idx)
    dev = [router.mem[i] for i in idx[: args.dev]]
    print(f"dev={len(dev)}  test={len(test)}  lambdas={args.lams}")

    # baselines on test (recorded perf/cost) for context
    a = {k: np.array([r[k] for r in test], float)
         for k in ("haiku_perf", "opus_perf", "haiku_cost", "opus_cost")}
    base = {
        "all-haiku": (a["haiku_cost"].mean(), a["haiku_perf"].mean()),
        "all-opus": (a["opus_cost"].mean(), a["opus_perf"].mean()),
    }
    orc_cost = np.where(a["haiku_perf"] >= 0.5, a["haiku_cost"], a["opus_cost"])
    orc_perf = np.where((a["haiku_perf"] >= 0.5) | (a["opus_perf"] >= 0.5), 1.0, 0.0)
    base["oracle"] = (orc_cost.mean(), orc_perf.mean())

    results = []
    total_opt_cost = 0.0
    for lam in args.lams:
        print(f"\n========== GEPA @ lambda={lam} ==========")
        rng = random.Random(args.seed)           # same seed -> comparable runs
        np.random.seed(args.seed)
        best, seed_ev, optc = run_gepa(router, dev, lam, args.iters,
                                       args.children, args.workers, rng)
        total_opt_cost += optc["total_optimization_usd"]
        instr_path = f"gepa_instr_lam{lam:g}.txt"
        with open(instr_path, "w") as f:
            f.write(best["instruction"])
        # evaluate the evolved instruction on held-out test
        t = eval_on_test(router, test, best["instruction"], args.workers)
        print(f"  TEST: acc={t['acc']:.3f} cost=${t['serving_cost']:.5f} "
              f"%opus={t['pct_opus']:.2f}  (dev reward={best['mean_reward']:.3f})")
        results.append({"lambda": lam, "instruction_path": instr_path,
                        "dev": {k: best[k] for k in ("mean_reward", "acc",
                                "serving_cost", "pct_opus")},
                        "test": t, "opt_cost_usd": optc["total_optimization_usd"]})

    # also evaluate the (unoptimized) seed on test as the starting point
    seed_test = eval_on_test(router, test, SEED_INSTRUCTION, args.workers)

    # ---- plot frontier ----
    fig, ax = plt.subplots(figsize=(8.5, 6))
    xs = [r["test"]["serving_cost"] for r in results]
    ys = [r["test"]["acc"] for r in results]
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], "-", color="crimson",
            alpha=0.5, zorder=2, label="GEPA frontier (lambda sweep)")
    for r in results:
        ax.scatter(r["test"]["serving_cost"], r["test"]["acc"], s=120,
                   marker="*", color="crimson", zorder=4)
        ax.annotate(f"λ={r['lambda']:g}",
                    (r["test"]["serving_cost"], r["test"]["acc"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.scatter(seed_test["serving_cost"], seed_test["acc"], s=110, marker="X",
               color="darkorange", zorder=4, label="GEPA seed (cost-sensitive baseline)")
    for name, (c, p) in base.items():
        m = {"oracle": "D", "all-haiku": "v", "all-opus": "^"}[name]
        ax.scatter([c], [p], s=90, marker=m, zorder=3, label=name)
    ax.set_xlabel("serving cost ($ / query)  — optimization & router cost tracked separately")
    ax.set_ylabel("accuracy")
    ax.set_title("GEPA cost/accuracy Pareto frontier via lambda sweep (held-out test)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("gepa_pareto.png", dpi=120)

    with open("gepa_pareto.json", "w") as f:
        json.dump({"results": results, "seed_test": seed_test,
                   "baselines": base, "total_optimization_usd": total_opt_cost}, f, indent=2)

    print("\n================ FRONTIER (held-out test) ================")
    print(f"{'lambda':>8}{'acc':>8}{'cost$/q':>10}{'%opus':>8}")
    print(f"{'seed':>8}{seed_test['acc']:>8.3f}{seed_test['serving_cost']:>10.5f}"
          f"{seed_test['pct_opus']:>8.2f}")
    for r in sorted(results, key=lambda x: x["lambda"]):
        t = r["test"]
        print(f"{r['lambda']:>8g}{t['acc']:>8.3f}{t['serving_cost']:>10.5f}{t['pct_opus']:>8.2f}")
    print(f"\ntotal GEPA optimization $ across sweep: {total_opt_cost:.4f}  (one-time)")
    print("saved gepa_pareto.png and gepa_pareto.json")


if __name__ == "__main__":
    main()
