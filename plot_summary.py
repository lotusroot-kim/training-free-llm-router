#!/usr/bin/env python3
"""Plot the cost/accuracy operating points of every strategy on one chart.

Reads one or more route_decisions*.jsonl files (each a router variant) plus the
recorded baselines, and draws accuracy vs serving $/query so you can see where
the memory router sits relative to all-haiku, all-opus, memory-prior, oracle.
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def baselines(rows):
    a = {k: np.array([r[k] for r in rows], float)
         for k in ("haiku_perf", "opus_perf", "haiku_cost", "opus_cost",
                   "neighbor_haiku_acc", "neighbor_opus_acc")}
    pts = {
        "all-haiku": (a["haiku_cost"].mean(), a["haiku_perf"].mean()),
        "all-opus": (a["opus_cost"].mean(), a["opus_perf"].mean()),
    }
    mp = a["neighbor_opus_acc"] > a["neighbor_haiku_acc"]
    pts["memory-prior"] = (np.where(mp, a["opus_cost"], a["haiku_cost"]).mean(),
                           np.where(mp, a["opus_perf"], a["haiku_perf"]).mean())
    orc_cost = np.where(a["haiku_perf"] >= 0.5, a["haiku_cost"], a["opus_cost"])
    orc_perf = np.where((a["haiku_perf"] >= 0.5) | (a["opus_perf"] >= 0.5), 1.0, 0.0)
    pts["oracle"] = (orc_cost.mean(), orc_perf.mean())
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", nargs="+",
                    default=["route_decisions.jsonl", "route_decisions_v2.jsonl",
                             "route_decisions_gepa.jsonl"])
    ap.add_argument("--labels", nargs="+",
                    default=["router(balanced, hand-written)",
                             "router(GEPA seed = cost-sensitive baseline)",
                             "router(GEPA-optimized)"])
    ap.add_argument("--out", default="summary.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 6))
    base_drawn = False
    for path, lab in zip(args.decisions, args.labels):
        rows = [json.loads(l) for l in open(path)]
        cost = np.mean([r["serving_cost"] for r in rows])
        acc = np.mean([r["perf"] for r in rows])
        pct_opus = np.mean([r["choice"] == "opus" for r in rows])
        ax.scatter([cost], [acc], s=160, marker="*", zorder=5,
                   label=f"{lab}  (%opus={pct_opus:.0%})")
        if not base_drawn:
            for name, (c, p) in baselines(rows).items():
                m = {"oracle": "D", "all-haiku": "v", "all-opus": "^",
                     "memory-prior": "s"}[name]
                ax.scatter([c], [p], s=90, marker=m, label=name)
            base_drawn = True

    ax.set_xlabel("serving cost ($ / query)  — router cost tracked separately")
    ax.set_ylabel("accuracy")
    ax.set_title("Training-free memory router: cost vs accuracy")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
