#!/usr/bin/env python3
"""Draw a clean GEPA cost/accuracy Pareto frontier from gepa_pareto.json.

Distinguishes the true Pareto-optimal lambda points (solid, connected) from
dominated ones (hollow), and overlays the seed + all-haiku/all-opus/oracle
references. Each point is annotated with its lambda and %opus.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def is_dominated(p, pts):
    """p=(cost,acc) dominated if some q is <= cost and >= acc with one strict."""
    for q in pts:
        if q is p:
            continue
        if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
            return True
    return False


def main():
    d = json.load(open("gepa_pareto.json"))
    res = sorted(d["results"], key=lambda x: x["lambda"])
    pts = [(r["test"]["serving_cost"], r["test"]["acc"]) for r in res]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Pareto-optimal subset (minimize cost, maximize accuracy)
    front = [r for r, p in zip(res, pts) if not is_dominated(p, pts)]
    front = sorted(front, key=lambda r: r["test"]["serving_cost"])
    fx = [r["test"]["serving_cost"] for r in front]
    fy = [r["test"]["acc"] for r in front]
    ax.plot(fx, fy, "-", color="crimson", lw=2, zorder=2,
            label="GEPA Pareto frontier")

    for r, p in zip(res, pts):
        dom = is_dominated(p, pts)
        ax.scatter(p[0], p[1], s=180, marker="*", zorder=5,
                   color="crimson",
                   facecolors="none" if dom else "crimson",
                   linewidths=1.6)
        note = f"λ={r['lambda']:g}\n{r['test']['pct_opus']:.0%} opus"
        ax.annotate(note, p, textcoords="offset points", xytext=(8, -4),
                    fontsize=8, color="crimson")

    # seed (GEPA starting point) and recorded baselines
    st = d["seed_test"]
    ax.scatter(st["serving_cost"], st["acc"], s=150, marker="X",
               color="darkorange", zorder=4,
               label=f"GEPA seed (cost-sensitive baseline, {st['pct_opus']:.0%} opus)")
    b = d["baselines"]
    for name, m, col in [("all-haiku", "v", "tab:blue"),
                         ("all-opus", "^", "tab:green"),
                         ("oracle", "D", "purple")]:
        c, p = b[name]
        ax.scatter([c], [p], s=110, marker=m, color=col, zorder=4, label=name)

    ax.annotate("cheaper →\n(lower cost)", (0.0042, 0.792), fontsize=8,
                color="gray", ha="center")
    ax.set_xlabel("serving cost ($ / query)   —   GEPA optimization & router cost tracked separately")
    ax.set_ylabel("accuracy")
    ax.set_title("GEPA cost/accuracy Pareto frontier via λ sweep (held-out 600-query test)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("gepa_pareto_clean.png", dpi=130)
    print("saved gepa_pareto_clean.png")
    print(f"Pareto-optimal lambdas: {[r['lambda'] for r in front]}")
    print(f"dominated lambdas:      "
          f"{[r['lambda'] for r,p in zip(res,pts) if is_dominated(p,pts)]}")


if __name__ == "__main__":
    main()
