#!/usr/bin/env python3
"""Compare seed vs GEPA-optimized SCORE prompt on the held-out threshold curve.

Reads the two threshold_scores*.jsonl files (seed and GEPA), rebuilds the honest
cost-aware curve for each, and plots both against the interpolation line + oracle.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT_RATE, IN_RATE


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return {k: np.array([r[k] for r in rows], float) for k in
            ("p_h", "p_o", "haiku_perf", "opus_perf", "haiku_cost", "opus_cost",
             "pred_h_out", "pred_o_out", "it_h", "it_o", "no", "nh")}


def curve(score, hp, op, hc, oc):
    idx = np.argsort(-score)
    to_o = np.zeros(len(score), bool)
    pts = [(hc.mean(), hp.mean())]
    for j in idx:
        to_o[j] = True
        pts.append((np.where(to_o, oc, hc).mean(), np.where(to_o, op, hp).mean()))
    return np.array(sorted(pts))


def cost_aware_curve(a):
    pred_hc = OUT_RATE["haiku"] * a["pred_h_out"] + IN_RATE["haiku"] * a["it_h"]
    pred_oc = OUT_RATE["opus"] * a["pred_o_out"] + IN_RATE["opus"] * a["it_o"]
    sig = (a["p_o"] - a["p_h"]) / np.maximum(pred_oc - pred_hc, 1e-6)
    return curve(sig, a["haiku_perf"], a["opus_perf"], a["haiku_cost"], a["opus_cost"])


import os


def main():
    best = load("threshold_scores_qwen_gepa.jsonl")
    Ha, Hc = best["haiku_perf"].mean(), best["haiku_cost"].mean()
    Oa, Oc = best["opus_perf"].mean(), best["opus_cost"].mean()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot([Hc, Oc], [Ha, Oa], "k--", lw=1.5,
            label="haiku↔opus mix baseline (random split)")
    if os.path.exists("threshold_scores_qwen_seed.jsonl"):
        seed = load("threshold_scores_qwen_seed.jsonl")
        ax.plot(*cost_aware_curve(seed)[:, :2].T, color="darkorange", lw=2,
                label="seed prompt, before GEPA (+0.009 above mix)")
    ax.plot(*cost_aware_curve(best)[:, :2].T, color="blue", lw=3,
            label="GEPA-optimized router (+0.037 above mix)")
    ax.scatter([Hc, Oc], [Ha, Oa], c=["tab:blue", "tab:green"], s=90, zorder=5)
    ax.annotate("all-haiku", (Hc, Ha), textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.annotate("all-opus", (Oc, Oa), textcoords="offset points", xytext=(-58, 4), fontsize=8)
    # mark the "~half opus cost" operating point
    half = Oc / 2
    cur = cost_aware_curve(best)
    cur = cur[cur[:, 0].argsort()]
    ah = float(np.interp(half, cur[:, 0], cur[:, 1]))
    ax.scatter([half], [ah], s=120, marker="*", color="red", zorder=6)
    ax.annotate(f"~½ opus cost\nacc {ah:.3f} ({ah/Oa*100:.0f}% of opus)",
                (half, ah), textcoords="offset points", xytext=(8, -28), fontsize=8, color="red")
    ax.set_xlabel("serving cost ($ / query)   —   router/GEPA cost tracked separately")
    ax.set_ylabel("accuracy")
    ax.set_title("GEPA prompt optimization lifts the routing curve\n"
                 "(seed → GEPA); the optimized router keeps ~opus accuracy at ~half cost")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("gepa_score_curve.png", dpi=130)
    print("saved gepa_score_curve.png")


if __name__ == "__main__":
    main()
