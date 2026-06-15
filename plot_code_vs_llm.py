#!/usr/bin/env python3
"""Compare three routers on the held-out cost/accuracy curve:
  - mix baseline (random haiku/opus split)
  - GEPA-evolved CODE policy   (0 LLM calls per query — just kNN + numpy)
  - GEPA-tuned LLM-judge        (1 haiku call per query)
Both routers are GEPA-optimized; the difference is whether an LLM runs at serve time.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = {"haiku": 5e-6, "opus": 25e-6}
IN = {"haiku": 1e-6, "opus": 5e-6}


def load(f):
    return [json.loads(l) for l in open(f)]


def curve(rows):
    a = {k: np.array([r[k] for r in rows], float) for k in
         ("p_h", "p_o", "pred_h_out", "pred_o_out", "it_h", "it_o",
          "haiku_perf", "opus_perf", "haiku_cost", "opus_cost")}
    ph = OUT["haiku"] * a["pred_h_out"] + IN["haiku"] * a["it_h"]
    po = OUT["opus"] * a["pred_o_out"] + IN["opus"] * a["it_o"]
    sig = (a["p_o"] - a["p_h"]) / np.maximum(po - ph, 1e-6)
    idx = np.argsort(-sig); n = len(sig); seen = np.zeros(n, bool)
    pts = [(a["haiku_cost"].mean(), a["haiku_perf"].mean())]
    for j in idx:
        seen[j] = True
        pts.append((np.where(seen, a["opus_cost"], a["haiku_cost"]).mean(),
                    np.where(seen, a["opus_perf"], a["haiku_perf"]).mean()))
    c = np.array(sorted(pts))
    best, fr = -1, []
    for cc, aa in c:
        if aa > best:
            fr.append((cc, aa)); best = aa
    return np.array(fr)


def main():
    seed = load("threshold_scores_code_seed.jsonl")
    code = load("threshold_scores_code.jsonl")
    llm = load("threshold_scores_qwen_gepa.jsonl")
    Hc, Ha = np.mean([r["haiku_cost"] for r in code]), np.mean([r["haiku_perf"] for r in code])
    Oc, Oa = np.mean([r["opus_cost"] for r in code]), np.mean([r["opus_perf"] for r in code])

    c_seed = curve(seed)
    c_code = curve(code)
    c_llm = curve(llm)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot([Hc, Oc], [Ha, Oa], "k--", lw=1.5, label="mix baseline (random split)")
    ax.plot(c_seed[:, 0], c_seed[:, 1], color="darkorange", lw=2.4, linestyle=(0, (4, 2)),
            label="plain KNN router (no GEPA)  —  0 LLM calls/query")
    ax.plot(c_code[:, 0], c_code[:, 1], color="green", lw=2.8,
            label="GEPA-evolved CODE policy  —  0 LLM calls/query")
    ax.plot(c_llm[:, 0], c_llm[:, 1], color="blue", lw=2.8,
            label="GEPA-tuned LLM-judge  —  1 haiku call/query")
    ax.scatter([Hc, Oc], [Ha, Oa], c=["tab:blue", "tab:green"], s=90, zorder=5)
    ax.annotate("all-haiku", (Hc, Ha), textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.annotate("all-opus", (Oc, Oa), textcoords="offset points", xytext=(-58, 4), fontsize=8)
    ax.set_xlabel("serving cost ($ / query)   —   router/GEPA cost tracked separately")
    ax.set_ylabel("accuracy")
    ax.set_title("Routing policy as evolved CODE (no serving LLM) vs LLM-judge\n"
                 "both GEPA-optimized; the code policy gets most of the gain for free at serve time")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("curve_code_vs_llm.png", dpi=130)
    print("saved curve_code_vs_llm.png")


if __name__ == "__main__":
    main()
