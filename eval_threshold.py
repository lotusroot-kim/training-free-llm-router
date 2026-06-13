#!/usr/bin/env python3
"""Threshold routing: score every test query once, then sweep a threshold on the
marginal signal (p_o - p_h) to trace a CONTINUOUS cost/accuracy curve.

This is the fix for "the router should beat the haiku<->opus interpolation
line". A binary HAIKU/OPUS decision gives one operating point; a continuous
score lets us order queries by how much they'd benefit from opus and escalate
only the top ones — exactly what the interpolation line cannot do.

Curve construction (uses RECORDED perf/cost, so the sweep itself is free):
  order queries by score desc; move them haiku->opus one at a time; each prefix
  is one (avg_cost, avg_accuracy) point. We compare the curve's area / its
  accuracy at matched cost against the all-haiku..all-opus straight line.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from router import MemoryRouter


def score_all(router, data, workers, ensemble=1, qvecs=None, cot=False):
    def one(r):
        qv = qvecs.get(r["id"]) if qvecs else None      # precomputed (e.g. Qwen) test vec
        p_h, p_o, info = router.score(r["query"], qvec=qv, ensemble=ensemble, cot=cot)
        return {"id": r["id"], "p_h": p_h, "p_o": p_o,
                "haiku_perf": r["haiku_perf"], "opus_perf": r["opus_perf"],
                "haiku_cost": r["haiku_cost"], "opus_cost": r["opus_cost"],
                # router's OWN cost prediction (honest: no peeking at recorded cost)
                "pred_h_out": info["pred_haiku_out"], "pred_o_out": info["pred_opus_out"],
                "it_h": r.get("input_tokens_h", 0), "it_o": r.get("input_tokens_o", 0),
                "nh": info["neighbor_haiku_acc"], "no": info["neighbor_opus_acc"]}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, data))


def curve(score, hp, op, hc, oc):
    """Sort by score desc, escalate haiku->opus one at a time. Returns sorted pts."""
    n = len(score)
    idx = np.argsort(-score)
    to_o = np.zeros(n, bool)
    pts = [(hc.mean(), hp.mean())]                      # all-haiku
    for j in idx:
        to_o[j] = True
        pts.append((np.where(to_o, oc, hc).mean(),
                    np.where(to_o, op, hp).mean()))      # one more on opus
    return np.array(sorted(pts))


def acc_at(c, budget):
    c = c[c[:, 0].argsort()]
    return float(np.interp(budget, c[:, 0], c[:, 1]))


def auc_above_line(c, hc, ha, oc, oa):
    """Mean (curve_acc - line_acc) over the cost range — >0 means beats interp."""
    cs = np.linspace(hc, oc, 50)
    line = ha + (cs - hc) * (oa - ha) / (oc - hc)
    cur = np.array([acc_at(c, b) for b in cs])
    return float((cur - line).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="../llmrouter/router_multihead_test.jsonl")
    ap.add_argument("--memory", default="memory.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score_instruction", default=None,
                    help="path to a score-prompt instruction (e.g. gepa_score_instruction.txt)")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="single-call self-ensemble: estimate N times per query and average (v3)")
    ap.add_argument("--cot", action="store_true",
                    help="chain-of-thought scoring: haiku reasons before emitting scores (1 call)")
    ap.add_argument("--test_qvecs", default=None,
                    help="jsonl of precomputed test embeddings {id, emb_doc/emb_query/embedding}; "
                         "use with a matching --memory (e.g. memory_qwen.jsonl) to swap retrieval")
    ap.add_argument("--qvec_field", default="emb_query",
                    help="which field in --test_qvecs holds the vector")
    ap.add_argument("--neighbor_text", action="store_true",
                    help="H1: include neighbor query snippets in the score prompt")
    ap.add_argument("--out", default="threshold_scores.jsonl")
    args = ap.parse_args()

    data = [json.loads(l) for l in open(args.test)]
    if args.limit:
        data = data[: args.limit]
    router = MemoryRouter(args.memory, k=args.k)
    router.show_neighbor_text = args.neighbor_text
    qvecs = None
    if args.test_qvecs:
        qvecs = {}
        for l in open(args.test_qvecs):
            d = json.loads(l)
            qvecs[d["id"]] = d.get(args.qvec_field) or d.get("embedding")
        print(f"using precomputed test embeddings from {args.test_qvecs} "
              f"(field={args.qvec_field}, memory={args.memory})")
    if args.score_instruction:
        router.score_instruction = open(args.score_instruction).read().strip()
        print(f"using score instruction from {args.score_instruction} "
              f"({len(router.score_instruction)} chars)")

    from common import OUT_RATE, IN_RATE
    rows = score_all(router, data, args.workers, ensemble=args.ensemble, qvecs=qvecs, cot=args.cot)
    if args.ensemble > 1:
        print(f"self-ensemble: {args.ensemble} lenses/query in ONE haiku call")
    if args.cot:
        print("CoT scoring: haiku reasons before scores (1 call, longer output)")
    a = {key: np.array([r[key] for r in rows], float) for key in
         ("p_h", "p_o", "haiku_perf", "opus_perf", "haiku_cost", "opus_cost",
          "pred_h_out", "pred_o_out", "it_h", "it_o", "nh", "no")}
    Ha, Hc = a["haiku_perf"].mean(), a["haiku_cost"].mean()
    Oa, Oc = a["opus_perf"].mean(), a["opus_cost"].mean()

    # router's OWN predicted cost (honest — built from its token estimates, never
    # from the recorded answer-model cost).
    pred_hc = OUT_RATE["haiku"] * a["pred_h_out"] + IN_RATE["haiku"] * a["it_h"]
    pred_oc = OUT_RATE["opus"] * a["pred_o_out"] + IN_RATE["opus"] * a["it_o"]

    # routing signals to compare
    marginal = a["p_o"] - a["p_h"]                       # haiku-judged marginal
    nb_marginal = a["no"] - a["nh"]                       # memory-prior marginal
    gain = a["opus_perf"] - a["haiku_perf"]              # oracle marginal
    # HONEST cost-aware: divide by the router's PREDICTED Δcost
    ca_pred = (a["p_o"] - a["p_h"]) / np.maximum(pred_oc - pred_hc, 1e-6)
    # LEAKY cost-aware (for reference only): divides by the recorded true Δcost
    ca_leak = (a["p_o"] - a["p_h"]) / np.maximum(a["opus_cost"] - a["haiku_cost"], 1e-6)

    c_marg = curve(marginal, a["haiku_perf"], a["opus_perf"], a["haiku_cost"], a["opus_cost"])
    c_nb = curve(nb_marginal, a["haiku_perf"], a["opus_perf"], a["haiku_cost"], a["opus_cost"])
    c_ca = curve(ca_pred, a["haiku_perf"], a["opus_perf"], a["haiku_cost"], a["opus_cost"])
    c_leak = curve(ca_leak, a["haiku_perf"], a["opus_perf"], a["haiku_cost"], a["opus_cost"])
    c_orc = curve(np.where(gain > 0, gain, -1), a["haiku_perf"], a["opus_perf"],
                  a["haiku_cost"], a["opus_cost"])

    print(f"test={len(rows)}  k={args.k}")
    print(f"all-haiku acc={Ha:.3f} ${Hc:.5f}   all-opus acc={Oa:.3f} ${Oc:.5f}")
    print(f"corr(p_h, haiku_perf)={np.corrcoef(a['p_h'], a['haiku_perf'])[0,1]:.3f}  "
          f"corr(p_o, opus_perf)={np.corrcoef(a['p_o'], a['opus_perf'])[0,1]:.3f}")
    print(f"corr(pred_Δcost, true_Δcost)="
          f"{np.corrcoef(pred_oc - pred_hc, a['opus_cost'] - a['haiku_cost'])[0,1]:.3f}")
    print("\nmean accuracy ABOVE the haiku<->opus interpolation line:")
    print(f"  haiku-judge p_o-p_h          : {auc_above_line(c_marg, Hc, Ha, Oc, Oa):+.4f}")
    print(f"  cost-aware / PREDICTED Δcost : {auc_above_line(c_ca, Hc, Ha, Oc, Oa):+.4f}  (honest)")
    print(f"  cost-aware / TRUE Δcost      : {auc_above_line(c_leak, Hc, Ha, Oc, Oa):+.4f}  (leaky, ref)")
    print(f"  memory-prior                 : {auc_above_line(c_nb, Hc, Ha, Oc, Oa):+.4f}")
    print(f"  oracle                       : {auc_above_line(c_orc, Hc, Ha, Oc, Oa):+.4f}")

    print("\naccuracy at fixed cost budget (honest cost-aware = /predicted Δcost):")
    print(f"{'budget':>8}{'line':>8}{'p_o-p_h':>9}{'/pred$':>8}{'mem-prior':>10}{'oracle':>8}")
    for b in [0.005, 0.006, 0.007, 0.008, 0.010]:
        if not (Hc <= b <= Oc):
            continue
        line = Ha + (b - Hc) * (Oa - Ha) / (Oc - Hc)
        print(f"{b:>8.4f}{line:>8.3f}{acc_at(c_marg,b):>9.3f}{acc_at(c_ca,b):>8.3f}"
              f"{acc_at(c_nb,b):>10.3f}{acc_at(c_orc,b):>8.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot([Hc, Oc], [Ha, Oa], "k--", lw=1.5, label="haiku↔opus interpolation (mix baseline)")
    ax.plot(c_nb[:, 0], c_nb[:, 1], color="gray", alpha=0.8, label="memory-prior (no haiku call)")
    ax.plot(c_marg[:, 0], c_marg[:, 1], color="crimson", lw=2, label="threshold: haiku-judge (p_o−p_h)")
    ax.plot(c_ca[:, 0], c_ca[:, 1], color="darkorange", lw=2,
            label="threshold: cost-aware / PREDICTED Δcost (honest)")
    ax.plot(c_leak[:, 0], c_leak[:, 1], color="darkorange", lw=1, ls="--", alpha=0.6,
            label="cost-aware / true Δcost (leaky, ref)")
    ax.plot(c_orc[:, 0], c_orc[:, 1], color="purple", lw=1.5, ls=":", label="oracle (cheapest correct)")
    ax.scatter([Hc, Oc], [Ha, Oa], c=["tab:blue", "tab:green"], s=90, zorder=5)
    ax.annotate("all-haiku", (Hc, Ha), textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.annotate("all-opus", (Oc, Oa), textcoords="offset points", xytext=(-60, 4), fontsize=8)
    ax.set_xlabel("serving cost ($ / query)   —   router cost tracked separately")
    ax.set_ylabel("accuracy")
    ax.set_title("Threshold routing: continuous cost/accuracy curve vs the mix baseline")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("threshold_curve.png", dpi=130)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez("threshold_curves.npz", c_marg=c_marg, c_ca=c_ca, c_leak=c_leak,
             c_nb=c_nb, c_orc=c_orc, Ha=Ha, Hc=Hc, Oa=Oa, Oc=Oc)
    print("\nsaved threshold_curve.png, threshold_scores.jsonl, threshold_curves.npz")


if __name__ == "__main__":
    main()
