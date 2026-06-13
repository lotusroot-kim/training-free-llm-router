#!/usr/bin/env python3
"""FREE post-hoc signal transforms on the champion's saved scores (no LLM calls).

The router already emitted p_h, p_o, predicted tokens for every test query
(threshold_scores_qwen_gepa.jsonl). The routing curve only depends on the ORDER
queries are escalated, i.e. on the sort key. We can try different keys built
from the SAME estimates for free and see which orders queries best.

Candidate keys (all honest — only use router outputs + known input tokens):
  base   : (p_o - p_h) / pred_Δcost                       [current champion]
  gain   : (p_o - p_h)                                     [marginal only]
  ratio  : p_o / (p_h + eps)                               [relative]
  logit  : (logit(p_o) - logit(p_h)) / pred_Δcost         [spread extremes]
  shrink : shrink p toward neighbor prior, then base       [denoise]
  power  : (p_o - p_h)**g / pred_Δcost                     [sharpen margin]
We report each key's mean accuracy above the haiku<->opus interpolation line.
"""
import json
import numpy as np

OUT_RATE = {"h": 5e-6, "o": 25e-6}
IN_RATE = {"h": 1e-6, "o": 5e-6}
SCORES = "threshold_scores_qwen_gepa.jsonl"


def above_line(signal, th, to, thc, toc):
    n = len(signal)
    Ha, Hc = th.mean(), thc.mean(); Oa, Oc = to.mean(), toc.mean()
    idx = np.argsort(-signal)
    seen = np.zeros(n, bool); pts = [(Hc, Ha)]
    for j in idx:
        seen[j] = True
        pts.append((np.where(seen, toc, thc).mean(), np.where(seen, to, th).mean()))
    c = np.array(sorted(pts)); cc = c[c[:, 0].argsort()]
    grid = np.linspace(Hc, Oc, 40)
    return float((np.interp(grid, cc[:, 0], cc[:, 1])
                  - (Ha + (grid - Hc) * (Oa - Ha) / (Oc - Hc))).mean())


def main():
    rows = [json.loads(l) for l in open(SCORES)]
    a = {k: np.array([r[k] for r in rows], float) for k in
         ("p_h", "p_o", "pred_h_out", "pred_o_out", "it_h", "it_o",
          "haiku_perf", "opus_perf", "haiku_cost", "opus_cost", "nh", "no")}
    pred_hc = OUT_RATE["h"] * a["pred_h_out"] + IN_RATE["h"] * a["it_h"]
    pred_oc = OUT_RATE["o"] * a["pred_o_out"] + IN_RATE["o"] * a["it_o"]
    dcost = np.maximum(pred_oc - pred_hc, 1e-6)
    th, to, thc, toc = a["haiku_perf"], a["opus_perf"], a["haiku_cost"], a["opus_cost"]
    eps = 1e-6
    def logit(p): return np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))

    keys = {
        "base (p_o-p_h)/Δcost": (a["p_o"] - a["p_h"]) / dcost,
        "gain p_o-p_h": a["p_o"] - a["p_h"],
        "ratio p_o/p_h": a["p_o"] / (a["p_h"] + eps),
        "logit-margin/Δcost": (logit(a["p_o"]) - logit(a["p_h"])) / dcost,
        "power g=1.5/Δcost": np.sign(a["p_o"]-a["p_h"])*np.abs(a["p_o"]-a["p_h"])**1.5 / dcost,
        "power g=0.5/Δcost": np.sign(a["p_o"]-a["p_h"])*np.abs(a["p_o"]-a["p_h"])**0.5 / dcost,
    }
    # shrink p toward neighbor prior by beta, then base
    for beta in (0.25, 0.5):
        ph = (1-beta)*a["p_h"] + beta*a["nh"]
        po = (1-beta)*a["p_o"] + beta*a["no"]
        keys[f"shrink b={beta}/Δcost"] = (po - ph) / dcost
    # blend router estimate with neighbor prior at the SIGNAL level
    for w in (0.25, 0.5, 0.75):
        s = w*((a["p_o"]-a["p_h"])/dcost) + (1-w)*((a["no"]-a["nh"])/dcost)
        keys[f"blend router{w}+prior"] = s

    res = sorted(((above_line(s, th, to, thc, toc), name) for name, s in keys.items()),
                 reverse=True)
    print(f"{'signal transform':<28}{'above_line':>12}")
    for v, name in res:
        mark = "  <- champion base" if name.startswith("base") else ""
        print(f"{name:<28}{v:>+12.4f}{mark}")


if __name__ == "__main__":
    main()
