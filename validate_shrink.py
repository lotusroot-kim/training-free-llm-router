#!/usr/bin/env python3
"""Honestly validate the shrink-to-prior signal tweak (no LLM calls).

The free signal sweep suggested blending ~25% neighbor-prior into haiku's
estimate helps (+0.0009). Picking that 0.25 on the test set is overfitting, so
here we: split the 600 saved champion scores into DEV/HOLDOUT (seeded), choose
the shrink coefficient β on DEV only, and report the HOLDOUT above-line for that
β vs the unshrunk champion. Repeated over several splits to see if it's real.
"""
import json
import numpy as np

OUT_RATE = {"h": 5e-6, "o": 25e-6}
IN_RATE = {"h": 1e-6, "o": 5e-6}
BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def above_line(sig, th, to, thc, toc):
    n = len(sig)
    Ha, Hc = th.mean(), thc.mean(); Oa, Oc = to.mean(), toc.mean()
    if Oc <= Hc:
        return 0.0
    idx = np.argsort(-sig); seen = np.zeros(n, bool); pts = [(Hc, Ha)]
    for j in idx:
        seen[j] = True
        pts.append((np.where(seen, toc, thc).mean(), np.where(seen, to, th).mean()))
    c = np.array(sorted(pts)); cc = c[c[:, 0].argsort()]
    grid = np.linspace(Hc, Oc, 40)
    return float((np.interp(grid, cc[:, 0], cc[:, 1])
                  - (Ha + (grid - Hc) * (Oa - Ha) / (Oc - Hc))).mean())


def signal(a, idx, beta):
    ph = (1 - beta) * a["p_h"][idx] + beta * a["nh"][idx]
    po = (1 - beta) * a["p_o"][idx] + beta * a["no"][idx]
    pred_hc = OUT_RATE["h"] * a["pred_h_out"][idx] + IN_RATE["h"] * a["it_h"][idx]
    pred_oc = OUT_RATE["o"] * a["pred_o_out"][idx] + IN_RATE["o"] * a["it_o"][idx]
    return (po - ph) / np.maximum(pred_oc - pred_hc, 1e-6)


def main():
    rows = [json.loads(l) for l in open("threshold_scores_qwen_gepa.jsonl")]
    a = {k: np.array([r[k] for r in rows], float) for k in
         ("p_h", "p_o", "pred_h_out", "pred_o_out", "it_h", "it_o",
          "haiku_perf", "opus_perf", "haiku_cost", "opus_cost", "nh", "no")}
    n = len(rows)

    def hl(idx, beta):
        return above_line(signal(a, idx, beta), a["haiku_perf"][idx], a["opus_perf"][idx],
                          a["haiku_cost"][idx], a["opus_cost"][idx])

    print(f"{'split':>6}{'β*(dev)':>9}{'holdout β=0':>13}{'holdout β*':>12}{'Δ':>9}")
    base_all, tuned_all = [], []
    for s in range(8):
        rng = np.random.RandomState(s)
        perm = rng.permutation(n)
        dev, hold = perm[: n // 2], perm[n // 2:]
        # choose β on dev
        bestb = max(BETAS, key=lambda b: hl(dev, b))
        h0 = hl(hold, 0.0)
        hb = hl(hold, bestb)
        base_all.append(h0); tuned_all.append(hb)
        print(f"{s:>6}{bestb:>9.1f}{h0:>13.4f}{hb:>12.4f}{hb-h0:>+9.4f}")
    print(f"\nmean holdout  β=0: {np.mean(base_all):+.4f}   "
          f"β*(dev-chosen): {np.mean(tuned_all):+.4f}   "
          f"mean Δ: {np.mean(np.array(tuned_all)-np.array(base_all)):+.4f}")
    print("If mean Δ > 0 consistently, shrink-to-prior is a real (if small) free win.")


if __name__ == "__main__":
    main()
