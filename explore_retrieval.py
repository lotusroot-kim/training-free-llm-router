#!/usr/bin/env python3
"""FREE retrieval exploration (no LLM calls) to find configs worth paying to test.

The expensive routing eval needs a haiku call per query. But we can cheaply
PROXY routing quality straight from the neighbors' recorded labels: build the
same honest cost-aware threshold curve the real router would, except using the
NEIGHBOR-AVERAGE estimates (memory-prior) instead of haiku's estimates. If a
retrieval config gives a memory-prior that already sits higher above the mix
line, haiku has a sharper substrate to work with — and GEPA can amplify it.

We sweep, all from cached embeddings:
  * embedding: Titan-256, Qwen-1024 (doc & query), Qwen truncated dims
  * k (neighbors)
  * hybrid: alpha*Titan_sim + (1-alpha)*Qwen_sim fusion
  * distance weighting of neighbors (uniform vs similarity-weighted)

Metric = mean accuracy above the haiku<->opus interpolation line, using the
neighbor-predicted marginal / neighbor-predicted Δcost (everything honest, from
memory tokens/cost, leave-self-out not needed since test∩memory=∅).
"""
import json
import numpy as np

TEST = "../llmrouter/router_multihead_test.jsonl"
OUT_RATE = {"h": 5e-6, "o": 25e-6}
IN_RATE = {"h": 1e-6, "o": 5e-6}


def load_test():
    return [json.loads(l) for l in open(TEST)]


def mem_arrays():
    mem = [json.loads(l) for l in open("memory.jsonl")]          # Titan + labels/tokens
    qm = {json.loads(l)["id"]: json.loads(l) for l in open("mem_qwen.jsonl")}
    hp = np.array([m["haiku_perf"] for m in mem], float)
    op = np.array([m["opus_perf"] for m in mem], float)
    ho = np.array([m.get("haiku_out", 0) for m in mem], float)
    oo = np.array([m.get("opus_out", 0) for m in mem], float)
    Titan = np.array([m["embedding"] for m in mem], np.float32)
    Qdoc = np.array([qm[m["id"]]["emb_doc"] for m in mem], np.float32)
    return mem, hp, op, ho, oo, Titan, Qdoc


def test_arrays(test):
    tt = {json.loads(l)["id"]: json.loads(l) for l in open("test_titan.jsonl")}
    qt = {json.loads(l)["id"]: json.loads(l) for l in open("test_qwen.jsonl")}
    Titan = np.array([tt[t["id"]]["embedding"] for t in test], np.float32)
    Qdoc = np.array([qt[t["id"]]["emb_doc"] for t in test], np.float32)
    Qq = np.array([qt[t["id"]]["emb_query"] for t in test], np.float32)
    th = np.array([t["haiku_perf"] for t in test], float)
    to = np.array([t["opus_perf"] for t in test], float)
    thc = np.array([t["haiku_cost"] for t in test], float)
    toc = np.array([t["opus_cost"] for t in test], float)
    ith = np.array([t.get("input_tokens_h", 0) for t in test], float)
    ito = np.array([t.get("input_tokens_o", 0) for t in test], float)
    return Titan, Qdoc, Qq, th, to, thc, toc, ith, ito


def above_line(signal, th, to, thc, toc):
    """Honest cost-aware curve area above the haiku<->opus interpolation line."""
    n = len(signal)
    Ha, Hc = th.mean(), thc.mean()
    Oa, Oc = to.mean(), toc.mean()
    idx = np.argsort(-signal)
    to_o = np.zeros(n, bool)
    pts = [(Hc, Ha)]
    for j in idx:
        to_o[j] = True
        pts.append((np.where(to_o, toc, thc).mean(), np.where(to_o, to, th).mean()))
    c = np.array(sorted(pts))
    grid = np.linspace(Hc, Oc, 40)
    cc = c[c[:, 0].argsort()]
    cur = np.interp(grid, cc[:, 0], cc[:, 1])
    line = Ha + (grid - Hc) * (Oa - Ha) / (Oc - Hc)
    return float((cur - line).mean())


def knn_signal(Smem_test, hp, op, ho, oo, k, weighted, ith, ito):
    """Neighbor-predicted cost-aware signal for each test query.

    Smem_test: [Ntest, Nmem] similarity. Returns (p_o-p_h)/pred_Δcost using
    neighbor-average perf and neighbor-average output tokens (router's substrate).
    """
    ntest = Smem_test.shape[0]
    ph = np.empty(ntest); po = np.empty(ntest)
    pho = np.empty(ntest); poo = np.empty(ntest)
    for i in range(ntest):
        sims = Smem_test[i]
        idx = np.argpartition(-sims, k)[:k]
        w = sims[idx] if weighted else np.ones(k)
        w = np.clip(w, 1e-6, None); w = w / w.sum()
        ph[i] = (w * hp[idx]).sum(); po[i] = (w * op[idx]).sum()
        pho[i] = (w * ho[idx]).sum(); poo[i] = (w * oo[idx]).sum()
    pred_hc = OUT_RATE["h"] * pho + IN_RATE["h"] * ith
    pred_oc = OUT_RATE["o"] * poo + IN_RATE["o"] * ito
    return (po - ph) / np.maximum(pred_oc - pred_hc, 1e-6)


def main():
    test = load_test()
    mem, hp, op, ho, oo, Titan_m, Qdoc_m = mem_arrays()
    Titan_t, Qdoc_t, Qq_t, th, to, thc, toc, ith, ito = test_arrays(test)

    def sim(A_t, A_m):
        return A_t @ A_m.T

    print(f"{'config':<42}{'k':>3}{'wgt':>5}{'above_line':>12}")
    results = []

    def run(name, Smt, k, weighted):
        sig = knn_signal(Smt, hp, op, ho, oo, k, weighted, ith, ito)
        a = above_line(sig, th, to, thc, toc)
        results.append((a, name, k, weighted))
        print(f"{name:<42}{k:>3}{str(weighted):>5}{a:>+12.4f}")

    S_titan = sim(Titan_t, Titan_m)
    S_qdoc = sim(Qdoc_t, Qdoc_m)
    S_qq = sim(Qq_t, Qdoc_m)          # query-instruct vs doc memory (asymmetric)

    for k in (4, 8, 12, 16, 24, 32):
        run("Titan-256", S_titan, k, False)
        run("Qwen doc->doc", S_qdoc, k, False)
        run("Qwen query->doc", S_qq, k, False)
    # similarity-weighted neighbors at the best-ish k
    for k in (8, 16):
        run("Titan-256 [w]", S_titan, k, True)
        run("Qwen query->doc [w]", S_qq, k, True)
    # hybrid fusion of Titan + Qwen similarity (rank-free, sim-add)
    for alpha in (0.25, 0.5, 0.75):
        Sh = alpha * S_titan + (1 - alpha) * S_qq
        for k in (8, 16):
            run(f"hybrid a={alpha} (Titan+Qwen)", Sh, k, False)
    # truncated Qwen dims (cheaper retrieval) — does 512/256 hold up?
    for d in (256, 512):
        Qm = Qdoc_m[:, :d]; Qm = Qm / np.linalg.norm(Qm, axis=1, keepdims=True)
        Qt = Qq_t[:, :d]; Qt = Qt / np.linalg.norm(Qt, axis=1, keepdims=True)
        run(f"Qwen query->doc dim={d}", sim(Qt, Qm), 16, False)

    print("\n=== TOP 12 configs by above_line (FREE proxy; memory-prior substrate) ===")
    for a, name, k, w in sorted(results, reverse=True)[:12]:
        print(f"  {a:+.4f}  {name}  k={k} wgt={w}")
    print("\nNote: this is the memory-PRIOR proxy (no haiku). Real routing adds "
          "haiku's read of the query on top; configs that lift this proxy give "
          "haiku+GEPA a better substrate. Pay to verify only the top few.")


if __name__ == "__main__":
    main()
