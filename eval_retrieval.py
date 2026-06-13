#!/usr/bin/env python3
"""Compare retrieval quality: Titan-256 vs Qwen3-Embedding-0.6B (1024).

Before paying for a full routing eval, check whether swapping the embedding
actually retrieves MORE USEFUL neighbors. A neighbor is useful if its recorded
labels predict the query's own labels. We measure, over the 600 test queries
(kNN into the 2400-query memory):

  * neighbor-predicts-haiku AUC : do neighbors' haiku_perf rank queries by their
    own haiku_perf?  (and same for opus, and for the routing margin opus-haiku)
  * label agreement@k           : fraction of neighbors sharing the query's
    "needs-opus" bit (haiku wrong & opus right)

Higher = the embedding groups problems whose models behave alike, which is
exactly what the router's memory needs. Titan vecs come from memory.jsonl;
Qwen vecs from mem_qwen.jsonl / test_qwen.jsonl (emb_doc and emb_query).
"""
import json

import numpy as np

TEST = "../llmrouter/router_multihead_test.jsonl"


def auc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[label == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_memory_labels():
    mem = [json.loads(l) for l in open("memory.jsonl")]
    return mem, {m["id"]: m for m in mem}


def knn_predict(Mmem, Mtest, mem, test, k):
    """For each test query, average neighbor labels -> predicted scores."""
    sims = Mtest @ Mmem.T                       # [Ntest, Nmem]
    hp = np.array([m["haiku_perf"] for m in mem])
    op = np.array([m["opus_perf"] for m in mem])
    needs = ((hp < 0.5) & (op >= 0.5)).astype(float)
    pred_h, pred_o, pred_needs = [], [], []
    for i in range(len(test)):
        idx = np.argpartition(-sims[i], k)[:k]
        pred_h.append(hp[idx].mean())
        pred_o.append(op[idx].mean())
        pred_needs.append(needs[idx].mean())
    return np.array(pred_h), np.array(pred_o), np.array(pred_needs)


def report(name, Mmem, Mtest, mem, test, k=8):
    th = np.array([t["haiku_perf"] for t in test])
    to = np.array([t["opus_perf"] for t in test])
    tneeds = ((th < 0.5) & (to >= 0.5)).astype(int)
    ph, po, pneeds = knn_predict(Mmem, Mtest, mem, test, k)
    print(f"\n=== {name} (k={k}) ===")
    print(f"  AUC neighbor->haiku_correct : {auc(ph, th.astype(int)):.4f}")
    print(f"  AUC neighbor->opus_correct  : {auc(po, to.astype(int)):.4f}")
    print(f"  AUC neighbor->needs_opus    : {auc(pneeds, tneeds):.4f}")
    print(f"  corr(pred_margin, true gain): "
          f"{np.corrcoef(po - ph, to - th)[0,1]:.4f}")
    return auc(pneeds, tneeds)


def main():
    mem, _ = load_memory_labels()
    test = [json.loads(l) for l in open(TEST)]

    # Titan-256 (memory vecs in memory.jsonl, test vecs in test_titan.jsonl)
    Titan_mem = np.array([m["embedding"] for m in mem], dtype=np.float32)
    ttitan = {json.loads(l)["id"]: json.loads(l) for l in open("test_titan.jsonl")}
    Titan_test = np.array([ttitan[t["id"]]["embedding"] for t in test], dtype=np.float32)

    # Qwen3-Embedding-0.6B (1024): doc (no instruction) and query (instruction)
    qmem = {json.loads(l)["id"]: json.loads(l) for l in open("mem_qwen.jsonl")}
    qtest = {json.loads(l)["id"]: json.loads(l) for l in open("test_qwen.jsonl")}
    Qmem_doc = np.array([qmem[m["id"]]["emb_doc"] for m in mem], dtype=np.float32)
    Qtest_doc = np.array([qtest[t["id"]]["emb_doc"] for t in test], dtype=np.float32)
    Qtest_q = np.array([qtest[t["id"]]["emb_query"] for t in test], dtype=np.float32)

    for k in (4, 8, 16):
        report("Titan-256", Titan_mem, Titan_test, mem, test, k)
        report("Qwen-1024 doc->doc", Qmem_doc, Qtest_doc, mem, test, k)
        report("Qwen-1024 doc<-query(instruct)", Qmem_doc, Qtest_q, mem, test, k)


if __name__ == "__main__":
    main()
