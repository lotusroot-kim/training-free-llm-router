#!/usr/bin/env python3
"""Apply a CODE routing policy (def route(neighbors)) to the held-out TEST set and
emit threshold_scores_*.jsonl in the same format plot_code_vs_llm.py consumes.

Memory + neighbors use the Qwen embeddings (memory_qwen.jsonl, test_qwen emb_query),
exactly like the GEPA-evolved code curve. True labels (haiku/opus perf, cost, input
tokens) are taken from an existing scored file keyed by id, so seed vs GEPA differ
ONLY in the policy code — a clean apples-to-apples baseline.
"""
import argparse
import json

import numpy as np

from gepa_code import compile_policy, neighbor_view
from router import MemoryRouter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="file with def route(neighbors)")
    ap.add_argument("--memory", default="memory_qwen.jsonl")
    ap.add_argument("--test_qvecs", default="test_qwen.jsonl")
    ap.add_argument("--labels", default="threshold_scores_qwen_gepa.jsonl",
                    help="existing scored file to copy true labels from (by id)")
    ap.add_argument("--qvec_field", default="emb_query")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    router = MemoryRouter(args.memory, k=args.k)
    fn = compile_policy(open(args.policy).read())
    qvecs = {json.loads(l)["id"]: json.loads(l)[args.qvec_field]
             for l in open(args.test_qvecs)}
    labels = {json.loads(l)["id"]: json.loads(l) for l in open(args.labels)}

    n = 0
    with open(args.out, "w") as f:
        for tid, lab in labels.items():
            qv = np.asarray(qvecs[tid], dtype=np.float32)
            view = neighbor_view(router, qv, exclude_id=tid)
            p_h, p_o, out_h, out_o = fn(view)
            rec = {"id": tid, "p_h": float(p_h), "p_o": float(p_o),
                   "pred_h_out": float(out_h), "pred_o_out": float(out_o),
                   "it_h": lab["it_h"], "it_o": lab["it_o"],
                   "haiku_perf": lab["haiku_perf"], "opus_perf": lab["opus_perf"],
                   "haiku_cost": lab["haiku_cost"], "opus_cost": lab["opus_cost"]}
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
