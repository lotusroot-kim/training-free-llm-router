#!/usr/bin/env python3
"""Build the router MEMORY from past (query, model) outcomes.

Source: ../llmrouter/routing_train.jsonl — one row per (query, model) with
recorded performance / cost / tokens. We collapse it to one record per query
holding BOTH models' outcomes, then attach a Titan embedding so we can later
retrieve the most similar past problems.

Output: memory.jsonl  (one line per past query)
  { id, query, task_name, difficulty,
    haiku_perf, opus_perf, haiku_cost, opus_cost, haiku_out, opus_out,
    embedding: [float, ...] }

Embedding tokens are summed and reported — this is a one-time build cost, kept
separate from any serving cost.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import embed, EMBED_DIM

SRC = "../llmrouter/routing_train.jsonl"


def load_pairs(src):
    """id -> merged record with haiku & opus outcomes."""
    rec = {}
    for line in open(src):
        d = json.loads(line)
        m = d["model_name"]
        if m not in ("claude-haiku-4-5", "claude-opus-4-8"):
            continue  # memory routes between haiku and opus only
        key = "haiku" if "haiku" in m else "opus"
        e = rec.setdefault(d["id"], {
            "id": d["id"], "query": d["query"],
            "task_name": d.get("task_name"), "difficulty": d.get("difficulty"),
        })
        e[f"{key}_perf"] = d["performance"]
        e[f"{key}_cost"] = d["cost_usd"]
        e[f"{key}_out"] = d["output_tokens"]
        e[f"{key}_in"] = d["input_tokens"]
    # keep only queries that have BOTH models measured
    return [r for r in rec.values()
            if "haiku_perf" in r and "opus_perf" in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="memory.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dim", type=int, default=EMBED_DIM)
    args = ap.parse_args()

    pairs = load_pairs(args.src)
    print(f"loaded {len(pairs)} past queries with both haiku & opus outcomes")

    done = {}
    if os.path.exists(args.out):  # resume
        for line in open(args.out):
            d = json.loads(line)
            done[d["id"]] = d
        print(f"resume: {len(done)} already embedded")

    todo = [p for p in pairs if p["id"] not in done]
    embed_tokens = 0

    def work(p):
        v, tok = embed(p["query"], dim=args.dim)
        p = dict(p)
        p["embedding"] = v
        return p, tok

    results = list(done.values())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            p, tok = fut.result()
            embed_tokens += tok
            results.append(p)
            if i % 100 == 0:
                print(f"  embedded {i}/{len(todo)}")

    with open(args.out, "w") as f:
        for p in results:
            f.write(json.dumps(p) + "\n")

    from common import TITAN_RATE
    print(f"\nwrote {len(results)} memory records -> {args.out}")
    print(f"embedding build tokens: {embed_tokens}  "
          f"(~${embed_tokens * TITAN_RATE:.6f}, one-time, NOT serving cost)")


if __name__ == "__main__":
    main()
