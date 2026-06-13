#!/usr/bin/env python3
"""Evaluate the training-free memory router on the held-out test set.

For each test query the router (Titan retrieval + a real haiku decision call)
picks haiku or opus. We then score the choice using the test set's RECORDED
perf/cost for that model — no answer model is re-invoked, so serving cost is
exact and reproducible.

Reported separately, as requested:
  * SERVING: routed accuracy & avg serving cost (the chosen answer model's $)
  * ROUTER : the router's own $ (Titan embeddings + haiku decisions), NOT added
             into serving cost.

Baselines for context: all-haiku, all-opus, oracle (cheapest correct model),
and memory-prior (route by neighbor success rates, no haiku call).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import embed
from router import MemoryRouter


def load_test(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="../llmrouter/router_multihead_test.jsonl")
    ap.add_argument("--memory", default="memory.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all test queries")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--instruction", default=None,
                    help="path to an instruction file (e.g. gepa_best_instruction.txt); "
                         "default uses the seed instruction baked into MemoryRouter")
    ap.add_argument("--out", default="route_decisions.jsonl")
    args = ap.parse_args()

    data = load_test(args.test)
    if args.limit:
        data = data[: args.limit]
    router = MemoryRouter(args.memory, k=args.k)
    if args.instruction:
        router.instruction = open(args.instruction).read().strip()
        print(f"using instruction from {args.instruction} ({len(router.instruction)} chars)")

    def route_one(r):
        choice, info = router.route(r["query"])
        return {
            "id": r["id"], "task_name": r.get("task_name"),
            "choice": choice, "perf": r[f"{choice}_perf"],
            "serving_cost": r[f"{choice}_cost"],
            "haiku_perf": r["haiku_perf"], "opus_perf": r["opus_perf"],
            "haiku_cost": r["haiku_cost"], "opus_cost": r["opus_cost"],
            **info,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(route_one, data), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  routed {i}/{len(data)}  "
                      f"(acc={np.mean([x['perf'] for x in rows]):.3f}, "
                      f"%opus={np.mean([x['choice']=='opus' for x in rows]):.2f})")

    a = {k: np.array([r[k] for r in rows], float)
         for k in ("perf", "serving_cost", "haiku_perf", "opus_perf",
                   "haiku_cost", "opus_cost", "neighbor_haiku_acc",
                   "neighbor_opus_acc")}
    choice = np.array([r["choice"] for r in rows])
    n = len(rows)

    # ---- baselines (recorded perf/cost) ----
    Ha, Hc = a["haiku_perf"].mean(), a["haiku_cost"].mean()
    Oa, Oc = a["opus_perf"].mean(), a["opus_cost"].mean()
    # oracle: cheapest model that is correct (haiku if it's right, else opus)
    orc_perf = np.where(a["haiku_perf"] >= 0.5, 1.0,
                        np.where(a["opus_perf"] >= 0.5, 1.0, 0.0))
    orc_cost = np.where(a["haiku_perf"] >= 0.5, a["haiku_cost"], a["opus_cost"])
    # memory-prior: route opus when neighbors say opus helps (no haiku call)
    mp = a["neighbor_opus_acc"] > a["neighbor_haiku_acc"]
    mp_perf = np.where(mp, a["opus_perf"], a["haiku_perf"])
    mp_cost = np.where(mp, a["opus_cost"], a["haiku_cost"])

    rep = router.cost_report()
    routed_acc = a["perf"].mean()
    routed_cost = a["serving_cost"].mean()
    pct_opus = (choice == "opus").mean()

    print("\n================ RESULTS ================")
    print(f"test queries: {n}   k={args.k}")
    print(f"\n{'strategy':<16}{'accuracy':>10}{'serving$/q':>13}")
    print(f"{'all-haiku':<16}{Ha:>10.4f}{Hc:>13.5f}")
    print(f"{'all-opus':<16}{Oa:>10.4f}{Oc:>13.5f}")
    print(f"{'memory-prior':<16}{mp_perf.mean():>10.4f}{mp_cost.mean():>13.5f}")
    print(f"{'MEMORY ROUTER':<16}{routed_acc:>10.4f}{routed_cost:>13.5f}   "
          f"(%opus={pct_opus:.2f})")
    print(f"{'oracle':<16}{orc_perf.mean():>10.4f}{orc_cost.mean():>13.5f}")

    print("\n---------------- ROUTER COST (separate ledger) ----------------")
    print(f"router total $        : {rep['router_cost_usd']:.6f}")
    print(f"router $ / query      : {rep['router_cost_per_call']:.8f}")
    print(f"  titan embed tokens  : {rep['router_embed_tokens']}")
    print(f"  haiku in/out tokens : {rep['router_in_tokens']} / {rep['router_out_tokens']}")
    print(f"serving $ / query     : {routed_cost:.6f}   (router NOT included)")
    print(f"serving+router $ / q  : {routed_cost + rep['router_cost_per_call']:.6f}   "
          f"(for reference only)")

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open("router_cost.json", "w") as f:
        json.dump({"serving_acc": routed_acc, "serving_cost_per_q": routed_cost,
                   "pct_opus": pct_opus, **rep}, f, indent=2)
    print(f"\nsaved {args.out} and router_cost.json")


if __name__ == "__main__":
    main()
