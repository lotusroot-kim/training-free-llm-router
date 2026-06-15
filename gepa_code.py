#!/usr/bin/env python3
"""GEPA that evolves the ROUTING POLICY as CODE (no LLM call at serving time).

The other router asks haiku (an LLM) to read the retrieved neighbors and output
each model's P(correct)+tokens — one LLM call per query. Here we replace that
LLM judge with a small PYTHON FUNCTION that maps neighbor statistics →
(p_haiku, p_opus, out_haiku, out_opus). GEPA evolves the *source code* of that
function: opus reads the function + its worst routing errors and rewrites the
code (reflective evolution of an algorithm, not a prompt).

Result: a training-free router whose policy is interpretable code and whose
SERVING cost is just kNN retrieval — ZERO LLM calls per query. GEPA (opus) is
only spent once, offline, to write the code.

Objective: same honest cost-aware "area above the haiku↔opus mix line".
Sandbox: evolved code runs with only numpy in scope, a timeout guard, and any
exception drops that candidate.
"""
import argparse
import json
import random
import re
import signal

import numpy as np

from common import opus_reflect, OPUS_IN_RATE, OPUS_OUT_RATE
from router import MemoryRouter   # reuse memory loading + kNN

OUT_RATE = {"haiku": 5e-6, "opus": 25e-6}
IN_RATE = {"haiku": 1e-6, "opus": 5e-6}

# The policy is a function route(neighbors) -> (p_h, p_o, out_h, out_o).
# `neighbors` is a list of dicts, each with:
#   sim, haiku_perf, opus_perf, haiku_out, opus_out  (from the retrieved memory)
SEED_CODE = '''
def route(neighbors):
    # neighbors: list of dicts {sim, haiku_perf, opus_perf, haiku_out, opus_out},
    # sorted by sim descending. Return (p_haiku, p_opus, out_haiku, out_opus).
    import numpy as np
    w = np.array([n["sim"] for n in neighbors], dtype=float)
    w = np.clip(w, 1e-6, None); w = w / w.sum()
    p_h = float((w * np.array([n["haiku_perf"] for n in neighbors])).sum())
    p_o = float((w * np.array([n["opus_perf"] for n in neighbors])).sum())
    out_h = float((w * np.array([n["haiku_out"] for n in neighbors])).sum())
    out_o = float((w * np.array([n["opus_out"] for n in neighbors])).sum())
    return p_h, p_o, out_h, out_o
'''.strip()


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def compile_policy(code):
    """Compile code string into a route() callable in a restricted namespace."""
    ns = {"np": np, "__builtins__": {"range": range, "len": len, "min": min,
          "max": max, "sum": sum, "abs": abs, "float": float, "int": int,
          "sorted": sorted, "enumerate": enumerate, "zip": zip, "__import__": __import__}}
    exec(code, ns)
    fn = ns.get("route")
    if not callable(fn):
        raise ValueError("no route() defined")
    return fn


def neighbor_view(router, qvec, exclude_id=None):
    nbrs = router._knn(qvec, exclude_id=exclude_id)
    return [{"sim": s, "haiku_perf": n["haiku_perf"], "opus_perf": n["opus_perf"],
             "haiku_out": n.get("haiku_out", 0), "opus_out": n.get("opus_out", 0)}
            for n, s in nbrs]


def above_line(p_h, p_o, oh, oo, hp, op, hc, oc, ith, ito):
    pred_hc = OUT_RATE["haiku"] * oh + IN_RATE["haiku"] * ith
    pred_oc = OUT_RATE["opus"] * oo + IN_RATE["opus"] * ito
    sig = (p_o - p_h) / np.maximum(pred_oc - pred_hc, 1e-6)
    n = len(sig)
    Ha, Hc, Oa, Oc = hp.mean(), hc.mean(), op.mean(), oc.mean()
    if Oc <= Hc:
        return 0.0
    idx = np.argsort(-sig); seen = np.zeros(n, bool); pts = [(Hc, Ha)]
    for j in idx:
        seen[j] = True
        pts.append((np.where(seen, oc, hc).mean(), np.where(seen, op, hp).mean()))
    c = np.array(sorted(pts)); cc = c[c[:, 0].argsort()]
    grid = np.linspace(Hc, Oc, 40)
    return float((np.interp(grid, cc[:, 0], cc[:, 1])
                  - (Ha + (grid - Hc) * (Oa - Ha) / (Oc - Hc))).mean())


class CodeEvaluator:
    def __init__(self, router, dev):
        self.router = router
        self.dev = dev
        self.views = [neighbor_view(router, d["embedding"], exclude_id=d["id"]) for d in dev]
        self.hp = np.array([d["haiku_perf"] for d in dev], float)
        self.op = np.array([d["opus_perf"] for d in dev], float)
        self.hc = np.array([d["haiku_cost"] for d in dev], float)
        self.oc = np.array([d["opus_cost"] for d in dev], float)
        self.ith = np.array([d.get("haiku_in", 0) for d in dev], float)
        self.ito = np.array([d.get("opus_in", 0) for d in dev], float)

    def evaluate(self, code):
        """Run the policy on every dev query; return (above_line, error_rows)."""
        try:
            fn = compile_policy(code)
        except Exception as e:
            return -1.0, [f"COMPILE ERROR: {e}"]
        ph = np.empty(len(self.dev)); po = np.empty(len(self.dev))
        oh = np.empty(len(self.dev)); oo = np.empty(len(self.dev))
        errs = []
        for i, view in enumerate(self.views):
            try:
                signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, 1.0)
                r = fn(view)
                signal.setitimer(signal.ITIMER_REAL, 0)
                if r is None or len(r) != 4:
                    return -1.0, [f"BAD RETURN: route() returned {r!r} (need 4 floats)"]
                ph[i], po[i], oh[i], oo[i] = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            except Exception as e:
                signal.setitimer(signal.ITIMER_REAL, 0)
                return -1.0, [f"RUNTIME ERROR on a query: {e}"]
        val = above_line(ph, po, oh, oo, self.hp, self.op, self.hc, self.oc, self.ith, self.ito)
        # collect a few worst routing decisions for reflection
        for i in np.argsort(-(np.abs(po - self.op) + np.abs(ph - self.hp)))[:12]:
            errs.append(
                f"query: pred p_h={ph[i]:.2f} p_o={po[i]:.2f} (true haiku={self.hp[i]:.0f} "
                f"opus={self.op[i]:.0f}); pred out_h={oh[i]:.0f} out_o={oo[i]:.0f} "
                f"(true cost h=${self.hc[i]:.4f} o=${self.oc[i]:.4f})")
        return val, errs


REFLECT = '''You are improving a PYTHON FUNCTION that routes a math query between HAIKU (cheap) and OPUS (strong). It receives `neighbors` — a list of the most similar past problems, each a dict with keys: sim, haiku_perf (0/1), opus_perf (0/1), haiku_out (tokens), opus_out (tokens) — sorted by sim descending. It must return (p_haiku, p_opus, out_haiku, out_opus): the estimated probability each model is correct and its output-token count.

A router escalates to OPUS in order of (p_opus - p_haiku) / predicted_extra_cost, where cost = output_tokens * rate. So the function must produce well-separated, calibrated probabilities and sensible token estimates from the neighbor stats alone (NO LLM is called at run time — this is pure code).

CURRENT FUNCTION:
```python
{code}
```

On a dev set this scored above_line={score:.4f} (higher is better; it measures how far the cost-aware routing curve sits above the haiku<->opus mix line). Worst-estimated cases:

{errs}

Diagnose why the estimates are weak (e.g. just averaging neighbors ignores how CLOSE they are, doesn't down-weight conflicting neighbors, doesn't use that opus usually beats haiku only on hard problems, token estimate ignores difficulty) and write an IMPROVED `def route(neighbors):` that estimates more sharply. You may use numpy as np. Rules:
- return exactly (p_haiku, p_opus, out_haiku, out_opus) as floats
- self-contained, deterministic, fast (no I/O, no network, no LLM)
- keep it robust (handle few/empty neighbors)
- keep it SHORT (under ~35 lines) and make sure EVERY path ends with `return (p_haiku, p_opus, out_haiku, out_opus)` — never fall off the end (that returns None)

Return ONLY a python code block with the new def route(neighbors).'''


def extract_code(txt):
    # prefer a fenced block; tolerate ```python / ``` / no fence
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", txt, re.S)
    code = m.group(1) if m else txt
    # strip any stray fences and keep from the first def route onward
    code = code.replace("```python", "").replace("```", "").strip()
    i = code.find("def route")
    if i < 0:
        return None
    return code[i:].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="memory_qwen.jsonl")
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gepa_code_policy.py")
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)
    router = MemoryRouter(args.memory, k=args.k)
    idx = list(range(len(router.mem))); rng.shuffle(idx)
    dev = [router.mem[i] for i in idx[: args.dev]]
    ev = CodeEvaluator(router, dev)
    print(f"dev={len(dev)} k={args.k}  (evolving ROUTING CODE; 0 LLM calls at serving)")

    reflect_cost = [0.0]; nref = [0]

    def mutate(code, score, errs):
        prompt = REFLECT.format(code=code, score=score,
                                errs="\n".join(f"- {e}" for e in errs[:12]))
        txt, it, ot = opus_reflect(prompt, max_tokens=1600)
        reflect_cost[0] += it * OPUS_IN_RATE + ot * OPUS_OUT_RATE; nref[0] += 1
        return extract_code(txt)

    sv, se = ev.evaluate(SEED_CODE)
    pool = [{"code": SEED_CODE, "score": sv, "errs": se, "gen": 0}]
    best = pool[0]
    print(f"[seed] above_line={sv:+.4f}")

    for it in range(1, args.iters + 1):
        # parent = sample from top half (keep diversity)
        top = sorted(pool, key=lambda c: -c["score"])[: max(2, len(pool) // 2)]
        for ch in range(args.children):
            parent = rng.choice(top)
            nc = mutate(parent["code"], parent["score"], parent["errs"])
            if not nc:
                continue
            s, e = ev.evaluate(nc)
            cand = {"code": nc, "score": s, "errs": e, "gen": it}
            pool.append(cand)
            tag = "  *NEW BEST*" if s > best["score"] else ""
            print(f"[iter {it}.{ch}] above_line={s:+.4f}{tag}")
            if s > best["score"]:
                best = cand

    print(f"\n=== GEPA-CODE DONE ===\nseed={sv:+.4f}  BEST={best['score']:+.4f} (gen {best['gen']})")
    print(f"opt $: {reflect_cost[0]:.4f} ({nref[0]} reflections, one-time)")
    print("\n--- best policy code ---\n" + best["code"])
    with open(args.out, "w") as f:
        f.write(best["code"] + "\n")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
