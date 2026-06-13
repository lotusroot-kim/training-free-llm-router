#!/usr/bin/env python3
"""Training-free memory router.

route(query):
  1. embed the query with Titan                          (router cost)
  2. retrieve the k most similar past problems from memory (cosine kNN)
  3. show their haiku/opus success/failure/cost history to haiku
  4. haiku replies HAIKU or OPUS                          (router cost)

The router NEVER touches the answer model here; it only decides. Serving cost
(what the chosen model would cost to answer) is looked up from recorded data in
eval — and the router's own cost is accumulated separately in `router_cost`.
"""
import json
import re
import threading

import numpy as np

from common import (embed, haiku_decide, TITAN_RATE,
                    HAIKU_IN_RATE, HAIKU_OUT_RATE)

# The INSTRUCTION block is the only part GEPA evolves; the neighbor history,
# query, and answer-format scaffold around it are fixed. Keep this as the seed.
SEED_INSTRUCTION = (
    "You route a math query to HAIKU (cheap, ~5x cheaper) or OPUS "
    "(stronger but ~5x the price). DEFAULT TO HAIKU to save money. Only "
    "choose OPUS if the similar past problems show HAIKU clearly fails "
    "while OPUS succeeds. If HAIKU often succeeds on similar problems, "
    "choose HAIKU even if OPUS also succeeds — paying 5x for the same "
    "right answer is waste."
)

# Seed instruction for the SCORE prompt (continuous estimates). GEPA tunes this
# to sharpen the predictions so the threshold curve rises toward the oracle.
SEED_SCORE_INSTRUCTION = (
    "For this math query, estimate for BOTH models: the probability it "
    "answers CORRECTLY, and how many OUTPUT TOKENS it would generate "
    "(harder problems → longer solutions → more tokens → higher cost). "
    "Base both on how the two models did on the most similar past "
    "problems below (their correctness and token counts), weighting "
    "closer matches (higher sim) more."
)


class MemoryRouter:
    def __init__(self, memory_path="memory.jsonl", k=8, instruction=SEED_INSTRUCTION,
                 score_instruction=SEED_SCORE_INSTRUCTION):
        self.k = k
        self.instruction = instruction
        self.score_instruction = score_instruction
        self.show_neighbor_text = False    # H1 toggle: include neighbor query snippets
        self.mem = []
        vecs = []
        for line in open(memory_path):
            d = json.loads(line)
            self.mem.append(d)
            vecs.append(d["embedding"])
        self.M = np.array(vecs, dtype=np.float32)        # [N, dim], titan-normalized
        self.ids = [d["id"] for d in self.mem]
        # cost ledgers (router only — never mixed into serving cost)
        self._lock = threading.Lock()
        self.router_cost = 0.0
        self.router_embed_tokens = 0
        self.router_in_tokens = 0
        self.router_out_tokens = 0
        self.n_calls = 0

    def _knn(self, qvec, exclude_id=None):
        sims = self.M @ np.asarray(qvec, dtype=np.float32)   # cosine (both unit-norm)
        # leave-self-out: when routing a query that lives in memory (dev/train
        # eval), drop its own record so we don't retrieve the answer.
        kk = self.k + (1 if exclude_id is not None else 0)
        idx = np.argpartition(-sims, kk)[:kk]
        idx = idx[np.argsort(-sims[idx])]
        out = [(self.mem[i], float(sims[i])) for i in idx
               if self.ids[i] != exclude_id]
        return out[: self.k]

    def _prompt(self, query, neighbors):
        lines = []
        for nb, sim in neighbors:
            hp = "correct" if nb["haiku_perf"] >= 0.5 else "WRONG"
            op = "correct" if nb["opus_perf"] >= 0.5 else "WRONG"
            lines.append(
                f"- [{nb.get('task_name','?')}/{nb.get('difficulty','?')}] "
                f"haiku={hp} (${nb['haiku_cost']:.4f}), "
                f"opus={op} (${nb['opus_cost']:.4f})"
            )
        hist = "\n".join(lines)
        h_acc = np.mean([n["haiku_perf"] for n, _ in neighbors])
        o_acc = np.mean([n["opus_perf"] for n, _ in neighbors])
        return (
            f"{self.instruction}\n\n"
            f"On the {len(neighbors)} most similar past problems, HAIKU was "
            f"correct {h_acc*100:.0f}% of the time, OPUS {o_acc*100:.0f}%.\n"
            f"{hist}\n\n"
            f"New query:\n{query}\n\n"
            "Answer with exactly one word: HAIKU or OPUS."
        )

    def _score_prompt(self, query, neighbors):
        """Prompt asking haiku to estimate, for each model, P(correct) AND the
        output-token count it would spend.

        The token estimates let the router predict cost ITSELF (honest: no peeking
        at the answer model's recorded cost), so the cost-aware signal
        (p_o - p_h) / predicted_Δcost can be built at route time. Neighbor output
        tokens are shown as evidence to anchor the estimate.
        """
        lines = []
        for nb, sim in neighbors:
            hp = "right" if nb["haiku_perf"] >= 0.5 else "WRONG"
            op = "right" if nb["opus_perf"] >= 0.5 else "WRONG"
            snippet = ""
            if self.show_neighbor_text:    # H1: let haiku re-judge similarity from text
                q = nb.get("query", "").replace("\n", " ")
                snippet = f' "{q[:140]}"' if q else ""
            lines.append(
                f"- sim={sim:.2f} [{nb.get('task_name','?')}/"
                f"{nb.get('difficulty','?')}] haiku={hp} ({int(nb.get('haiku_out',0))} tok), "
                f"opus={op} ({int(nb.get('opus_out',0))} tok){snippet}")
        hist = "\n".join(lines)
        return (
            f"{self.score_instruction}\n\n"
            f"Most similar past problems:\n{hist}\n\n"
            f"New query:\n{query}\n\n"
            "Reply in EXACTLY this format, nothing else:\n"
            "HAIKU=<correct%0-100>,<out_tokens> OPUS=<correct%0-100>,<out_tokens>"
        )

    # Three independent estimation LENSES for the single-call self-ensemble (v3).
    # Each asks haiku to judge the same query from a different angle; averaging
    # their estimates cuts the variance of one snap judgement — all in ONE call,
    # so serving stays at one haiku request per query.
    _ENSEMBLE_LENSES = [
        "LENS 1 — neighbor-vote: weigh the similar past problems above, trusting "
        "the closest matches most.",
        "LENS 2 — intrinsic-difficulty: ignore the neighbors and judge the new "
        "query on its own — how many non-obvious steps, and would a fast cheap "
        "model slip on any of them?",
        "LENS 3 — disagreement: focus on where HAIKU and OPUS DIVERGED on similar "
        "problems, and whether this query shares that failure mode.",
    ]

    def _score_prompt_ensemble(self, query, neighbors):
        lines = []
        for nb, sim in neighbors:
            hp = "right" if nb["haiku_perf"] >= 0.5 else "WRONG"
            op = "right" if nb["opus_perf"] >= 0.5 else "WRONG"
            lines.append(
                f"- sim={sim:.2f} [{nb.get('task_name','?')}/"
                f"{nb.get('difficulty','?')}] haiku={hp} ({int(nb.get('haiku_out',0))} tok), "
                f"opus={op} ({int(nb.get('opus_out',0))} tok)")
        hist = "\n".join(lines)
        lenses = "\n".join(self._ENSEMBLE_LENSES)
        return (
            f"{self.score_instruction}\n\n"
            f"Most similar past problems:\n{hist}\n\n"
            f"New query:\n{query}\n\n"
            "Estimate THREE TIMES independently, each from a different angle, then "
            "the angles will be averaged. Use these angles:\n"
            f"{lenses}\n\n"
            "Reply in EXACTLY this format, one line per lens, nothing else:\n"
            "L1 HAIKU=<correct%>,<out_tokens> OPUS=<correct%>,<out_tokens>\n"
            "L2 HAIKU=<correct%>,<out_tokens> OPUS=<correct%>,<out_tokens>\n"
            "L3 HAIKU=<correct%>,<out_tokens> OPUS=<correct%>,<out_tokens>"
        )

    def _score_prompt_cot(self, query, neighbors):
        """CoT scoring: let haiku REASON briefly about the query's own structure
        (steps, traps, where a fast model would slip) before emitting the scores.

        Same single serving call — only the output is longer. The reasoning is
        meant to sharpen p_o/p_h (which neighbor stats alone estimate weakly,
        corr≈0.17) by actually inspecting the problem.
        """
        lines = []
        for nb, sim in neighbors:
            hp = "right" if nb["haiku_perf"] >= 0.5 else "WRONG"
            op = "right" if nb["opus_perf"] >= 0.5 else "WRONG"
            lines.append(
                f"- sim={sim:.2f} [{nb.get('task_name','?')}/"
                f"{nb.get('difficulty','?')}] haiku={hp} ({int(nb.get('haiku_out',0))} tok), "
                f"opus={op} ({int(nb.get('opus_out',0))} tok)")
        hist = "\n".join(lines)
        return (
            f"{self.score_instruction}\n\n"
            f"Most similar past problems:\n{hist}\n\n"
            f"New query:\n{query}\n\n"
            "Think step by step in 2-4 short sentences: what does this problem "
            "actually require (how many steps, any traps or heavy computation), "
            "where would a FAST cheap model (HAIKU) most likely slip, and would "
            "the STRONGER model (OPUS) get it. Combine that with the neighbor "
            "outcomes. THEN, on a final line, output exactly:\n"
            "FINAL: HAIKU=<correct%0-100>,<out_tokens> OPUS=<correct%0-100>,<out_tokens>"
        )

    def score(self, query, qvec=None, q_embed_tokens=0, exclude_id=None, ensemble=1, cot=False):
        """Return (p_h, p_o, info) where info also has predicted out tokens/cost.

        ensemble=1: one estimate (original). ensemble>1: several estimates in one
        call, averaged (v3). cot=True: haiku reasons briefly before the scores
        (one call, longer output). Routing is deferred to a threshold signal.
        """
        if qvec is None:
            qvec, q_embed_tokens = embed(query)
        neighbors = self._knn(qvec, exclude_id=exclude_id)

        nb = [n for n, _ in neighbors]
        nh = float(np.mean([n["haiku_perf"] for n in nb]))
        no = float(np.mean([n["opus_perf"] for n in nb]))
        nh_out = float(np.mean([n.get("haiku_out", 0) for n in nb]))
        no_out = float(np.mean([n.get("opus_out", 0) for n in nb]))

        if cot:
            prompt = self._score_prompt_cot(query, neighbors)
            txt, it, ot = haiku_decide(prompt, max_tokens=300, prefill=None)
            full = txt
            # parse the LAST occurrence so the FINAL line wins over any in-reasoning numbers
            ms = list(re.finditer(
                r"HAIKU\s*=\s*(\d{1,3})\s*%?\s*,\s*(\d{1,6}).*?OPUS\s*=\s*(\d{1,3})\s*%?\s*,\s*(\d{1,6})",
                full, re.I | re.S))
            if ms:
                m = ms[-1]
                p_h = min(int(m.group(1)), 100) / 100.0
                out_h = float(m.group(2))
                p_o = min(int(m.group(3)), 100) / 100.0
                out_o = float(m.group(4))
            else:
                p_h, p_o, out_h, out_o = nh, no, nh_out, no_out
            n_lens = 1
        elif ensemble > 1:
            prompt = self._score_prompt_ensemble(query, neighbors)
            txt, it, ot = haiku_decide(prompt, max_tokens=120, prefill="L1 HAIKU=")
            full = "L1 HAIKU=" + txt
            # one (p_h,p_o,out_h,out_o) per parsed lens line, then average.
            # tolerate an optional "%" after the probability (haiku sometimes adds it).
            ph_l, po_l, oh_l, oo_l = [], [], [], []
            for m in re.finditer(
                    r"HAIKU\s*=\s*(\d{1,3})\s*%?\s*,\s*(\d{1,6}).*?OPUS\s*=\s*(\d{1,3})\s*%?\s*,\s*(\d{1,6})",
                    full, re.I):
                ph_l.append(min(int(m.group(1)), 100) / 100.0)
                oh_l.append(float(m.group(2)))
                po_l.append(min(int(m.group(3)), 100) / 100.0)
                oo_l.append(float(m.group(4)))
            p_h = float(np.mean(ph_l)) if ph_l else nh
            p_o = float(np.mean(po_l)) if po_l else no
            out_h = float(np.mean(oh_l)) if oh_l else nh_out
            out_o = float(np.mean(oo_l)) if oo_l else no_out
            n_lens = len(ph_l)
        else:
            prompt = self._score_prompt(query, neighbors)
            txt, it, ot = haiku_decide(prompt, max_tokens=24, prefill="HAIKU=")
            full = "HAIKU=" + txt
            mh = re.search(r"HAIKU\s*=\s*(\d{1,3})\s*%?\s*,\s*(\d{1,6})", full, re.I)
            mo = re.search(r"OPUS\s*=\s*(\d{1,3})\s*%?\s*,\s*(\d{1,6})", full, re.I)
            p_h = min(int(mh.group(1)), 100) / 100.0 if mh else nh
            p_o = min(int(mo.group(1)), 100) / 100.0 if mo else no
            out_h = float(mh.group(2)) if mh else nh_out
            out_o = float(mo.group(2)) if mo else no_out
            n_lens = 1

        with self._lock:
            self.router_embed_tokens += q_embed_tokens
            self.router_in_tokens += it
            self.router_out_tokens += ot
            self.router_cost += (q_embed_tokens * TITAN_RATE
                                 + it * HAIKU_IN_RATE + ot * HAIKU_OUT_RATE)
            self.n_calls += 1

        info = {"raw": txt.strip(), "neighbor_haiku_acc": nh,
                "neighbor_opus_acc": no, "pred_haiku_out": out_h,
                "pred_opus_out": out_o, "nb_haiku_out": nh_out, "nb_opus_out": no_out,
                "n_lens": n_lens, "top_sim": neighbors[0][1] if neighbors else 0.0}
        return p_h, p_o, info

    def route(self, query, qvec=None, q_embed_tokens=0, exclude_id=None):
        """Return ('haiku'|'opus', info). qvec/tokens can be precomputed to batch embedding."""
        if qvec is None:
            qvec, q_embed_tokens = embed(query)
        neighbors = self._knn(qvec, exclude_id=exclude_id)
        prompt = self._prompt(query, neighbors)
        txt, it, ot = haiku_decide(prompt)
        with self._lock:
            self.router_embed_tokens += q_embed_tokens
            self.router_in_tokens += it
            self.router_out_tokens += ot
            self.router_cost += (q_embed_tokens * TITAN_RATE
                                 + it * HAIKU_IN_RATE + ot * HAIKU_OUT_RATE)
            self.n_calls += 1

        choice = "opus" if re.search(r"opus", txt, re.I) else "haiku"
        # memory prior from neighbors (used for analysis / fallback)
        nb = [n for n, _ in neighbors]
        info = {
            "decision_text": txt.strip(),
            "neighbor_ids": [n["id"] for n in nb],
            "neighbor_haiku_acc": float(np.mean([n["haiku_perf"] for n in nb])),
            "neighbor_opus_acc": float(np.mean([n["opus_perf"] for n in nb])),
            "top_sim": neighbors[0][1] if neighbors else 0.0,
        }
        return choice, info

    def cost_report(self):
        return {
            "router_cost_usd": self.router_cost,
            "router_embed_tokens": self.router_embed_tokens,
            "router_in_tokens": self.router_in_tokens,
            "router_out_tokens": self.router_out_tokens,
            "n_calls": self.n_calls,
            "router_cost_per_call": self.router_cost / max(self.n_calls, 1),
        }
