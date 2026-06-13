# Overnight performance exploration — notes

Goal: push the honest cost-aware "above mix-line" metric past the current
champion (**Qwen retrieval + Qwen-tuned GEPA, k=8 → +0.0366**, oracle gap 0.014)
on the held-out 600-query test, while keeping serving at 1 embed + 1 haiku call.

Method: explore cheaply first (FREE memory-prior proxy, no LLM calls in
`explore_retrieval.py`), then pay (~$0.3 each) to verify only promising configs.

## ★ KEY DIAGNOSIS (do this first in the morning) ★

Measured the ceiling: keep the SAME marginal signal (p_o−p_h) but divide by the
TRUE Δcost instead of the predicted one:

| signal | above_line |
|---|---|
| (p_o−p_h) / **predicted** Δcost (champion) | +0.0365 |
| (p_o−p_h) / **true** Δcost (perfect-cost ceiling) | **+0.0493** |
| oracle | +0.0502 |

**→ ~0.013 of the remaining 0.014 oracle gap is purely COST-PREDICTION error.**
The marginal/accuracy signal is already near-optimal (corr 0.28 with true gain).
The one lever that matters now is **predicting output tokens / Δcost better**
(currently corr(pred Δcost, true Δcost) ≈ 0.15). Nail that and we essentially
hit oracle — with no extra serving call.

Concrete plan: a GEPA pass (or a small dev-fit regressor on neighbor token
stats) whose objective is **token-prediction correlation**, not above-line. Or
feed richer token evidence (neighbor token mean+spread per model) and re-tune.
Everything else (k, neighbor text, ensembling, signal transforms) was tested and
does NOT help — see below.

## Champion (start of night)
| config | above_line | oracle gap |
|---|---|---|
| Qwen retrieval + Qwen-tuned GEPA, k=8 | **+0.0366** | 0.014 |

## Experiments run

### 1. k sweep (FREE proxy said k=16 best; REAL routing disagreed)
- FREE memory-prior proxy: Qwen query→doc k=8 +0.0138 vs **k=16 +0.0203** (proxy loves bigger k — neighbor average is more stable).
- REAL routing (haiku reads the neighbor list): k=8 **+0.0366** vs k=16 **+0.0240**.
- **Lesson:** proxy ≠ real. With more neighbors the *average* stabilizes but the
  *prompt* gets noisier for haiku to read, and the Qwen-tuned prompt was evolved
  on a k=8 neighbor list. **k=8 stays champion.** Proxy is good for ranking
  embeddings, NOT for picking k.

### 2. H1 — neighbor query snippets in the prompt (PAID, $0.3)
- champion + neighbor text: **+0.0316** vs champion +0.0366. WORSE.
- Same root cause as k: the Qwen-tuned prompt was evolved on a text-free, k=8
  neighbor list. Adding text lengthens/noises the prompt. Would need a GEPA
  re-tune *with* text to judge fairly.

### 3. k=16 with real routing (PAID, $0.3)
- **+0.0240** vs champion +0.0366 (k=8). WORSE — see §1.

### 4. FREE post-hoc signal transforms (explore_signal.py, no LLM calls)
Re-sorting the SAME saved champion scores by different keys:
| transform | above_line |
|---|---|
| shrink p toward neighbor-prior by 0.25, then /Δcost | **+0.0374** |
| blend 0.75·router + 0.25·prior signal | **+0.0374** |
| base (p_o−p_h)/Δcost (champion) | +0.0365 |
| logit-margin /Δcost | +0.0361 |
| power g=0.5 /Δcost | +0.0361 |
| gain only (p_o−p_h) | +0.0004 |

- A SMALL free win (+0.0009): blending ~25% of the neighbor prior into the
  router's estimate denoises haiku's overconfidence. **But picking the shrink
  coefficient on the test set is overfitting** — must fit β on a dev split and
  confirm on test before claiming it. Framework left as `explore_signal.py`.

## Summary for the morning

**Champion is robust: Qwen retrieval + Qwen-tuned GEPA, k=8, no neighbor text
→ +0.0366 (oracle gap 0.014).** Everything that *changes the haiku input*
(bigger k, neighbor text) HURT, because the GEPA prompt is tightly co-adapted to
the k=8 / text-free neighbor format. The only positive lever found is a tiny
FREE post-hoc shrink-to-prior (+0.0009), pending honest dev-split validation.

### 5. Honest dev/holdout validation of shrink-to-prior (validate_shrink.py)
- Split the 600 saved scores DEV/HOLDOUT ×8 seeds; pick β on DEV, score HOLDOUT.
- **mean holdout Δ = −0.0002** (β=0 chosen on 2/8 splits; one split −0.0087).
- **VERDICT: the +0.0009 was overfit to the test set; shrink-to-prior is NOT a
  real win.** Champion stays at +0.0366. Good reminder to validate every tweak
  on held-out data.

**Highest-EV next steps (ranked):**
1. ~~Honest shrink-to-prior~~ — TESTED, overfit, no real gain (see §5).
2. **Co-tune retriever+prompt+format together**: the recurring lesson is that
   components are co-adapted. A single GEPA run that *also* sees neighbor text at
   k=12 (not k=8) might unlock what §1–2 couldn't in isolation. ~$9, higher risk.
3. **Better answer-cost prediction**: corr(pred Δcost, true Δcost)≈0.15 is the
   weakest link (oracle uses true cost → +0.050). A GEPA pass whose objective is
   *token-prediction correlation* (not just above-line) could sharpen the
   denominator. ~$3.
4. **Diversified retrieval (MMR)**: dedupe near-identical neighbors so k=8 carries
   more information; pairs naturally with a re-tune. Implemented stub idea only.

Do NOT repeat: bigger k alone, neighbor text alone, self-ensemble averaging —
all measured worse.

### 6. Can we improve cost prediction for FREE? (no — needs a GEPA pass)
Tried replacing/blending haiku's token estimate with neighbor token means
(honest, from memory):
| cost predictor | corr(Δcost) | above_line |
|---|---|---|
| haiku-estimated (champion) | **0.327** | **0.0365** |
| neighbor-token-mean k=8 | 0.241 | 0.0308 |
| blend 0.75·haiku + 0.25·neighbor | 0.336 | 0.0364 |

- haiku's token estimate is ALREADY better than a raw neighbor average (it also
  reads the query's own difficulty). Blending barely moves corr (0.327→0.336)
  and not above_line. **No free lunch here.**
- So the cost-prediction lever (the one that matters, §KEY DIAGNOSIS) genuinely
  needs a learning step: a GEPA pass whose objective is token/Δcost correlation,
  or feeding richer per-model token evidence (mean+spread) and re-tuning. ~$3,
  to be done with fresh budget — left for the morning.

### 7. Cost-prediction GEPA re-tune (A/B) — BOTH WORSE than champion
Warm-started from the best Qwen prompt, re-evolved the TOKEN-prediction rules.
- **A** objective=above_line: corr(Δcost) **0.327→0.496** (big!), but above_line
  **0.0365→0.0201**. WORSE.
- **B** objective=cost_corr (accuracy-guarded): corr(Δcost) 0.281, above_line
  **0.0080**. WORSE.

**Why the §KEY-DIAGNOSIS ceiling was misleading:** that ceiling held the marginal
signal FIXED and only swapped in true Δcost. But re-tuning the prompt for cost
ALSO changed the token *distribution* and the marginal estimates. Measured the
true-cost ceiling per prompt:
| prompt | marg_corr | above(pred$) | above(TRUE$ ceiling) |
|---|---|---|---|
| champion | 0.278 | **0.0365** | **0.0493** |
| A | 0.304 | 0.0201 | 0.0387 |
| B | 0.311 | 0.0080 | 0.0360 |

The re-tuned prompts emphasized "hard → 3k-20k opus tokens", which RAISED rank
correlation but distorted the Δcost *magnitude* used as the routing denominator,
and lowered the whole achievable ceiling. **Lesson: corr(Δcost) is the wrong
proxy — optimizing it does not optimize routing. The champion's lower-corr but
better-scaled cost estimate routes better.** Champion stays best at +0.0365.

**Revised takeaway:** the +0.013 "cost-prediction headroom" is real ONLY if you
could improve cost *without* perturbing the marginal/scale — which a single
shared prompt can't do cleanly. A truly separate, calibrated token regressor
(fit on dev, e.g. isotonic on neighbor-token features) is the honest way to test
that headroom, and it must not add a serving call. Left for later.

### 8. Reasoning (CoT) scoring — WORSE than champion
Let haiku reason 2-4 sentences about the query's structure (steps, traps, where
a cheap model slips) before emitting scores; same single serving call, longer
output (~10× router cost, still its own ledger).
| prompt | marg_corr | above(pred$) | above(TRUE$) |
|---|---|---|---|
| champion (snap scoring) | 0.278 | **0.0365** | 0.0493 |
| CoT scoring | 0.246 | **0.0138** | 0.0280 |

- corr(p_o,opus)↑ slightly (0.17→0.19) BUT marg_corr↓ (0.278→0.246), cost
  corr↓ (0.33→0.24), and the achievable ceiling collapsed (0.049→0.028).
- haiku tends to OVER-TRUST its own reasoning → scores pile up at extremes
  (p_o=1.00 a lot) → less discriminative, same failure shape as self-ensemble.
- Confound: the champion prompt was GEPA-tuned for SNAP scoring; bolting CoT on
  top is a format mismatch (same lesson as k16 / neighbor-text / A-B). A fair
  test needs GEPA re-tuned FOR the CoT format (~$3-4) — not run yet.

## Pattern across ALL attempts (the real lesson)
Every change to the haiku INPUT/OUTPUT format (bigger k, neighbor text,
self-ensemble, cost-retune, CoT) underperformed the champion, because the
champion prompt is tightly co-adapted to its exact (Qwen-k8, snap-score) setup.
The only wins came from CO-OPTIMIZING the whole pipeline (Qwen retrieval + GEPA
re-tuned on it). **Isolated tweaks lose; joint re-tuning wins.** Any future idea
(CoT, new retriever, richer evidence) must be paired with a GEPA re-tune in the
SAME format to have a fair shot.

### 9. CoT + JOINT GEPA re-tune — the fair test, still FAILS
The honest "co-optimize the format" test the §8 lesson demanded: warm-start from
the champion Qwen prompt and run the full GEPA v2 (pareto-sample/merge/lineage/
resample) WITH cot=True, so the prompt is evolved FOR the CoT format on the Qwen
neighbor distribution.
- dev above_line: seed **+0.0003**, GEPA BEST **−0.0036** — GEPA could not find a
  CoT prompt that even matches the snap-scoring seed, let alone the champion's
  dev +0.0277. Every mutation went negative. Cost $13.85 (long CoT rollouts).
- **Conclusion: CoT is fundamentally ill-suited here, not just un-tuned.** Even
  joint optimization can't rescue it. Why: reasoning makes haiku CONFIDENT, so
  p_o/p_h pile at the extremes (p_o=1.00) and lose the cross-query RANKING that
  threshold routing depends on. Routing needs relative ordering, not per-item
  certainty — and snap scoring preserves ordering better than CoT.

**Final verdict on the whole search:** the champion (Qwen retrieval + Qwen-tuned
snap-score GEPA, k=8, +0.0366, oracle gap 0.014) is a robust local optimum. Five
distinct "improvement" ideas (bigger k, neighbor text, self-ensemble, cost-retune
A/B, CoT — even with joint re-tuning) ALL underperformed it. The architecture's
ceiling under one-haiku-call serving is ~+0.037; closing the last 0.014 to oracle
needs true per-query cost (unavailable at route time) or a second serving call.

## Files added tonight
- `explore_retrieval.py` — FREE retrieval-quality sweep (memory-prior proxy)
- `explore_signal.py`    — FREE post-hoc signal-transform sweep
- `validate_shrink.py`   — honest dev/holdout validation (caught the overfit)
- `router.py`            — added `show_neighbor_text` (H1 toggle, off by default)
- `eval_threshold.py`    — added `--neighbor_text` flag
All serving-call counts unchanged (1 embed + 1 haiku per query). Champion config
and files are untouched; nothing here regressed the committed result.
