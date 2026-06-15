def route(neighbors):
    import numpy as np
    if not neighbors:
        return (0.5, 0.6, 500.0, 500.0)
    sim = np.clip(np.array([n["sim"] for n in neighbors], float), 1e-6, 1.0)
    # sharp distance-based weights: emphasize close neighbors
    w = np.exp((sim - sim.max()) / 0.15) * sim
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w**2)          # effective neighbor count
    hp = np.array([n["haiku_perf"] for n in neighbors], float)
    op = np.array([n["opus_perf"] for n in neighbors], float)
    ho = np.array([n["haiku_out"] for n in neighbors], float)
    oo = np.array([n["opus_out"] for n in neighbors], float)
    # Beta-style shrinkage toward priors; strength shrinks as n_eff grows
    k = 2.0
    ph_raw = (w * hp).sum(); po_raw = (w * op).sum()
    p_h = (ph_raw * n_eff + 0.55 * k) / (n_eff + k)
    p_o = (po_raw * n_eff + 0.80 * k) / (n_eff + k)
    # opus should not look worse than haiku for routing; enforce edge on hard cases
    p_o = max(p_o, p_h + (1.0 - p_h) * 0.25)
    p_h = min(p_h, 0.985); p_o = min(p_o, 0.99)
    # token estimates: similarity-weighted, blended with weighted median (robust)
    def est(x):
        idx = np.argsort(x); cw = np.cumsum(w[idx])
        med = x[idx][np.searchsorted(cw, 0.5).clip(0, len(x)-1)]
        return float(0.6 * (w * x).sum() + 0.4 * med)
    out_h = max(est(ho), 1.0); out_o = max(est(oo), 1.0)
    return (float(p_h), float(p_o), out_h, out_o)
