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
