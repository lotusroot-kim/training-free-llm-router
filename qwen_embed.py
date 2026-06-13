#!/usr/bin/env python3
"""Embed memory + test queries with Qwen3-Embedding-0.6B on GPU.

Produces, for both files, two embedding variants per query:
  * doc  : no instruction (symmetric / document-style)
  * query: instruction-prefixed (asymmetric retrieval-query style)

We later try memory=doc with test=doc (symmetric) and test=query (asymmetric).
Output: <name>_qwen.jsonl with {id, emb_doc:[...], emb_query:[...]}.
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIM = 1024
MAXLEN = 1024
BATCH = 64
TASK = "Given a math word/competition problem, retrieve past math problems of similar topic and difficulty"


def last_token_pool(h, mask):
    left = (mask[:, -1].sum() == mask.shape[0])
    if left:
        return h[:, -1]
    lens = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), lens]


def main():
    inp, out = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(inp)]
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
    print(f"loaded {MODEL}; {len(rows)} queries from {inp}", flush=True)

    def encode(texts):
        out_vecs = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            bd = tok(chunk, padding=True, truncation=True, max_length=MAXLEN,
                     return_tensors="pt").to(model.device)
            with torch.no_grad():
                h = model(**bd).last_hidden_state
            e = last_token_pool(h, bd["attention_mask"])[:, :DIM]
            e = F.normalize(e, p=2, dim=1)
            out_vecs.append(e.float().cpu())
            if (i // BATCH) % 5 == 0:
                print(f"  {i+len(chunk)}/{len(texts)}", flush=True)
        return torch.cat(out_vecs).tolist()

    plain = [r["query"] for r in rows]
    instructed = [f"Instruct: {TASK}\nQuery:{r['query']}" for r in rows]
    emb_doc = encode(plain)
    emb_query = encode(instructed)

    with open(out, "w") as f:
        for r, ed, eq in zip(rows, emb_doc, emb_query):
            f.write(json.dumps({"id": r["id"], "emb_doc": ed, "emb_query": eq}) + "\n")
    print(f"wrote {out} (dim={DIM})", flush=True)


if __name__ == "__main__":
    main()
