#!/usr/bin/env python3
"""Shared constants and Bedrock helpers for the training-free memory router.

The idea (no fine-tuning anywhere): we keep a MEMORY of past (query, model)
outcomes — did haiku/opus get it right, what did each cost, how many tokens —
and at route time we retrieve the most similar past problems and let a cheap
model (haiku) read those outcomes and decide which model should answer.

Two cost ledgers are kept strictly separate:
  * SERVING cost  — what the chosen answer model would cost (from recorded data)
  * ROUTER  cost  — what the router itself spends (Titan embedding + the haiku
                    decision call). This is tracked but NOT added to serving.
"""
import json
import time

import boto3
from botocore.config import Config

REGION = "us-west-2"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
OPUS = "us.anthropic.claude-opus-4-8"
TITAN = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 256

# answer-model cost (same rates as the training-based repo; cost linear in tokens)
OUT_RATE = {"haiku": 5e-6, "opus": 25e-6}   # $ / output token
IN_RATE = {"haiku": 1e-6, "opus": 5e-6}     # $ / input token

# router-side cost rates
HAIKU_OUT_RATE = 5e-6                        # router decision is a haiku call
HAIKU_IN_RATE = 1e-6
OPUS_OUT_RATE = 25e-6                         # GEPA reflector is an opus call
OPUS_IN_RATE = 5e-6
TITAN_RATE = 0.02 / 1_000_000                # $0.02 / 1M input tokens (Titan v2)

_br = boto3.client(
    "bedrock-runtime", region_name=REGION,
    config=Config(read_timeout=300, max_pool_connections=128,
                  retries={"max_attempts": 10, "mode": "adaptive"}),
)


def embed(text, dim=EMBED_DIM):
    """Return (vector, input_tokens) from Titan v2. input_tokens drives router cost."""
    for attempt in range(6):
        try:
            r = _br.invoke_model(
                modelId=TITAN,
                body=json.dumps({"inputText": text, "dimensions": dim, "normalize": True}),
            )
            body = json.loads(r["body"].read())
            return body["embedding"], body.get("inputTextTokenCount", 0)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def haiku_decide(prompt, max_tokens=4, prefill="The better model is"):
    """Call haiku for a routing decision. Returns (text, in_tokens, out_tokens).

    We prefill the assistant turn so the model emits the model name immediately
    instead of starting a chain of reasoning (which would be cut off and waste
    tokens). The returned text is the continuation after the prefill.
    """
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    if prefill:
        messages.append({"role": "assistant", "content": [{"text": prefill}]})
    for attempt in range(6):
        try:
            r = _br.converse(
                modelId=HAIKU,
                messages=messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
            )
            txt = r["output"]["message"]["content"][0]["text"]
            u = r["usage"]
            return txt, u["inputTokens"], u["outputTokens"]
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def opus_reflect(prompt, max_tokens=600):
    """Call OPUS as the GEPA reflector. Returns (text, in_tokens, out_tokens)."""
    for attempt in range(6):
        try:
            r = _br.converse(
                modelId=OPUS,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens},
            )
            txt = r["output"]["message"]["content"][0]["text"]
            u = r["usage"]
            return txt, u["inputTokens"], u["outputTokens"]
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
