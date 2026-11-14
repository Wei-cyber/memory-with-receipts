"""Embeddings: Voyage when a key is configured, a local hashing embedder otherwise.

Both produce unit vectors of config.EMBED_DIMS, so the Atlas vector index
definition is the same either way. Vectors from the two are not comparable -
re-seed after switching.
"""
import hashlib
import math
import re

import httpx

from . import config

_STOP = set(
    "a an and are as at be but by for from has have if in into is it its of on or "
    "not that the their then there these this to was were will with when than so "
    "do did does usually after before over".split()
)


def name() -> str:
    return f"voyage:{config.VOYAGE_MODEL}" if config.VOYAGE_API_KEY else "local-hash"


def _stem(w: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _tokens(text: str) -> list[str]:
    words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * config.EMBED_DIMS
    for tok in _tokens(text):
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        vec[h % config.EMBED_DIMS] += 1.0 if (h >> 63) else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    """input_type is "document" for memories and "query" for alerts."""
    if not config.VOYAGE_API_KEY:
        return [_hash_embed(t) for t in texts]
    resp = httpx.post(
        f"{config.VOYAGE_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {config.VOYAGE_API_KEY}"},
        json={
            "input": texts,
            "model": config.VOYAGE_MODEL,
            "input_type": input_type,
            "output_dimension": config.EMBED_DIMS,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda d: d["index"])]
