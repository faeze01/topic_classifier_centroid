"""Retrieve top candidate topics for a post via centroid cosine similarity."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.centroids import get_embedding_with_retry
from src.config import SECONDARY_MAX_GAP
from src.post_text import build_embed_text

DEFAULT_THRESHOLD = 0.6134


def retrieve_candidates(
    title: str,
    body: str,
    centroids: dict[str, np.ndarray],
    top_k: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Return top-k topic candidates with similarity scores and confidence labels."""
    if not centroids:
        return []

    text = build_embed_text(title, body)
    post_emb = get_embedding_with_retry(text)

    topics = list(centroids.keys())
    vectors = np.stack([centroids[t] for t in topics])
    sims = cosine_similarity(post_emb.reshape(1, -1), vectors)[0]

    k = min(top_k, len(topics))
    top_indices = np.argsort(sims)[::-1][:k]

    return [
        {
            "topic": topics[i],
            "similarity": float(sims[i]),
            "confidence": "strong" if sims[i] >= threshold else "weak",
        }
        for i in top_indices
    ]


def selected_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    chosen = [candidates[0]]
    sim1 = float(candidates[0]["similarity"])
    if len(candidates) > 1 and (sim1 - float(candidates[1]["similarity"])) < SECONDARY_MAX_GAP:
        chosen.append(candidates[1])
    if len(candidates) > 2 and (sim1 - float(candidates[2]["similarity"])) < SECONDARY_MAX_GAP:
        chosen.append(candidates[2])
    return chosen


def topics_from_scores(candidates: list[dict]) -> list[str]:
    return [str(c["topic"]) for c in selected_candidates(candidates)]
