"""Retrieve top candidate topics for a post via centroid cosine similarity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.centroids import (  # noqa: E402
    configure_stdout,
    deduplicate_posts,
    get_embedding_with_retry,
    load_centroids,
    load_posts,
    safe_print,
)
from src.config import CENTROIDS_PATH, DATA_PATH  # noqa: E402
from src.post_text import build_embed_text  # noqa: E402

DEFAULT_THRESHOLD = 0.6134

HARDCODED_EXAMPLES: list[tuple[str, str]] = [
    (
        "شکست تیم ملی فوتبال",
        "شاگردان کی‌روش در بازی دوستانه مقابل ژاپن با نتیجه یک بر دو مغلوب شدند.",
    ),
    (
        "راهنمای پخت نان سنگک",
        "برای درست کردن نان سنگک خانگی به آرد کامل، آب ولرم و مایه خمیر نیاز دارید.",
    ),
]


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


def _print_results(title: str, candidates: list[dict], *, ground_truth: str | None = None) -> None:
    display_title = title if len(title) <= 80 else f"{title[:77]}..."
    safe_print()
    safe_print(f"Post: {display_title}")
    if ground_truth is not None:
        safe_print(f"Ground truth topics: {ground_truth}")
    for rank, candidate in enumerate(candidates, start=1):
        safe_print(
            f"  {rank}. {candidate['topic']:<20} "
            f"{candidate['similarity']:.4f}  {candidate['confidence']}"
        )


def main() -> None:
    configure_stdout()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    safe_print(f"Loaded {len(centroids)} centroids from {CENTROIDS_PATH}")

    for title, body in HARDCODED_EXAMPLES:
        candidates = retrieve_candidates(title, body, centroids)
        _print_results(title, candidates)

    df = deduplicate_posts(load_posts(DATA_PATH))
    row = df.iloc[0]
    candidates = retrieve_candidates(str(row["title"]), str(row["body"]), centroids)
    _print_results(
        str(row["title"]),
        candidates,
        ground_truth=str(row["topics"]),
    )


if __name__ == "__main__":
    main()
