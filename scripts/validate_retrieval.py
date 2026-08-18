"""Diagnostic: check retrieval hit rate and rank1-rank5 similarity gap on real posts (read-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.centroids import (  # noqa: E402
    configure_stdout,
    deduplicate_posts,
    load_centroids,
    load_posts,
    parse_topics,
    safe_print,
)
from src.config import CENTROIDS_PATH, DATA_PATH  # noqa: E402
from src.retrieve import retrieve_candidates  # noqa: E402

SAMPLE_SIZE = 30
RANDOM_SEED = 42
MAX_MISSES_SHOWN = 5


def main() -> None:
    configure_stdout()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    safe_print(f"Loaded {len(centroids)} centroids from {CENTROIDS_PATH}")

    df = deduplicate_posts(load_posts(DATA_PATH))
    safe_print(f"unique posts after dedup: {len(df)}")

    n_sample = min(SAMPLE_SIZE, len(df))
    sample_df = df.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)
    safe_print(f"sampled posts: {n_sample} (seed={RANDOM_SEED})")

    safe_print()
    safe_print(f"Running retrieve_candidates on {n_sample} posts via Ollama...")

    hits = 0
    gaps: list[float] = []
    misses: list[dict] = []

    for _, row in sample_df.iterrows():
        title = str(row["title"])
        body = str(row["body"])
        true_topics = parse_topics(row["topics"])

        candidates = retrieve_candidates(title, body, centroids)
        candidate_topics = [c["topic"] for c in candidates]

        is_hit = any(t in candidate_topics for t in true_topics)
        if is_hit:
            hits += 1
        else:
            misses.append(
                {
                    "title": title,
                    "true_topics": true_topics,
                    "top5": candidate_topics,
                }
            )

        if len(candidates) >= 2:
            gaps.append(candidates[0]["similarity"] - candidates[-1]["similarity"])

    safe_print()
    safe_print("=" * 60)
    safe_print("RESULTS")
    safe_print("=" * 60)
    safe_print(f"Hit rate: {hits}/{n_sample} ({100.0 * hits / n_sample:.1f}%)")

    if gaps:
        safe_print(f"Average rank1-rank5 similarity gap: {np.mean(gaps):.4f}")
    else:
        safe_print("Average rank1-rank5 similarity gap: N/A (no candidates returned)")

    safe_print()
    safe_print("=" * 60)
    safe_print(f"CLEAREST MISSES (up to {MAX_MISSES_SHOWN})")
    safe_print("=" * 60)
    if not misses:
        safe_print("No misses — every sampled post had a true topic in its top-5.")
    else:
        for i, miss in enumerate(misses[:MAX_MISSES_SHOWN], start=1):
            safe_print()
            safe_print(f"{i}. {miss['title']}")
            safe_print(f"   true topics: {', '.join(miss['true_topics'])}")
            safe_print(f"   top-5 candidates: {', '.join(miss['top5'])}")

    safe_print()
    safe_print("Done. No data files were modified.")


if __name__ == "__main__":
    main()
