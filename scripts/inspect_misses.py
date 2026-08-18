"""Diagnostic: print full cleaned body text for ورزشی retrieval misses (read-only)."""

from __future__ import annotations

import sys
from pathlib import Path

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
from src.preprocess import clean_text  # noqa: E402
from src.retrieve import retrieve_candidates  # noqa: E402

TARGET_TOPIC = "ورزشی"
SAMPLE_SIZE = 10
RANDOM_SEED = 42


def is_miss(candidates: list[dict], target_topic: str) -> bool:
    return target_topic not in [c["topic"] for c in candidates]


def print_miss(
    *,
    title: str,
    body: str,
    true_topics: list[str],
    candidates: list[dict],
) -> None:
    safe_print()
    safe_print("=" * 60)
    safe_print(f"MISS: {title}")
    safe_print("=" * 60)
    safe_print(f"True topics: {', '.join(true_topics)}")
    safe_print("Top-5 candidates:")
    for i, candidate in enumerate(candidates, start=1):
        safe_print(
            f"  {i}. {candidate['topic']:<20} "
            f"{candidate['similarity']:.4f}  {candidate['confidence']}"
        )
    safe_print("Body (cleaned):")
    safe_print(clean_text(body))


def main() -> None:
    configure_stdout()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    df = deduplicate_posts(load_posts(DATA_PATH))
    topic_lists = [parse_topics(t) for t in df["topics"]]

    mask = [TARGET_TOPIC in topics for topics in topic_lists]
    pool = df.loc[mask].reset_index(drop=True)
    sample_df = pool.sample(
        n=min(SAMPLE_SIZE, len(pool)), random_state=RANDOM_SEED
    ).reset_index(drop=True)

    safe_print(
        f"Inspecting {len(sample_df)} sampled posts for '{TARGET_TOPIC}' misses "
        f"(seed={RANDOM_SEED})..."
    )

    miss_count = 0
    for _, row in sample_df.iterrows():
        title = str(row["title"])
        body = str(row["body"])
        true_topics = parse_topics(row["topics"])

        candidates = retrieve_candidates(title, body, centroids)
        if not is_miss(candidates, TARGET_TOPIC):
            continue

        miss_count += 1
        print_miss(
            title=title,
            body=body,
            true_topics=true_topics,
            candidates=candidates,
        )

    safe_print()
    safe_print(f"Done. {miss_count} miss(es) printed. No data files were modified.")


if __name__ == "__main__":
    main()
