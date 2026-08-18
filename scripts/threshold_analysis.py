"""Diagnostic: compare own-topic vs random-topic centroid similarity to pick a retrieval threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.centroids import (  # noqa: E402
    deduplicate_posts,
    get_embedding_with_retry,
    load_centroids,
    load_posts,
    parse_topics,
)
from src.config import CENTROIDS_PATH, DATA_PATH  # noqa: E402
from src.post_text import build_embed_text  # noqa: E402

SAMPLE_SIZE = 400
RANDOM_TOPICS_PER_POST = 3
RANDOM_SEED = 42
PROGRESS_EVERY = 25


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), **kwargs)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


def filter_eligible_posts(df: pd.DataFrame, centroid_topics: set[str]) -> pd.DataFrame:
    """Keep posts with at least one true topic present in the centroid set."""

    def is_eligible(topics_value: object) -> bool:
        true_topics = parse_topics(topics_value)
        return bool(true_topics) and any(t in centroid_topics for t in true_topics)

    mask = df["topics"].map(is_eligible)
    return df.loc[mask].reset_index(drop=True)


def print_distribution(label: str, values: np.ndarray) -> None:
    safe_print()
    safe_print(f"--- {label} (n={len(values)}) ---")
    safe_print(f"  min:    {np.min(values):.4f}")
    safe_print(f"  max:    {np.max(values):.4f}")
    safe_print(f"  mean:   {np.mean(values):.4f}")
    safe_print(f"  median: {np.median(values):.4f}")
    p10, p25, p75, p90 = np.percentile(values, [10, 25, 75, 90])
    safe_print(f"  p10:    {p10:.4f}")
    safe_print(f"  p25:    {p25:.4f}")
    safe_print(f"  p75:    {p75:.4f}")
    safe_print(f"  p90:    {p90:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a cosine-similarity threshold for topic retrieval."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Number of posts to sample (default: {SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for sampling and negative topics (default: {RANDOM_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    all_topics = sorted(centroids.keys())
    centroid_topics = set(all_topics)
    safe_print(f"Loaded {len(centroids)} centroids from {CENTROIDS_PATH}")

    safe_print(f"CSV path: {DATA_PATH}")
    df = load_posts(DATA_PATH)
    safe_print(f"loaded rows: {len(df)}")

    df = deduplicate_posts(df)
    safe_print(f"unique posts after dedup: {len(df)}")

    df = filter_eligible_posts(df, centroid_topics)
    safe_print(f"eligible posts (non-empty topics in centroid set): {len(df)}")

    n_sample = min(args.sample_size, len(df))
    if n_sample == 0:
        safe_print("ERROR: no eligible posts to sample.")
        sys.exit(1)

    sample_df = df.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    safe_print(f"sampled posts: {n_sample} (seed={args.seed})")

    rng = np.random.default_rng(args.seed)
    own_sims: list[float] = []
    random_sims: list[float] = []
    post_max_sims: list[float] = []

    safe_print()
    safe_print(f"Embedding {n_sample} posts via Ollama (this may take a while)...")

    for i, row in sample_df.iterrows():
        text = build_embed_text(row["title"], row["body"])
        post_emb = get_embedding_with_retry(text)

        true_topics = [t for t in parse_topics(row["topics"]) if t in centroid_topics]
        for topic in true_topics:
            own_sims.append(cosine_sim(post_emb, centroids[topic]))

        negative_pool = [t for t in all_topics if t not in set(true_topics)]
        n_random = min(RANDOM_TOPICS_PER_POST, len(negative_pool))
        if n_random:
            chosen = rng.choice(negative_pool, size=n_random, replace=False)
            for topic in chosen:
                random_sims.append(cosine_sim(post_emb, centroids[topic]))

        all_sims = [cosine_sim(post_emb, centroids[t]) for t in all_topics]
        post_max_sims.append(max(all_sims))

        done = i + 1
        if done % PROGRESS_EVERY == 0 or done == n_sample:
            safe_print(f"  processed {done}/{n_sample}...")

    own_arr = np.array(own_sims, dtype=np.float64)
    random_arr = np.array(random_sims, dtype=np.float64)
    max_arr = np.array(post_max_sims, dtype=np.float64)

    safe_print()
    safe_print("=" * 60)
    safe_print("SIMILARITY DISTRIBUTIONS")
    safe_print("=" * 60)
    print_distribution("Own-topic similarities", own_arr)
    print_distribution("Random-topic similarities", random_arr)

    median_own = float(np.median(own_arr))
    median_random = float(np.median(random_arr))
    threshold = (median_own + median_random) / 2

    safe_print()
    safe_print("=" * 60)
    safe_print("SUGGESTED THRESHOLD")
    safe_print("=" * 60)
    safe_print(f"  median (own-topic):    {median_own:.4f}")
    safe_print(f"  median (random-topic): {median_random:.4f}")
    safe_print(f"  suggested threshold:   {threshold:.4f}  (midpoint of medians)")

    zero_candidates = int(np.sum(max_arr < threshold))
    zero_pct = 100.0 * zero_candidates / n_sample

    safe_print()
    safe_print("=" * 60)
    safe_print("*** ZERO-CANDIDATE WARNING ***")
    safe_print("=" * 60)
    safe_print(
        f"  {zero_candidates} / {n_sample} sampled posts ({zero_pct:.1f}%) "
        f"would have ZERO retrieval candidates above the suggested threshold."
    )
    if zero_pct > 20:
        safe_print("  >>> Threshold may be too strict; consider a fallback rule in retrieve.py.")
    elif zero_pct > 10:
        safe_print("  >>> Moderate zero-candidate rate; plan a fallback for edge cases.")
    else:
        safe_print("  >>> Zero-candidate rate looks reasonable for threshold-based retrieval.")

    safe_print()
    safe_print("Done. No data files were modified.")


if __name__ == "__main__":
    main()
