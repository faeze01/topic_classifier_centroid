"""Diagnostic: evaluate retrieval quality on real posts for sports/food/tech topics (read-only)."""

from __future__ import annotations

import difflib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

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

TARGET_KEYWORDS = ["ورزش", "غذا", "تکنولوژی"]
SAMPLE_SIZE = 10
RANDOM_SEED = 42
PROGRESS_EVERY = 5
TITLE_TRUNC = 60
FUZZY_CUTOFF = 0.5
FUZZY_N = 5


@dataclass
class TopicResolution:
    keyword: str
    resolved_topic: str | None = None
    match_type: str | None = None
    candidates: list[str] = field(default_factory=list)


@dataclass
class TopicSummary:
    keyword: str
    resolved_topic: str | None
    total: int = 0
    hit_at_1: int = 0
    hit_at_5: int = 0
    missed: int = 0
    ranks: list[int] = field(default_factory=list)

    @property
    def avg_rank(self) -> float | None:
        if not self.ranks:
            return None
        return sum(self.ranks) / len(self.ranks)


def truncate_title(title: str, max_len: int = TITLE_TRUNC) -> str:
    if len(title) <= max_len:
        return title
    return f"{title[: max_len - 3]}..."


def collect_unique_topics(topic_lists: list[list[str]]) -> list[str]:
    return sorted({topic for topics in topic_lists for topic in topics})


def resolve_keyword(keyword: str, unique_topics: list[str]) -> TopicResolution:
    """Resolve a keyword to an actual dataset topic name without guessing."""
    resolution = TopicResolution(keyword=keyword)

    substring_matches = [
        topic
        for topic in unique_topics
        if keyword in topic or topic in keyword
    ]

    if len(substring_matches) == 1:
        resolution.resolved_topic = substring_matches[0]
        resolution.match_type = "substring"
        safe_print(
            f"Resolved '{keyword}' -> '{resolution.resolved_topic}' (substring match)"
        )
        return resolution

    if len(substring_matches) > 1:
        resolution.candidates = substring_matches
        safe_print(f"WARNING: ambiguous substring matches for '{keyword}':")
        for candidate in substring_matches:
            safe_print(f"  - {candidate}")
        return resolution

    fuzzy_matches = difflib.get_close_matches(
        keyword, unique_topics, n=FUZZY_N, cutoff=FUZZY_CUTOFF
    )
    if len(fuzzy_matches) == 1:
        resolution.resolved_topic = fuzzy_matches[0]
        resolution.match_type = "fuzzy"
        safe_print(
            f"Resolved '{keyword}' -> '{resolution.resolved_topic}' "
            f"(NOTE: fuzzy match only, please verify)"
        )
        return resolution

    resolution.candidates = fuzzy_matches
    safe_print(f"WARNING: no reliable match for '{keyword}'")
    if fuzzy_matches:
        safe_print("  fuzzy candidates:")
        for candidate in fuzzy_matches:
            safe_print(f"  - {candidate}")
    else:
        safe_print("  no fuzzy candidates found; all unique topics:")
        for topic in unique_topics:
            safe_print(f"  - {topic}")
    return resolution


def sample_posts_for_topic(
    df: pd.DataFrame,
    topic_lists: list[list[str]],
    topic: str,
    n: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    mask = [topic in topics for topics in topic_lists]
    pool = df.loc[mask].reset_index(drop=True)
    if pool.empty:
        safe_print(f"WARNING: no posts found for topic '{topic}'")
        return pool
    if len(pool) < n:
        safe_print(
            f"WARNING: only {len(pool)} posts available for '{topic}', sampling all of them"
        )
    return pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)


def find_target_rank(candidates: list[dict], target_topic: str) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate["topic"] == target_topic:
            return rank
    return None


def print_post_result(
    *,
    title: str,
    body: str,
    true_topics: list[str],
    target_topic: str,
    candidates: list[dict],
) -> int | None:
    body_len = len(str(body))
    rank = find_target_rank(candidates, target_topic)
    rank_display = str(rank) if rank is not None else "not in top-5"

    safe_print()
    safe_print(f"Title: {truncate_title(title)}")
    safe_print(f"Body length: {body_len} chars")
    safe_print(f"True topics: {', '.join(true_topics)}")
    safe_print("Top-5 candidates:")
    for i, candidate in enumerate(candidates, start=1):
        safe_print(
            f"  {i}. {candidate['topic']:<20} "
            f"{candidate['similarity']:.4f}  {candidate['confidence']}"
        )
    safe_print(f"Target topic rank: {rank_display}")
    return rank


def print_topic_summary(summary: TopicSummary) -> None:
    safe_print()
    safe_print("=" * 60)
    label = summary.resolved_topic or summary.keyword
    safe_print(f"SUMMARY for '{label}' (keyword: {summary.keyword})")
    safe_print("=" * 60)
    safe_print(f"  hit@1: {summary.hit_at_1}/{summary.total}")
    safe_print(f"  hit@5: {summary.hit_at_5}/{summary.total}")
    safe_print(f"  missed: {summary.missed}/{summary.total}")
    if summary.avg_rank is not None:
        safe_print(f"  average rank (when found): {summary.avg_rank:.2f}")
    else:
        safe_print("  average rank (when found): N/A")


def print_overall_table(summaries: list[TopicSummary]) -> None:
    safe_print()
    safe_print("=" * 72)
    safe_print("OVERALL SUMMARY")
    safe_print("=" * 72)
    safe_print(
        f"{'Keyword':<14} {'Resolved Topic':<22} {'Hit@1':<14} {'Hit@5':<14} {'Avg Rank'}"
    )
    safe_print("-" * 72)

    for summary in summaries:
        if summary.resolved_topic is None:
            safe_print(
                f"{summary.keyword:<14} {'SKIPPED':<22} {'-':<14} {'-':<14} {'-'}"
            )
            continue

        hit1 = f"{summary.hit_at_1}/{summary.total} ({100.0 * summary.hit_at_1 / summary.total:.0f}%)"
        hit5 = f"{summary.hit_at_5}/{summary.total} ({100.0 * summary.hit_at_5 / summary.total:.0f}%)"
        avg_rank = f"{summary.avg_rank:.2f}" if summary.avg_rank is not None else "N/A"
        safe_print(
            f"{summary.keyword:<14} {summary.resolved_topic:<22} {hit1:<14} {hit5:<14} {avg_rank}"
        )


def main() -> None:
    configure_stdout()

    if not CENTROIDS_PATH.exists():
        safe_print(f"ERROR: centroids file not found: {CENTROIDS_PATH}")
        sys.exit(1)

    centroids = load_centroids(CENTROIDS_PATH)
    safe_print(f"Loaded {len(centroids)} centroids from {CENTROIDS_PATH}")

    df = deduplicate_posts(load_posts(DATA_PATH))
    safe_print(f"unique posts after dedup: {len(df)}")

    topic_lists = [parse_topics(t) for t in df["topics"]]
    unique_topics = collect_unique_topics(topic_lists)
    safe_print(f"unique topics in dataset: {len(unique_topics)}")

    safe_print()
    safe_print("=" * 60)
    safe_print("TOPIC RESOLUTION")
    safe_print("=" * 60)

    resolutions = [resolve_keyword(keyword, unique_topics) for keyword in TARGET_KEYWORDS]
    resolved_items = [
        (resolution.keyword, resolution.resolved_topic)
        for resolution in resolutions
        if resolution.resolved_topic is not None
    ]

    total_posts = len(resolved_items) * SAMPLE_SIZE
    processed = 0
    summaries: list[TopicSummary] = []

    if not resolved_items:
        safe_print()
        safe_print("ERROR: no keywords resolved to dataset topics; nothing to evaluate.")
        print_overall_table(
            [
                TopicSummary(keyword=resolution.keyword, resolved_topic=None)
                for resolution in resolutions
            ]
        )
        sys.exit(1)

    safe_print()
    safe_print("=" * 60)
    safe_print(f"RETRIEVAL EVALUATION ({total_posts} posts via Ollama)")
    safe_print("=" * 60)

    for keyword, resolved_topic in resolved_items:
        safe_print()
        safe_print("-" * 60)
        safe_print(f"Evaluating topic '{resolved_topic}' (keyword: '{keyword}')")
        safe_print("-" * 60)

        sample_df = sample_posts_for_topic(df, topic_lists, resolved_topic)
        summary = TopicSummary(keyword=keyword, resolved_topic=resolved_topic)

        if sample_df.empty:
            summaries.append(summary)
            print_topic_summary(summary)
            continue

        for _, row in sample_df.iterrows():
            title = str(row["title"])
            body = str(row["body"])
            true_topics = parse_topics(row["topics"])

            candidates = retrieve_candidates(title, body, centroids)
            rank = print_post_result(
                title=title,
                body=body,
                true_topics=true_topics,
                target_topic=resolved_topic,
                candidates=candidates,
            )

            summary.total += 1
            if rank == 1:
                summary.hit_at_1 += 1
                summary.hit_at_5 += 1
                summary.ranks.append(rank)
            elif rank is not None:
                summary.hit_at_5 += 1
                summary.ranks.append(rank)
            else:
                summary.missed += 1

            processed += 1
            if processed % PROGRESS_EVERY == 0 or processed == total_posts:
                safe_print()
                safe_print(f"processed {processed}/{total_posts}...")

        summaries.append(summary)
        print_topic_summary(summary)

    for resolution in resolutions:
        if resolution.resolved_topic is None and not any(
            s.keyword == resolution.keyword for s in summaries
        ):
            summaries.append(
                TopicSummary(keyword=resolution.keyword, resolved_topic=None)
            )

    summaries.sort(key=lambda s: TARGET_KEYWORDS.index(s.keyword))
    print_overall_table(summaries)

    safe_print()
    safe_print("Done. No data files were modified.")


if __name__ == "__main__":
    main()
