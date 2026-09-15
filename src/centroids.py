"""Build one centroid vector per topic by averaging embeddings of posts labeled with that topic."""

from __future__ import annotations

import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import CENTROIDS_PATH, DATA_PATH, EMBEDDINGS_PATH
from src.embeddings import get_embedding, get_embeddings
from src.post_text import build_embed_text

PROGRESS_EVERY = 100
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5
BATCH_SIZE = 16
PRIMARY_COUNT_THRESHOLD = 1000
MAX_REQUESTS_PER_SECOND = 100
RATE_WINDOW_SECONDS = 1.0
TIMEOUT_RETRY_SECONDS = 20.0

_request_times: deque[float] = deque()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def safe_print(*args, **kwargs) -> None:
    """Print without crashing on Windows consoles that cannot encode Persian text."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), **kwargs)


def raise_field_size_limit() -> None:
    """Raise csv field size limit (needed for large HTML body fields on python engine)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)
            if limit < 128 * 1024:
                csv.field_size_limit(10**7)
                return


def load_posts(path: Path) -> pd.DataFrame:
    """Load the posts CSV, skipping malformed lines."""
    raise_field_size_limit()
    return pd.read_csv(
        path,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
    )


def deduplicate_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate post_id rows (topics is identical across duplicates)."""
    return df.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)


def parse_topics(topics_value: object) -> list[str]:
    """Split a comma-separated topics string into a clean list of names."""
    if topics_value is None or (isinstance(topics_value, float) and pd.isna(topics_value)):
        return []
    return [t.strip() for t in str(topics_value).split(",") if t.strip()]


def _wait_for_rate_limit() -> None:
    """Stay at or below MAX_REQUESTS_PER_SECOND HTTP calls (Arvan throttle)."""
    now = time.monotonic()
    while _request_times and now - _request_times[0] >= RATE_WINDOW_SECONDS:
        _request_times.popleft()
    if len(_request_times) >= MAX_REQUESTS_PER_SECOND:
        sleep_for = RATE_WINDOW_SECONDS - (now - _request_times[0]) + 0.05
        if sleep_for > 0:
            time.sleep(sleep_for)
        now = time.monotonic()
        while _request_times and now - _request_times[0] >= RATE_WINDOW_SECONDS:
            _request_times.popleft()
    _request_times.append(time.monotonic())


def _is_rate_or_timeout(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "rate" in text
        or "timeout" in text
        or "timed out" in text
        or "too many" in text
    )


def get_embedding_with_retry(text: str) -> np.ndarray:
    """Call get_embedding, retrying on transient network errors.

    If the model reports the input still exceeds its context length, truncate
    further and retry immediately (no delay needed for this case).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _wait_for_rate_limit()
            return get_embedding(text)
        except Exception as exc:  # noqa: BLE001 - network/server errors
            if attempt == MAX_RETRIES:
                raise
            if "context length" in str(exc).lower():
                text = text[: len(text) // 2]
                safe_print(f"  retry {attempt}/{MAX_RETRIES} after context-length error; truncated to {len(text)} chars")
            elif _is_rate_or_timeout(exc):
                safe_print(f"  retry {attempt}/{MAX_RETRIES} after throttle/timeout; sleep {TIMEOUT_RETRY_SECONDS:.0f}s")
                time.sleep(TIMEOUT_RETRY_SECONDS)
            else:
                safe_print(f"  retry {attempt}/{MAX_RETRIES} after error: {exc}")
                time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("unreachable")  # for type checkers; loop always returns or raises


def load_saved_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Load {post_id: vector} from the persistent embeddings file."""
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    ids = [str(i) for i in data["post_ids"].tolist()]
    vectors = data["embeddings"]
    return {pid: vectors[i] for i, pid in enumerate(ids)}


def save_saved_embeddings(mapping: dict[str, np.ndarray], path: Path) -> None:
    """Write the full {post_id: vector} map. Never deleted after a successful run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(mapping.keys())
    vectors = np.stack([mapping[pid] for pid in ids]).astype(np.float32)
    np.savez(path, post_ids=np.array(ids, dtype=object), embeddings=vectors)


def _embed_batch_with_retry(texts: list[str]) -> np.ndarray:
    working = list(texts)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _wait_for_rate_limit()
            return get_embeddings(working)
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                raise
            if "context length" in str(exc).lower():
                working = [t[: max(1, len(t) // 2)] for t in working]
                safe_print(
                    f"  retry {attempt}/{MAX_RETRIES} after context-length error; truncated batch"
                )
            elif _is_rate_or_timeout(exc):
                safe_print(
                    f"  retry {attempt}/{MAX_RETRIES} after throttle/timeout; sleep {TIMEOUT_RETRY_SECONDS:.0f}s"
                )
                time.sleep(TIMEOUT_RETRY_SECONDS)
            else:
                safe_print(f"  retry {attempt}/{MAX_RETRIES} after error: {exc}")
                time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError("unreachable")


def embed_posts(post_ids: list[str], texts: list[str]) -> np.ndarray:
    """Embed each post, saving every batch to EMBEDDINGS_PATH and resuming by post_id."""
    saved = load_saved_embeddings(EMBEDDINGS_PATH)
    missing = [(pid, text) for pid, text in zip(post_ids, texts) if pid not in saved]
    already = len(post_ids) - len(missing)
    if already:
        safe_print(f"Resuming embeddings: {already}/{len(post_ids)} already saved.")
    if missing:
        safe_print(
            f"Embedding {len(missing)} posts via cloud API "
            f"(max {MAX_REQUESTS_PER_SECOND} requests/sec, batch {BATCH_SIZE})..."
        )

    done_new = 0
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        batch_texts = [item[1] for item in batch]
        try:
            vectors = _embed_batch_with_retry(batch_texts)
        except Exception as exc:  # noqa: BLE001
            safe_print(f"  batch failed ({exc}); falling back to one-by-one")
            vectors = np.stack([get_embedding_with_retry(t) for t in batch_texts])
        for (pid, _), vector in zip(batch, vectors):
            saved[pid] = np.asarray(vector, dtype=np.float32)
        done_new += len(batch)
        total_done = already + done_new
        if total_done % PROGRESS_EVERY < BATCH_SIZE or done_new == len(missing):
            safe_print(f"Embedded {total_done}/{len(post_ids)}...")
            save_saved_embeddings(saved, EMBEDDINGS_PATH)

    if missing:
        save_saved_embeddings(saved, EMBEDDINGS_PATH)
    else:
        safe_print(f"All {len(post_ids)} embeddings already in {EMBEDDINGS_PATH}")

    return np.stack([saved[pid] for pid in post_ids]).astype(np.float32)


def raw_topic_counts(topic_lists: list[list[str]]) -> dict[str, int]:
    """Count posts that carry each topic, regardless of rank."""
    counts: dict[str, int] = {}
    for topics in topic_lists:
        for topic in topics:
            counts[topic] = counts.get(topic, 0) + 1
    return counts


def build_topic_index(
    topic_lists: list[list[str]],
    *,
    primary_threshold: int = PRIMARY_COUNT_THRESHOLD,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Map each topic to post indices used for its centroid.

    Topics with more than `primary_threshold` labeled posts only include posts
    where that topic is first in the label list. Rarer topics keep every post
    that carries the label.
    """
    counts = raw_topic_counts(topic_lists)
    topic_index: dict[str, list[int]] = {}
    for i, topics in enumerate(topic_lists):
        if not topics:
            continue
        first = topics[0]
        for topic in topics:
            if counts.get(topic, 0) > primary_threshold and topic != first:
                continue
            topic_index.setdefault(topic, []).append(i)
    return topic_index, counts


def compute_centroids(
    embeddings: np.ndarray, topic_index: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    """Compute the mean embedding (centroid) for each topic."""
    centroids: dict[str, np.ndarray] = {}
    for topic, indices in topic_index.items():
        if not indices:
            continue
        centroids[topic] = embeddings[indices].mean(axis=0).astype(np.float32)
    return centroids


def save_centroids(centroids: dict[str, np.ndarray], path: Path) -> None:
    """Save {topic: vector} to an NPZ file as two aligned arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    topics = np.array(list(centroids.keys()), dtype=object)
    vectors = np.stack(list(centroids.values()))
    np.savez(path, topics=topics, centroids=vectors)


def load_centroids(path: Path) -> dict[str, np.ndarray]:
    """Load {topic: vector} back from an NPZ file saved by save_centroids."""
    data = np.load(path, allow_pickle=True)
    return dict(zip(data["topics"].tolist(), data["centroids"]))


def print_topic_report(
    topic_index: dict[str, list[int]], raw_counts: dict[str, int]
) -> None:
    """Print raw label counts vs posts actually used in each centroid."""
    safe_print()
    safe_print(f"Centroid posts (threshold {PRIMARY_COUNT_THRESHOLD}; fewest -> most used):")
    for topic, indices in sorted(topic_index.items(), key=lambda kv: len(kv[1])):
        raw = raw_counts.get(topic, 0)
        used = len(indices)
        flag = "  [first-only]" if raw > PRIMARY_COUNT_THRESHOLD else ""
        safe_print(f"  {topic}: {used} used / {raw} labeled{flag}")
    safe_print(f"Total topics: {len(topic_index)}")


def main() -> None:
    configure_stdout()

    safe_print(f"CSV path: {DATA_PATH}")
    df = load_posts(DATA_PATH)
    safe_print(f"loaded rows: {len(df)}")

    df = deduplicate_posts(df)
    safe_print(f"unique posts after dedup: {len(df)}")

    post_ids = [str(pid) for pid in df["post_id"].tolist()]
    topic_lists = [parse_topics(t) for t in df["topics"]]
    texts = [build_embed_text(t, b) for t, b in zip(df["title"], df["body"])]

    embeddings = embed_posts(post_ids, texts)
    safe_print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    topic_index, raw_counts = build_topic_index(topic_lists)
    centroids = compute_centroids(embeddings, topic_index)

    save_centroids(centroids, CENTROIDS_PATH)
    safe_print(f"Saved {len(centroids)} centroids to {CENTROIDS_PATH}")

    print_topic_report(topic_index, raw_counts)


if __name__ == "__main__":
    main()
